# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic ViT on JFT-300M."""

from functools import partial  # pylint: disable=g-importing-member so standard
import importlib
import multiprocessing
import numbers
import os

from absl import app
from absl import flags
from absl import logging
from clu import parameter_overview
from clu import periodic_actions
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from tensorflow.io import gfile

import dune.experimental.big_vision.fewshot as fewshot
import dune.experimental.big_vision.input_pipeline as input_pipeline
import dune.experimental.big_vision.models.argus.pp_ops  # pylint: disable=unused-import

import dune.experimental.big_vision.utils as u
import dune.task_adapt.data.preprocess  # pylint: disable=unused-import
import dune.tools.components.preprocess.builder as pp_builder


# TODO(lbeyer): make nice and dynamic. Used for configurable imports.
BASEDIR = "dune.experimental.big_vision"


ml_collections.config_flags.DEFINE_config_file(
    "config", None, "Training configuration.", lock_config=True)

flags.DEFINE_string("workdir", default=None, help="Work unit directory.")

FLAGS = flags.FLAGS

# Adds jax flags to the program.
jax.config.parse_flags_with_absl()


def main(argv):
  del argv

  config = FLAGS.config
  workdir = FLAGS.workdir

  if config.get("dataset_dir"):
    logging.info("data_dir=%s", config.dataset_dir)
  logging.info("Workdir: %s", workdir)

  save_checkpoint_path = None
  if config.get("checkpoint_steps"):
    gfile.makedirs(workdir)
    save_checkpoint_path = os.path.join(workdir, "checkpoint.npz")

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  # This seed makes the Jax part of things (like model init) deterministic.
  # However, full training still won't be deterministic, for example due to the
  # tf.data pipeline not being deterministic even if we would set TF seed.
  rng = jax.random.PRNGKey(config.get("seed", 0))

  def write_note(note):
    if jax.host_id() == 0:
      logging.info("NOTE: %s", note)
  write_note("Initializing...")

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get("keep_checkpoint_steps"):
    assert config.get("checkpoint_steps"), "Specify `checkpoint_steps`."
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
        f"`keep_checkpoint_steps` ({config.checkpoint_steps}) should be"
        f"divisible by `checkpoint_steps ({config.checkpoint_steps}).`")

  batch_size = config.batch_size
  batch_size_eval = config.get("batch_size_eval", batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(f"Batch sizes ({batch_size} and {batch_size_eval}) must "
                     f"be divisible by device number ({jax.device_count()})")

  local_batch_size = batch_size // jax.host_count()
  local_batch_size_eval = batch_size_eval // jax.host_count()
  logging.info(
      "Global batch size %d on %d hosts results in %d local batch size. "
      "With %d dev per host (%d dev total), that's a %d per-device batch size.",
      batch_size, jax.host_count(), local_batch_size,
      jax.local_device_count(), jax.device_count(),
      local_batch_size // jax.local_device_count())

  write_note("Initializing train dataset...")
  train_ds = input_pipeline.get_data(
      dataset=config.dataset,
      split=config.train_split,
      data_dir=fillin(config.get("dataset_dir")),
      batch_size=local_batch_size,
      preprocess_fn=pp_builder.get_preprocess_fn(config.pp_train),
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch=config.get("prefetch_to_host", 2),
      cache=False)

  # Start prefetching already.
  train_iter = u.start_input_pipeline(
      train_ds, config.get("prefetch_to_device", 1), pad=local_batch_size)
  # We always pad to local_batch_size_eval even when less would be enough in
  # order to minimize memory fragmentation.

  write_note("Initializing val dataset(s)...")
  def _get_val_split(dataset, split, pp_eval, data_dir=None):
    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_pipeline.get_num_examples(
        dataset, split, data_dir=fillin(data_dir))
    val_steps = int(np.ceil(nval_img / batch_size_eval))
    logging.info("Running validation for %d steps for %s, %s", val_steps,
                 dataset, split)

    val_it = input_pipeline.get_data(
        dataset=dataset,
        split=split,
        data_dir=fillin(data_dir),
        batch_size=local_batch_size_eval,
        preprocess_fn=pp_builder.get_preprocess_fn(pp_eval),
        cache=config.get("val_cache", "batched"),
        repeat_after_batching=True,
        prefetch=0,  # Save memory since we cache.
        drop_remainder=False,
        shuffle_files=False)
    val_it = u.start_input_pipeline(
        val_it, config.get("prefetch_to_device", 1), pad=local_batch_size_eval)

    return (val_it, val_steps)

  if isinstance(config.val_split, str):
    val_ds = {"val": _get_val_split(config.dataset, config.val_split,
                                    config.pp_eval, config.get("dataset_dir"))}
  else:
    val_ds = {t[0]: _get_val_split(*t[1:]) for t in config.val_split}

  ntrain_img = input_pipeline.get_num_examples(
      config.dataset, config.train_split,
      data_dir=fillin(config.get("dataset_dir")))
  steps_per_epoch = ntrain_img / batch_size

  if config.get("num_epochs"):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get("total_steps"), "Set either num_epochs or total_steps"
  else:
    total_steps = config.total_steps

  logging.info(
      "Running for %d steps, that means %f epochs and %f steps per epoch",
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)
  mw = u.BigVisionMetricWriter(xm_xp.id, xm_wu.id, steps_per_epoch)

  write_note(f"Initializing {config.model_name} model...")
  model_mod = importlib.import_module(f"{BASEDIR}.models.{config.model_name}")
  model = model_mod.Model(
      num_classes=config.num_classes, **config.get("model", {}))

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend="cpu")
  def init(rng):
    image_size = tuple(train_ds.element_spec["image"].shape[1:])
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input))["params"]

    # Set bias in the head to a low value, such that loss is small initially.
    params["head"]["bias"] = jnp.full_like(
        params["head"]["bias"], config.get("init_head_bias", 0))

    return params

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  if jax.host_id() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    mw.measure("num_params", num_params)

  @partial(jax.pmap, axis_name="batch")
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    logits, _ = model.apply({"params": flax.core.freeze(params)}, images)

    losses = getattr(u, config.get("loss", "sigmoid_xent"))(
        logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name="batch")

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name="batch")
    n = jax.lax.psum(mask, axis_name="batch")
    return ncorrect, loss, n

  # Setup function for computing representation.
  @partial(jax.pmap, axis_name="batch")
  def representation_fn(params, images, labels, mask):
    _, outputs = model.apply({"params": flax.core.freeze(params)}, images)
    representation = outputs[config.fewshot.representation_layer]
    representation = jax.lax.all_gather(representation, "batch")
    labels = jax.lax.all_gather(labels, "batch")
    mask = jax.lax.all_gather(mask, "batch")
    return representation, labels, mask

  # Load the optimizer either from our folder or from flax.
  opt_name = config.get("optim_name", "momentum_hp")
  write_note(f"Initializing {opt_name} optimizer...")
  try:
    opt_mod = importlib.import_module(f"{BASEDIR}.optims.{opt_name}")
    opt_def = opt_mod.Optimizer(**config.get("optim", {}))
  except ModuleNotFoundError:
    opt_def = getattr(flax.optim, opt_name)(**config.get("optim", {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
  def update_fn(opt, lr, images, labels, rng):
    """Update step."""

    measurements = {}

    if config.get("mixup") and config.mixup.p:
      rng, (images, labels), _ = u.mixup(rng, images, labels, **config.mixup)

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      logits, _ = model.apply(
          {"params": flax.core.freeze(params)}, images,
          train=True, rngs={"dropout": rng_model_local})
      return getattr(u, config.get("loss", "sigmoid_xent"))(
          logits=logits, labels=labels)

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    l, g = u.accumulate_gradient(jax.value_and_grad(loss_fn), opt.target,
                                 images, labels,
                                 config.get("grad_accum_steps"))
    l, g = jax.lax.pmean((l, g), axis_name="batch")

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get("grad_accum_steps", 1) == 1 or config.get("grad_clip_norm"):
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements["l2_grads"] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if config.get("grad_clip_norm"):
      g_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
      g = jax.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    decay_rules = config.get("weight_decay", []) or []
    if isinstance(decay_rules, numbers.Number):
      decay_rules = [(".*kernel.*", decay_rules)]
    sched_m = lr/config.lr.base if config.get("weight_decay_decouple") else lr
    def decay_fn(v, wd):
      return (1.0 - sched_m * wd) * v
    opt = opt.replace(target=u.tree_map_with_regex(
        decay_fn, opt.target, decay_rules, name="weight decay"))

    params, _ = jax.tree_flatten(opt.target)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))

    return opt, l, rng, measurements

  # Other things besides optimizer state to be stored.
  checkpoint_extra = dict(accum_train_time=0.0)

  # Decide how to initialize training. The order is important.
  # 1. Always resumes from the existing checkpoint, e.g. resumes a finetune job.
  # 2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  # 3. Initialize model from something, e,g, start a fine-tuning job.
  # 4. Train from scratch.
  resume_checkpoint_path = None
  if save_checkpoint_path and gfile.exists(save_checkpoint_path):
    resume_checkpoint_path = save_checkpoint_path
  elif config.get("resume"):
    resume_checkpoint_path = fillin(config.resume)
  if resume_checkpoint_path:
    write_note("Resume training from checkpoint...")
    checkpoint = {"opt": opt_cpu, "extra": checkpoint_extra}
    _, checkpoint_tree = jax.tree_flatten(checkpoint)
    loaded = u.load_checkpoint(checkpoint_tree, resume_checkpoint_path)
    # bfloat16 type gets lost when data is saved to disk, so we recover it.
    checkpoint = jax.tree_map(u.recover_dtype, loaded)
    opt_cpu, checkpoint_extra = checkpoint["opt"], checkpoint["extra"]
  elif config.get("model_init"):
    write_note(f"Initialize model from {config.model_init}...")
    loaded = model_mod.load(params_cpu, config.model_init, config.get("model"))
    opt_cpu = opt_cpu.replace(target=loaded)
    if jax.host_id() == 0:
      logging.info("Restored parameter overview:")
      parameter_overview.log_parameter_overview(loaded)

  write_note("Kicking off misc stuff...")
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  chrono = u.Chrono(first_step, total_steps, batch_size,
                    checkpoint_extra["accum_train_time"])
  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=workdir, first_profile=first_step + 10)

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = u.create_learning_rate_schedule(
      batch_size, total_steps, steps_per_epoch, **config.get("lr", {}))
  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  lr_iter = u.prefetch_scalar(map(lr_fn, range(first_step, total_steps)),
                              config.get("prefetch_to_device", 1))

  write_note(f"Replicating...\n{chrono.note}")
  opt_repl = flax_utils.replicate(opt_cpu)

  write_note(f"Initializing few-shotters...\n{chrono.note}")
  if "fewshot" in config:
    fewshotter = fewshot.FewShotEvaluator(
        representation_fn, config.fewshot,
        config.fewshot.get("batch_size") or batch_size_eval)

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax_utils.replicate(rng_loop)
  checkpoint_writer = None

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  train_loss = -jnp.inf
  val_loss = -jnp.inf
  best_l2 = {1: -jnp.inf}

  write_note(f"First step compilations...\n{chrono.note}")
  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch, lr_repl in zip(
      range(first_step + 1, total_steps + 1), train_iter, lr_iter):
    mw.step_start(step)

    with jax.profiler.TraceContext("train_step", step_num=step, _r=1):
      opt_repl, loss_value, rngs_loop, extra_measurements = update_fn(
          opt_repl,
          lr_repl,
          train_batch["image"],
          train_batch["labels"],
          rng=rngs_loop)

    if jax.host_id() == 0:
      profiler(step)

    # Checkpoint saving
    if u.itstime(step, config.get("checkpoint_steps"), total_steps, host=0):
      write_note("Checkpointing...")
      chrono.pause()
      u.checkpointing_timeout(checkpoint_writer,
                              config.get("checkpoint_timeout", 1))
      checkpoint_extra["accum_train_time"] = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if u.itstime(step, config.get("keep_checkpoint_steps"), total_steps):
        write_note("Keeping a checkpoint copy...")
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint = {"opt": opt_cpu, "extra": checkpoint_extra}
      checkpoint_writer = pool.apply_async(
          u.save_checkpoint, (checkpoint, save_checkpoint_path, copy_step))
      chrono.resume()

    # Report training progress
    if u.itstime(step, config.log_training_steps, total_steps, host=0):
      write_note("Reporting training progress...")
      train_loss = loss_value[0]  # Keep to return for reproducibility tests.
      mw.measure("learning_rate", lr_repl[0])
      mw.measure("training_loss", loss_value[0])
      for name, value in extra_measurements.items():
        mw.measure(name, value[0])
      chrono.tick(step, mw.measure, write_note)

    # Report validation performance
    if u.itstime(step, config.log_eval_steps, total_steps):
      write_note("Evaluating on the validation set...")
      chrono.pause()
      for val_name, (val_iter, val_steps) in val_ds.items():
        ncorrect, loss, nseen = 0, 0, 0
        for _, batch in zip(range(val_steps), val_iter):
          batch_ncorrect, batch_losses, batch_n = evaluation_fn(
              opt_repl.target, batch["image"], batch["labels"], batch["mask"])
          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.
          ncorrect += np.sum(np.array(batch_ncorrect[0]))
          loss += np.sum(np.array(batch_losses[0]))
          nseen += np.sum(np.array(batch_n[0]))
        val_loss = loss / nseen  # Keep to return for reproducibility tests.
        mw.measure(f"{val_name}_prec@1", ncorrect / nseen)
        mw.measure(f"{val_name}_loss", val_loss)
      chrono.resume()

    if "fewshot" in config:
      # Compute few-shot on-the-fly evaluation.
      if u.itstime(step, config.fewshot.log_steps, total_steps):
        chrono.pause()
        write_note(f"Few-shot evaluation...\n{chrono.note}")
        # Keep `best_l2` to return for reproducibility tests.
        results, best_l2 = fewshotter.run_all(opt_repl.target,
                                              config.fewshot.datasets)
        fewshotter.walk_results(mw.measure, results, best_l2)
        chrono.resume()
    mw.step_end()

  write_note(f"Done!\n{chrono.note}")
  pool.close()
  pool.join()
  mw.close()

  # Return final training loss, validation loss, and fewshot best l2s for
  # reproducibility test cases.
  return train_loss, val_loss, best_l2


if __name__ == "__main__":
  app.run(main)
