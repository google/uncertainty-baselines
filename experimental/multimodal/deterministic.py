# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Deterministic CLIP."""
import collections
import functools
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
import flax
import jax
import jax.numpy as jnp
import ml_collections.config_flags
import numpy as np
import robustness_metrics as rm

import tensorflow as tf
import uncertainty_baselines as ub
# TODO(jallingham): Fork remaining utils once imports below merged into UB API.
import train_utils  # local file import from baselines.jft
# NOTE: Usually we do not allow cross-imports between subdirectories. We are
# doing so here because this is an experimental directory and the offending
# utils are soon to have much of their functionality merged into the UB API.
import checkpoint_utils  # local file import from experimental.multimodal
import input_utils  # local file import from experimental.multimodal
import multimodal_utils  # local file import from experimental.multimodal
import preprocess_utils  # local file import from experimental.multimodal
import simple_tokenizer  # local file import from experimental.multimodal

# TODO(dusenberrymw): Open-source remaining imports.
fewshot = None
# pylint: disable=g-bad-import-order,g-import-not-at-top
from big_vision.evaluators import fewshot
# NOTE: This import is still required in order to build the preprocessing
# registry for use in fewshot computation.
from big_vision.pp import ops_image  # pylint: disable=unused-import
from big_vision.pp import ops_general  # pylint: disable=unused-import

ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS


def main(config, output_dir):

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  if config.get('data_dir'):
    logging.info('data_dir=%s', config.data_dir)
  logging.info('Output dir: %s', output_dir)
  tf.io.gfile.makedirs(output_dir)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    save_checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing...')

  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get('keep_checkpoint_steps'):
    assert config.get('checkpoint_steps'), 'Specify `checkpoint_steps`.'
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
        f'`keep_checkpoint_steps` ({config.keep_checkpoint_steps}) should be'
        f'divisible by `checkpoint_steps ({config.checkpoint_steps}).`')

  batch_size = config.batch_size
  batch_size_eval = config.get('batch_size_eval', batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(f'Batch sizes ({batch_size} and {batch_size_eval}) must '
                     f'be divisible by device number ({jax.device_count()})')

  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = batch_size_eval // jax.process_count()
  logging.info(
      'Global batch size %d on %d hosts results in %d local batch size. '
      'With %d devices per host (%d devices total), that\'s a %d per-device '
      'batch size.', batch_size, jax.process_count(), local_batch_size,
      jax.local_device_count(), jax.device_count(),
      local_batch_size // jax.local_device_count())

  write_note('Initializing train dataset...')
  rng, train_ds_rng = jax.random.split(rng)
  train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())
  train_ds = input_utils.get_data(
      dataset=config.dataset,
      split=config.train_split,
      rng=train_ds_rng,
      process_batch_size=local_batch_size,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_train, available_ops=preprocess_utils.all_ops()),
      shuffle_buffer_size=config.shuffle_buffer_size,
      prefetch_size=config.get('prefetch_to_host', 2),
      data_dir=config.get('data_dir'),
      ignore_errors=True)

  write_note('Initializing val dataset(s)...')

  def _get_val_split(dataset, split, pp_eval, rng, data_dir=None):
    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_utils.get_num_examples(
        dataset,
        split=split,
        process_batch_size=local_batch_size_eval,
        drop_remainder=False,
        data_dir=data_dir)
    val_steps = int(np.ceil(nval_img / batch_size_eval))
    logging.info('Running validation for %d steps for %s, %s', val_steps,
                 dataset, split)

    if isinstance(pp_eval, str):
      pp_eval = preprocess_spec.parse(
          spec=pp_eval, available_ops=preprocess_utils.all_ops())

    rng = jax.random.fold_in(rng, jax.process_index())
    val_ds = input_utils.get_data(
        dataset=dataset,
        split=split,
        rng=rng,
        process_batch_size=local_batch_size_eval,
        preprocess_fn=pp_eval,
        cache=config.get('val_cache', 'batched'),
        num_epochs=1,
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=config.get('prefetch_to_host', 2),
        drop_remainder=False,
        data_dir=data_dir,
        ignore_errors=True)

    return val_ds

  rng, val_ds_rng = jax.random.split(rng)
  eval_ds_splits = {
      'val':
          _get_val_split(
              config.dataset,
              split=config.val_split,
              pp_eval=config.pp_eval,
              rng=val_ds_rng,
              data_dir=config.get('data_dir'))
  }

  if config.get('test_split'):
    rng, test_ds_rng = jax.random.split(rng)
    eval_ds_splits.update({
        'test':
            _get_val_split(
                config.dataset,
                split=config.test_split,
                pp_eval=config.pp_eval,
                rng=test_ds_rng,
                data_dir=config.get('data_dir'))
    })

  zeroshot_ds_splits = {}
  for zeroshot_name, zeroshot_config in config.get('zeroshot_eval_datasets',
                                                   {}).items():
    rng, zeroshot_ds_rng = jax.random.split(rng)
    preprocess_fn = preprocess_spec.parse(
        spec=zeroshot_config['pp_spec'],
        available_ops=preprocess_utils.all_ops())
    zeroshot_ds_splits.update({
        f'zeroshot_{zeroshot_name}':
            _get_val_split(
                zeroshot_config['dataset'],
                split=zeroshot_config['split'],
                pp_eval=preprocess_fn,
                rng=zeroshot_ds_rng,
                data_dir=config.get('data_dir'))
    })

  ntrain_img = input_utils.get_num_examples(
      config.dataset,
      split=config.train_split,
      process_batch_size=local_batch_size,
      data_dir=config.get('data_dir'))
  steps_per_epoch = ntrain_img // batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info('Total train data points: %d', ntrain_img)
  logging.info(
      'Running for %d steps, that means %f epochs and %d steps per epoch',
      total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)

  write_note('Initializing model...')
  logging.info('config.model = %s', config.model)
  model = ub.models.clip(**config.model)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    text_size = tuple(train_ds.element_spec['text'].shape[2:])
    logging.info('image_size = %s', image_size)
    logging.info('text_size = %s', text_size)
    dummy_image = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    dummy_text = jnp.zeros((local_batch_size,) + text_size, jnp.int32)
    variables = model.init(rng, dummy_image, dummy_text)  # flax.core.unfreeze()
    # Split model parameters into trainable and untrainable collections.
    states, params = variables.pop('params')
    params = flax.core.unfreeze(params)
    # del variables
    return params, states

  rng, rng_init = jax.random.split(rng)
  params_cpu, states_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    writer.write_scalars(step=0, scalars={'num_params': num_params})

  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, states, images, texts, mask):
    variable_dict = {'params': flax.core.freeze(params), **states}
    zimg, ztext = model.apply(variable_dict, images, texts, scale_logits=True)

    losses, logits = multimodal_utils.bidirectional_contrastive_loss(
        zimg, ztext, reduction=False)

    loss = jax.lax.psum(losses * mask, axis_name='batch')

    labels = jnp.eye(len(logits), dtype=logits.dtype)
    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = (top1_idx == jnp.arange(len(logits))).astype(jnp.float32)
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, mask], axis_name='batch')
    return ncorrect, loss, n, metric_args

  def make_zeroshot_evaluation_fn(params, states, classnames_key, prompts_key):
    @jax.jit
    def encode_text(texts):
      variable_dict = {'params': flax.core.freeze(params), **states}
      return model.apply(
          variable_dict,
          texts,
          normalize=False,
          scale_logits=False,
          method=model.encode_text)

    templates = multimodal_utils.get_zeroshot_template(prompts_key)

    tokenizer = simple_tokenizer.SimpleTokenizer()
    tokenize_fn = simple_tokenizer.make_tokenize_fn(tokenizer,
                                                    config.tokenizer_max_len)

    logging.info('Make ztxt')
    ztxt = []
    for clsname in multimodal_utils.get_zeroshot_class_names(classnames_key):
      token_fn = lambda text: tokenize_fn(tf.constant(text, dtype=tf.string))
      texts = jnp.array(
          [token_fn(template.format(clsname)) for template in templates])
      class_embeddings = encode_text(texts)
      class_embedding = class_embeddings.mean(0)
      class_embedding *= jax.lax.rsqrt(jnp.sum(class_embedding**2))
      class_embedding *= jnp.sqrt(jnp.exp(params['logit_scale']))
      ztxt.append(class_embedding)
    ztxt = jnp.stack(ztxt, axis=1)
    logging.info('Done ztxt')

    @functools.partial(jax.pmap, axis_name='batch')
    def zeroshot_evaluation_fn(params, states, images, labels, mask):
      # Get zero-shot classifier logits.
      variable_dict = {'params': flax.core.freeze(params), **states}
      zimg = model.apply(variable_dict, images, method=model.encode_image)
      logits = jnp.dot(zimg, ztxt)

      losses = train_utils.softmax_xent(logits=logits, labels=labels)
      loss = jax.lax.psum(losses * mask, axis_name='batch')

      top1_idx = jnp.argmax(logits, axis=1)
      # Extracts the label at the highest logit index for each image.
      top1_correct = jnp.take_along_axis(
          labels, top1_idx[:, None], axis=1)[:, 0]
      ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
      n = jax.lax.psum(mask, axis_name='batch')

      metric_args = jax.lax.all_gather([logits, labels, mask],
                                       axis_name='batch')
      return ncorrect, loss, n, metric_args

    return zeroshot_evaluation_fn

  # Setup function for computing representation.
  @functools.partial(jax.pmap, axis_name='batch')
  def representation_fn(params, images, labels, mask):
    representation = model.apply({'params': flax.core.freeze(params)},
                                 images,
                                 method=model.encode_image)
    representation = jax.lax.all_gather(representation, 'batch')
    labels = jax.lax.all_gather(labels, 'batch')
    mask = jax.lax.all_gather(mask, 'batch')
    return representation, labels, mask

  # Load the optimizer from flax.
  opt_name = config.get('optim_name')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  weight_decay_rules = config.get('weight_decay', []) or []
  rescale_value = 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, states, lr, images, texts, rng):
    """Update step."""
    measurements = {}

    # Split rng and return next_rng for the following step.
    rng, next_rng = jax.random.split(rng, 2)

    def loss_fn(params, states, images, texts):
      variable_dict = {'params': flax.core.freeze(params), **states}
      (zimg, ztext), updated_states = model.apply(
          variable_dict, images, texts, mutable=list(states.keys()))

      loss, logits = multimodal_utils.bidirectional_contrastive_loss(
          zimg, ztext, reduction=True)
      return loss, (logits, updated_states)
    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    (l, (logits, states)), g = train_utils.accumulate_gradient_with_states(
        jax.value_and_grad(loss_fn, has_aux=True), opt.target, states, images,
        texts, config.get('grad_accum_steps'))
    l, g = jax.lax.pmean((l, g), axis_name='batch')
    measurements['training_loss'] = l
    measurements['logit_scale'] = opt.target['logit_scale']

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get('grad_accum_steps', 1) == 1 or config.get('grad_clip_norm'):
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if config.get('grad_clip_norm'):
      g_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
      g = jax.tree_util.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    opt = opt.replace(target=weight_decay_fn(opt.target, lr))

    params, _ = jax.tree_flatten(opt.target)
    measurements['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))

    top1_idx = jnp.argmax(logits, axis=1)
    top1_correct = top1_idx == jnp.arange(len(logits))
    prec1 = jax.lax.psum(jnp.sum(top1_correct), axis_name='batch') / batch_size
    measurements['training_prec@1'] = prec1
    measurements['learning_rate'] = lr
    return opt, states, next_rng, measurements

  reint_params = []
  checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
      train_loop_rngs=rng,
      save_checkpoint_path=save_checkpoint_path,
      init_optimizer=opt_cpu,
      init_params=params_cpu,
      init_fixed_model_states=states_cpu,
      default_reinit_params=reint_params,
      config=config)
  train_loop_rngs = checkpoint_data.train_loop_rngs
  opt_cpu = checkpoint_data.optimizer
  states_cpu = checkpoint_data.fixed_model_states
  accumulated_train_time = checkpoint_data.accumulated_train_time

  write_note('Kicking off misc stuff...')
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  logging.info('first_step = %s', first_step)
  if first_step == 0 and jax.process_index() == 0:
    writer.write_hparams(dict(config))

  chrono = train_utils.Chrono(
      first_step, total_steps, batch_size, accumulated_train_time)

  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir, first_profile=first_step + 10)

  # TODO(dusenberrymw): Remove manual replication by updating pmap axes.
  write_note(f'Replicating...\n{chrono.note}')
  opt_repl = flax.jax_utils.replicate(opt_cpu)
  states_repl = flax.jax_utils.replicate(states_cpu)

  write_note(f'Initializing few-shotters...\n{chrono.note}')
  fewshotter = None
  if 'fewshot' in config and fewshot is not None:
    fewshotter = fewshot.FewShotEvaluator(
        representation_fn, config.fewshot,
        config.fewshot.get('batch_size') or batch_size_eval)

  checkpoint_writer = None

  lr_fn = train_utils.create_learning_rate_schedule(total_steps,
                                                    **config.get('lr', {}))

  # Prefetch all iterators, starting at the current first step.
  if first_step > 0:
    write_note('Advancing the dataset after resuming from a checkpoint...')
    # TODO(dusenberrymw): Look into checkpointing dataset state instead.
    train_ds = train_ds.skip(first_step)

  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  train_iter = input_utils.start_input_pipeline(
      train_ds, config.get('prefetch_to_device', 1))
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(first_step, total_steps)),
      config.get('prefetch_to_device', 1))

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  train_loss = -jnp.inf
  val_loss = {val_name: -jnp.inf for val_name, _ in eval_ds_splits.items()}
  fewshot_results = {'dummy': {(0, 1): -jnp.inf}}

  write_note(f'First step compilations...\n{chrono.note}')
  for step in range(first_step + 1, total_steps + 1):
    with jax.profiler.StepTraceAnnotation('train_step', step_num=step):
      train_batch = next(train_iter)
      lr_repl = next(lr_iter)
      if not config.get('only_eval', False):
        opt_repl, states_repl, train_loop_rngs, extra_measurements = update_fn(
            opt_repl,
            states_repl,
            lr_repl,
            train_batch['image'],
            train_batch['text'],
            rng=train_loop_rngs)

    if jax.process_index() == 0:
      profiler(step)

    # Checkpoint saving.
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.get('checkpoint_steps'), total_steps, process=0):
      write_note('Checkpointing...')
      chrono.pause()
      train_utils.checkpointing_timeout(checkpoint_writer,
                                        config.get('checkpoint_timeout', 1))
      accumulated_train_time = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      opt_cpu = jax.tree_util.tree_map(lambda x: np.array(x[0]), opt_repl)
      states_cpu = jax.tree_map(lambda x: np.array(x[0]), states_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if train_utils.itstime(step, config.get('keep_checkpoint_steps'),
                             total_steps):
        write_note('Keeping a checkpoint copy...')
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint_data = checkpoint_utils.CheckpointData(
          train_loop_rngs=train_loop_rngs,
          optimizer=opt_cpu,
          fixed_model_states=states_cpu,
          accumulated_train_time=accumulated_train_time)

      checkpoint_writer = pool.apply_async(
          checkpoint_utils.checkpoint_trained_model,
          (checkpoint_data, save_checkpoint_path, copy_step))
      chrono.resume()

    # Report training progress.
    if not config.get('only_eval', False) and train_utils.itstime(
        step, config.log_training_steps, total_steps, process=0):
      write_note('Reporting training progress...')
      timing_measurements, note = chrono.tick(step)
      write_note(note)
      train_measurements = {}
      train_measurements.update(flax.jax_utils.unreplicate(extra_measurements))
      train_measurements.update(timing_measurements)
      writer.write_scalars(step, train_measurements)
      # Keep train_loss to return for reproducibility tests.
      train_loss = train_measurements['training_loss']

    # Report validation performance.
    if config.get('only_eval', False) or train_utils.itstime(
        step, config.log_eval_steps, total_steps):
      write_note('Evaluating on the validation set...')
      chrono.pause()

      eval_fns = collections.defaultdict(lambda: evaluation_fn)
      for zeroshot_name, zeroshot_config in config.get('zeroshot_eval_datasets',
                                                       {}).items():
        eval_fns[f'zeroshot_{zeroshot_name}'] = make_zeroshot_evaluation_fn(
            flax.jax_utils.unreplicate(opt_repl.target), states_repl,
            zeroshot_config['classnames_key'], zeroshot_config['prompts_key'])

      for eval_name, eval_ds in (eval_ds_splits | zeroshot_ds_splits).items():
        # Sets up evaluation metrics.
        ece_num_bins = config.get('ece_num_bins', 15)
        auc_num_bins = config.get('auc_num_bins', 1000)
        ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
        calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
        oc_auc_0_5 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.005,
                                                       num_bins=auc_num_bins)
        oc_auc_1 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.01,
                                                     num_bins=auc_num_bins)
        oc_auc_2 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.02,
                                                     num_bins=auc_num_bins)
        oc_auc_5 = rm.metrics.OracleCollaborativeAUC(oracle_fraction=0.05,
                                                     num_bins=auc_num_bins)

        # Runs evaluation loop.
        eval_iter = input_utils.start_input_pipeline(
            eval_ds, config.get('prefetch_to_device', 1))
        eval_fn = eval_fns[eval_name]
        ncorrect, loss, nseen = 0, 0, 0
        for batch in eval_iter:
          batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
              eval_fn(opt_repl.target, states_repl, batch['image'],
                      batch['text'], batch['mask']))
          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.
          ncorrect += np.sum(np.array(batch_ncorrect[0]))
          loss += np.sum(np.array(batch_losses[0]))
          nseen += np.sum(np.array(batch_n[0]))

          # Here we parse batch_metric_args to compute uncertainty metrics.
          # (e.g., ECE or Calibration AUC).
          logits, labels, masks = batch_metric_args
          masks = np.array(masks[0], dtype=bool)
          logits = np.array(logits[0])
          probs = jax.nn.softmax(logits)
          # From one-hot to integer labels, as required by ECE.
          int_labels = np.argmax(np.array(labels[0]), axis=-1)
          int_preds = np.argmax(logits, axis=-1)
          confidence = np.max(probs, axis=-1)
          for p, c, l, d, m in zip(probs, confidence, int_labels, int_preds,
                                   masks):
            ece.add_batch(p[m, :], label=l[m])
            calib_auc.add_batch(d[m], label=l[m], confidence=c[m])
            # TODO(jereliu): Extend to support soft multi-class probabilities.
            oc_auc_0_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_1.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_2.add_batch(d[m], label=l[m], custom_binning_score=c[m])
            oc_auc_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])

        val_loss[eval_name] = loss / nseen  # Keep for reproducibility tests.
        val_measurements = {
            f'{eval_name}_prec@1': ncorrect / nseen,
            f'{eval_name}_loss': val_loss[eval_name],
        }

        val_measurements[f'{eval_name}_ece'] = ece.result()['ece']
        val_measurements[f'{eval_name}_calib_auc'] = calib_auc.result()[
            'calibration_auc']
        val_measurements[f'{eval_name}_oc_auc_0.5%'] = oc_auc_0_5.result()[
            'collaborative_auc']
        val_measurements[f'{eval_name}_oc_auc_1%'] = oc_auc_1.result()[
            'collaborative_auc']
        val_measurements[f'{eval_name}_oc_auc_2%'] = oc_auc_2.result()[
            'collaborative_auc']
        val_measurements[f'{eval_name}_oc_auc_5%'] = oc_auc_5.result()[
            'collaborative_auc']
        writer.write_scalars(step, val_measurements)
      chrono.resume()

    if 'fewshot' in config and fewshotter is not None:
      # Compute few-shot on-the-fly evaluation.
      if config.get('only_eval', False) or train_utils.itstime(
          step, config.fewshot.log_steps, total_steps):
        chrono.pause()
        write_note(f'Few-shot evaluation...\n{chrono.note}')
        # Keep `results` to return for reproducibility tests.
        fewshot_results, best_l2 = fewshotter.run_all(opt_repl.target,
                                                      config.fewshot.datasets)
        # TODO(jjren): make fewshotters take states_repl
        # TODO(dusenberrymw): Remove this once fewshot.py is updated.
        def make_writer_measure_fn(step):

          def writer_measure(name, value):
            writer.write_scalars(step, {name: value})

          return writer_measure

        fewshotter.walk_results(
            make_writer_measure_fn(step), fewshot_results, best_l2)
        chrono.resume()

    if config.get('only_eval', False):
      break
    elif config.get('testing_failure_step'):
      # Break early to simulate infra failures in test cases.
      if config.testing_failure_step == step:
        break

  write_note(f'Done!\n{chrono.note}')
  pool.close()
  pool.join()
  writer.close()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  return train_loss, val_loss, fewshot_results


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.config_with_absl()

  # TODO(dusenberrymw): Refactor `main` such that there is a `train_eval`
  # function that returns values for tests and does not directly access flags,
  # and then have `main` return None.

  def _main(argv):
    del argv
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.
