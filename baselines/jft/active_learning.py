# pylint: disable=line-too-long
"""
This script implements a basic Active Learning loop using predictive entropy as acquisition function.

The below command is for running this script on a TPU-VM.

Execute in `baselines/jft`:

python3 active_learning.py \
  --output_dir="~/cifar-10/vit-16-i21k" \
  --config="experiments/imagenet21k_vit_base16_finetune_cifar10.py" \
  --config.model_init="gs://ub-data/ImageNet21k_ViT-B16_ImagetNet21k_ViT-B_16_28592399.npz" \
  --config.batch_size=256 \
  --config.total_steps=50

Note the strongly reduced total_steps
"""
# pylint: enable=line-too-long

import math
import numbers
from functools import partial

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_datasets as tfds
import tqdm
import uncertainty_baselines as ub
from absl import app, flags
import al_utils  # pylint: disable=unused-import # to register Cifar10Subset as dataset
import checkpoint_utils, input_utils, preprocess_utils, train_utils
from clu import preprocess_spec
from ml_collections.config_flags import config_flags

config_flags.DEFINE_config_file(
  'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')

FLAGS = flags.FLAGS

# TODO: can we use float("-inf") here?
NINF_SCORE = -100

def get_id_entropy(*, model, opt_repl, iter_ds):
  """Obtain entropy scores for each datapoint.

  Args:
    model: a initialized model.
    opt_repl: an optimizer with parameters.
    iter_ds: a dataset.

  Returns:
    a tuple of numpy arrays of ids and entropies.
  """
  @partial(jax.pmap, axis_name="batch")
  def compute_batch_entropy(params, images, mask):
    logits, _ = model.apply({"params": flax.core.freeze(params)},
                            images,
                            train=False)

    log_probs = jax.nn.log_softmax(logits)
    probs = jax.nn.softmax(logits)

    weighted_nats = -probs * log_probs
    # One simple trick to avoid NaNs later on.
    weighted_nats = jnp.where(jnp.isnan(weighted_nats), 0, weighted_nats)

    entropy = jnp.sum(weighted_nats, axis=-1, keepdims=False)

    # Take care of the mask.
    entropy = jnp.where(mask, entropy, NINF_SCORE)

    # TODO: For multi host this requires:
    # entropy = jax.lax.all_gather(entropy, axis_name='batch')
    return entropy

  entropies = []
  ids = []
  for batch in iter_ds:
    batch_entropy = compute_batch_entropy(opt_repl.target,
                                          batch["image"],
                                          batch["mask"])
    # TODO: if we run on multi host, this needs to be used as batch_entropy[0]

    flat_batch_entropy = np.array(batch_entropy).ravel()
    flat_id = np.array(batch["id"]).ravel()
    assert flat_batch_entropy.shape == flat_id.shape
    entropies.append(flat_batch_entropy)
    ids.append(flat_id)

  entropies = np.concatenate(entropies)
  ids = np.concatenate(ids)

  # NOTE: due to batch padding, entropies/ids will be of size:
  # if training set size % batch size > 0:
  # (training set size // batch size + 1) * batch size
  # else:
  # just training set size

  return ids, entropies


def select_acquisition_batch_indices(*,
                                     model,
                                     opt_repl,
                                     pool_ds,
                                     acquisition_batch_size,
                                     ignored_ids,
                                     prefetch_to_device=1):
  """Select what data points to acquire from the pool set.

  Args:
    model: an initialized model.
    opt_repl: an optimizer with parameters.
    pool_ds: the pool dataset.
    acquisition_batch_size: the number of data point to acquire.
    ignored_ids: the ids to ignore (previously acquired).
    prefetch_to_device: number of batches to prefetc (default: 1).

  Returns:
    a tuple of lists with the ids to be acquired and their scores.
  """
  iter_ds = input_utils.start_input_pipeline(pool_ds, prefetch_to_device)
  ids, scores = get_id_entropy(model=model, opt_repl=opt_repl, iter_ds=iter_ds)

  # Ignore already selected ids
  # TODO: vectorize this
  ids_list = ids.tolist()
  for ignored_id in ignored_ids:
    scores[ids_list.index(ignored_id)] = NINF_SCORE

  f_ent = scores[scores > 0]
  print(f"Score statistics pool set - "
        f"min: {f_ent.min()}, mean: {f_ent.mean()}, max: {f_ent.max()}")

  partitioned_scorers = np.argpartition(-scores, acquisition_batch_size)
  top_scorers = partitioned_scorers[:acquisition_batch_size]

  top_ids = ids[top_scorers].tolist()
  top_scores = scores[top_scorers].tolist()


  print(f"Data selected - ids: {top_ids}, with scores: {top_scores}")

  return top_ids, top_scores


def get_accuracy(*, evaluation_fn, opt_repl, ds, prefetch_to_device=1):
  """A helper function to obtain accuracy over a dataset

  Args:
    evaluation_fn: a function that evaluates a forward pass in a model.
    opt_repl: an optimizer with parameters.
    ds: a dataset.
    prefetch_to_device: number of batches to prefetc (default: 1).

  Returns:
    The accuracy as a float between 0 and 1.
  """
  iter_ds = input_utils.start_input_pipeline(ds, prefetch_to_device)

  ncorrect, nseen = [], []
  for batch in iter_ds:
    batch_ncorrect, _, batch_n, _ = evaluation_fn(
      opt_repl.target, batch["image"], batch["labels"], batch["mask"]
    )

    ncorrect += [batch_ncorrect[0]]
    nseen += [batch_n[0]]

  ncorrect = np.sum(ncorrect)
  nseen = np.sum(nseen)

  return ncorrect / nseen


def finetune(*,
             update_fn,
             opt_repl,
             lr_fn,
             ds,
             rngs_loop,
             total_steps,
             prefetch_to_device=1):
  """Finetunes a model on a dataset.

  Args:
    update_fn: a function that updates the model given relevant inputs.
    opt_repl: the optimizer.
    lr_fn: a function that returns the learning rate given a step.
    ds: the dataset to finetune on.
    rngs_loop: the rng for the loop.
    total_steps: the total number of fine-tuning steps to take.
    prefetch_to_device: number of batches to prefetc (default: 1).

  Returns:
    The optimizer with updated parameters and the updated rng.
  """
  iter_ds = input_utils.start_input_pipeline(ds, prefetch_to_device)
  lr_iter = train_utils.prefetch_scalar(
    map(lr_fn, range(total_steps)), prefetch_to_device
  )

  for _, train_batch, lr_repl in zip(tqdm.trange(1, total_steps + 1),
                                     iter_ds,
                                     lr_iter):
    opt_repl, _, rngs_loop, _ = update_fn(opt_repl,
                                          lr_repl,
                                          train_batch["image"],
                                          train_batch["labels"],
                                          rngs_loop)

  return opt_repl, rngs_loop


def make_init_fn(model, image_shape, local_batch_size, config):
  """Make the init function.

  Args:
    model: The model to init.
    image_shape: The shape of the input images.
    local_batch_size: the local device batch size.
    config: the full config for the experiment.

  Returns:
    The init function
  """
  @partial(jax.jit, backend="cpu")
  def init(rng):
    dummy_input = jnp.zeros((local_batch_size,) + image_shape, jnp.float32)

    params = flax.core.unfreeze(
      model.init(rng, dummy_input, train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    params['head']['bias'] = jnp.full_like(
      params['head']['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

    return params

  return init


def make_update_fn(model, config):
  """Make the update function.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that updates the model for one step.
  """

  @partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
  def update_fn(opt, lr, images, labels, rng):
    """Update step."""

    measurements = {}

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index("batch"))

    def loss_fn(params, images, labels):
      logits, _ = model.apply(
        {"params": flax.core.freeze(params)},
        images,
        train=True,
        rngs={"dropout": rng_model_local},
      )
      return getattr(train_utils, config.get("loss", "sigmoid_xent"))(
        logits=logits, labels=labels
      )

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    l, g = train_utils.accumulate_gradient(
      jax.value_and_grad(loss_fn),
      opt.target,
      images,
      labels,
      config.get("grad_accum_steps"),
    )
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
      g = jax.tree_util.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    decay_rules = config.get("weight_decay", []) or []
    if isinstance(decay_rules, numbers.Number):
      decay_rules = [(".*kernel.*", decay_rules)]
    sched_m = lr / config.lr.base if config.get("weight_decay_decouple") else lr

    def decay_fn(v, wd):
      return (1.0 - sched_m * wd) * v

    opt = opt.replace(
      target=train_utils.tree_map_with_regex(decay_fn, opt.target, decay_rules)
    )

    params, _ = jax.tree_flatten(opt.target)
    measurements["l2_params"] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))

    return opt, l, rng, measurements
  return update_fn


def make_evaluation_fn(model, config):
  """Make evaluation function.

  Args:
    model: The model to be used in evaluation.
    config: The config of the experiment.

  Returns:
    The evaluation function.
  """
  @partial(jax.pmap, axis_name="batch")
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    logits, out = model.apply({"params": flax.core.freeze(params)},
                              images,
                              train=False)

    losses = getattr(train_utils, config.get("loss", "sigmoid_xent"))(
      logits=logits, labels=labels, reduction=False
    )
    loss = jax.lax.psum(losses * mask, axis_name="batch")

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name="batch")
    n = jax.lax.psum(mask, axis_name="batch")

    # NOTE: this works on multi host devices already
    metric_args = jax.lax.all_gather(
        [logits, labels, out["pre_logits"], mask], axis_name="batch"
    )

    return ncorrect, loss, n, metric_args
  return evaluation_fn


def main(config, output_dir):
  print(config)

  # Keep the ID for filtering the pool set
  keep_id = 'keep(["image", "labels", "id"])'
  # HACK: assumes the keep is at the end
  id_pp_eval_split = config.pp_eval.split("|")
  id_pp_eval = "|".join(id_pp_eval_split[:-1] + [keep_id])

  # Download dataset
  data_builder = tfds.builder("cifar10")
  data_builder.download_and_prepare()

  seed = config.get("seed", 0)
  rng = jax.random.PRNGKey(seed)

  batch_size = config.batch_size
  batch_size_eval = config.get("batch_size_eval", batch_size)

  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = batch_size_eval // jax.process_count()

  # TODO: val_ds for early stopping

  test_ds = input_utils.get_data(
    dataset=config.dataset,
    split="test",
    rng=None,
    host_batch_size=local_batch_size_eval,
    preprocess_fn=preprocess_spec.parse(
      spec=config.pp_eval, available_ops=preprocess_utils.all_ops()
    ),
    shuffle=False,
    prefetch_size=config.get("prefetch_to_host", 2),
    num_epochs=1,  # Only repeat once.
  )

  model = ub.models.vision_transformer(
      num_classes=config.num_classes, **config.get("model", {})
  )

  image_shape = tuple(test_ds.element_spec['image'].shape[2:])
  init = make_init_fn(model, image_shape, local_batch_size, config)

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  # Load the optimizer from flax.
  opt_name = config.get("optim_name")
  opt_def = getattr(flax.optim, opt_name)(**config.get("optim", {}))

  # We jit this, such that the arrays that are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  reinit_params = config.get("model_reinit_params",
                             ("head/kernel", "head/bias"))
  loaded = checkpoint_utils.load_from_pretrained_checkpoint(
    params_cpu,
    config.model_init,
    config.model.representation_size,
    config.model.classifier,
    reinit_params,
  )

  opt_cpu = opt_cpu.replace(target=loaded)

  # TODO: This shouldn't be needed but opt_cpu is being donated
  opt_cpu = jax.device_get(opt_cpu)

  update_fn = make_update_fn(model, config)
  evaluation_fn = make_evaluation_fn(model, config)

  # NOTE: if we could `enumerate` before `filter` in `create_dataset` of CLU
  # then this dataset creation could be simplified.
  # https://github.com/google/CommonLoopUtils/blob/main/clu/deterministic_data.py#L340
  # CLU is explicitly not accepting outside contributions at the moment.
  train_subset_data_builder = tfds.builder(
      "cifar10_subset", subset_ids={"train": [], "test": None}
  )
  train_subset_data_builder.download_and_prepare()

  pool_subset_data_builder = tfds.builder(
      "cifar10_subset", subset_ids={"train": None, "test": None}
  )
  pool_subset_data_builder.download_and_prepare()

  rng, pool_ds_rng = jax.random.split(rng)

  # NOTE: below line is necessary on multi host setup
  # pool_ds_rng = jax.random.fold_in(pool_ds_rng, jax.process_index())

  pool_train_ds = input_utils.get_data(
    dataset=pool_subset_data_builder,
    split=config.train_split,
    rng=pool_ds_rng,
    host_batch_size=local_batch_size,
    preprocess_fn=preprocess_spec.parse(
        spec=id_pp_eval, available_ops=preprocess_utils.all_ops()
    ),
    shuffle=False,
    drop_remainder=False,
    prefetch_size=config.get("prefetch_to_host", 2),
    num_epochs=1,  # Don't repeat
  )

  test_accuracies = []
  training_size = []

  rng, rng_loop = jax.random.split(rng, 2)
  rngs_loop = flax_utils.replicate(rng_loop)

  current_train_ds_length = len(train_subset_data_builder.subset_ids["train"])
  while current_train_ds_length < config.get("max_labels", 150):
    print(f"Training set size: {current_train_ds_length}")

    if current_train_ds_length > 0:
      rng, train_ds_rng = jax.random.split(rng)
      train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())

      # Repeat dataset to have oversampled epochs and bootstrap more batches
      number_of_batches = current_train_ds_length / config.batch_size
      num_repeats = math.ceil(config.total_steps / number_of_batches)
      print(f"Repeating dataset {num_repeats} times")

      current_train_ds = input_utils.get_data(
        dataset=train_subset_data_builder,
        split=config.train_split,
        rng=train_ds_rng,
        host_batch_size=local_batch_size,
        preprocess_fn=preprocess_spec.parse(
            spec=config.pp_train, available_ops=preprocess_utils.all_ops()
        ),
        shuffle_buffer_size=config.shuffle_buffer_size,
        prefetch_size=config.get("prefetch_to_host", 2),
        num_epochs=num_repeats,
      )

      # NOTE: warmup and decay are not a good fit for the small training set
      # lr_fn = train_utils.create_learning_rate_schedule(config.total_steps,
      #                                                   **config.get('lr', {})
      #                                                   )
      lr_fn = lambda x: config.lr.base

      opt_repl = flax_utils.replicate(opt_cpu)
      current_opt_repl, rngs_loop = finetune(update_fn=update_fn,
                                             opt_repl=opt_repl,
                                             lr_fn=lr_fn,
                                             ds=current_train_ds,
                                             rngs_loop=rngs_loop,
                                             total_steps=config.total_steps)

      test_accuracy = get_accuracy(evaluation_fn=evaluation_fn,
                                   opt_repl=current_opt_repl,
                                   ds=test_ds)
    else:
      # Accuracy at start of AL loop - expected to be random
      current_opt_repl = flax_utils.replicate(opt_cpu)
      test_accuracy = get_accuracy(evaluation_fn=evaluation_fn,
                                   opt_repl=current_opt_repl,
                                   ds=test_ds)

    print(f"Accuracy at {current_train_ds_length}: {test_accuracy}")

    test_accuracies.append(test_accuracy)
    training_size.append(current_train_ds_length)

    acquisition_batch_ids, _ = select_acquisition_batch_indices(
      model=model,
      opt_repl=current_opt_repl,
      pool_ds=pool_train_ds,
      acquisition_batch_size=config.get("acquisition_batch_size", 10),
      ignored_ids=train_subset_data_builder.subset_ids["train"],
    )

    train_subset_data_builder.subset_ids["train"].extend(acquisition_batch_ids)
    current_train_ds_length = len(train_subset_data_builder.subset_ids["train"])

  print("########################")
  print(f"Final acquired training ids: "
        f"{train_subset_data_builder.subset_ids['train']}")
  print(f"Final Accuracy: {test_accuracies[-1]}")

  # TODO: save the final checkpoint

  return  train_subset_data_builder.subset_ids["train"], test_accuracies


if __name__ == "__main__":
  jax.config.config_with_absl()

  def _main(argv):
    del argv
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.
