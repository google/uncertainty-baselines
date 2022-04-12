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

# pylint: disable=line-too-long
r"""Active learning loop.

This script implements a basic Active Learning loop using predictive entropy as
acquisition function.

The below command is for running this script on a TPU-VM.

Execute in `baselines/jft`:

python3 active_learning.py \
  --config='experiments/vit_l32_active_learning_cifar.py' \
  --config.model_init='gs://ub-checkpoints/ImageNet21k_ViT-L32/1/checkpoint.npz' \
  --output_dir active_learning_results


Use `gs://ub-checkpoints/ImageNet21k_BE-L32/baselines-jft-0209_205214/1/checkpoint.npz` for BE
"""
# pylint: enable=line-too-long

from functools import partial  # pylint: disable=g-importing-member standard use
import logging
import math
import multiprocessing

from absl import app
from absl import flags
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags
import numpy as np
import tensorflow_datasets as tfds
import tqdm
import uncertainty_baselines as ub

import al_utils  # local file import from baselines.jft
import batchensemble_utils  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import deterministic_utils  # local file import from baselines.jft
import input_utils  # local file import from baselines.jft
import ood_utils  # local file import from baselines.jft
import preprocess_utils  # local file import from baselines.jft
import train_utils  # local file import from baselines.jft
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

NINF_SCORE = float('-inf')


def get_ids_logits_masks(*,
                         model,
                         opt_repl,
                         ds,
                         use_pre_logits=False,
                         average_logits=True,
                         prefetch_to_device=1,
                         config=None):
  """Obtain (pre) logits for each datapoint.

  This can be then used to compute entropies, and so on.

  Args:
    model: a initialized model.
    opt_repl: an optimizer with parameters.
    ds: a dataset.
    use_pre_logits: if True, return pre logit instead of logit
    average_logits: if True, average the logits.
    prefetch_to_device: how many batches to prefix
    config: experiment config.

  Returns:
    a tuple of jnp arrays of ids, logits, labels and masks.
  """

  @partial(jax.pmap, axis_name='batch')
  def compute_batch_outputs(params, images):
    logits, out = model.apply({'params': flax.core.freeze(params)},
                              images,
                              train=False)
    if config and config.model_type == 'batchensemble':
      ens_size = config.model.transformer.ens_size
      loss_name = config.get('loss', 'sigmoid_xent')
      logits = jnp.asarray(jnp.split(logits, ens_size))
      if loss_name == 'sigmoid_xent':
        if average_logits:
          logits = batchensemble_utils.log_average_sigmoid_probs(logits)
      elif loss_name == 'softmax_xent':
        if average_logits:
          logits = batchensemble_utils.log_average_softmax_probs(logits)
      else:
        raise ValueError(f'Loss name: {loss_name} not supported.')

      if use_pre_logits:
        # pre_logits [batch_size, hidden_size, ens_size]
        pre_logits = jnp.transpose(
            jnp.asarray(jnp.split(out['pre_logits'], ens_size)), axes=[1, 2, 0])
        output = pre_logits
      else:
        output = logits
    else:
      if use_pre_logits:
        # pre_logits [batch_size, hidden_size]
        output = out['pre_logits']
      else:
        output = logits

    # TODO(joost,andreas): For multi host this requires:
    # output = jax.lax.all_gather(output, axis_name='batch')
    return output

  iter_ds = input_utils.start_input_pipeline(ds, prefetch_to_device)

  outputs = []
  ids = []
  labels = []
  masks = []
  for _, batch in enumerate(iter_ds):
    batch_id = batch['id']
    batch_label = batch['labels']
    batch_mask = batch['mask']
    batch_output = compute_batch_outputs(opt_repl.target, batch['image'])

    # This moves the batch_output from TPU to CPU right away.
    batch_output = jax.device_put(batch_output, jax.devices('cpu')[0])

    # TODO(joost,andreas): if we run on multi host, we need to index
    # batch_outputs: batch_outputs[0]
    ids.append(batch_id)
    outputs.append(batch_output)
    labels.append(batch_label)
    masks.append(batch_mask)

  if average_logits:
    # 0 dimension is TPU shard, 1 is batch
    outputs = jnp.concatenate(outputs, axis=1)
  else:
    # 0 dimension is TPU shard, 1 is ensemble, 2 is batch
    outputs = jnp.concatenate(outputs, axis=2)

  ids = jnp.concatenate(ids, axis=1)
  labels = jnp.concatenate(labels, axis=1)
  masks = jnp.concatenate(masks, axis=1)
  # NOTE(joost,andreas): due to batch padding, entropies/ids will be of size:
  # if training set size % batch size > 0:
  # (training set size // batch size + 1) * batch size
  # else:
  # just training set size

  return ids, outputs, labels, masks


def get_entropy_scores(logits, masks):
  """Obtain scores using entropy scoring.

  Args:
    logits: the logits of the pool set.
    masks: the masks belonging to the pool set.

  Returns:
    a list of scores belonging to the pool set.
  """
  log_probs = jax.nn.log_softmax(logits)
  probs = jax.nn.softmax(logits)

  weighted_nats = -probs * log_probs
  # One simple trick to avoid NaNs later on.
  weighted_nats = jnp.where(jnp.isnan(weighted_nats), 0, weighted_nats)
  entropy = jnp.sum(weighted_nats, axis=-1, keepdims=False)
  entropy = jnp.where(masks, entropy, NINF_SCORE)

  return entropy


def get_bald_scores(logits, masks):
  """Obtain scores using BALD scoring.

  Args:
    logits: the logits of the pool set, first dimension is the ensemble.
    masks: the masks belonging to the pool set.

  Returns:
    a list of scores belonging to the pool set.
  """

  # TPU shard, ensemble size, batch size, logits
  _, ens_size, _, _ = logits.shape

  log_probs = jax.nn.log_softmax(logits)
  probs = jax.nn.softmax(logits)

  weighted_nats = -probs * log_probs
  weighted_nats = jnp.where(jnp.isnan(weighted_nats), 0, weighted_nats)

  marginal_entropy = jnp.mean(jnp.sum(weighted_nats, axis=-1), axis=1)

  marginal_log_probs = jax.nn.logsumexp(log_probs, axis=1) - jnp.log(ens_size)
  marginal_probs = jnp.mean(probs, axis=1)

  weighted_marginal_nats = -marginal_probs * marginal_log_probs
  weighted_marginal_nats = jnp.where(
      jnp.isnan(weighted_marginal_nats), 0, weighted_marginal_nats)

  entropy_marginal = jnp.sum(weighted_marginal_nats, axis=-1)

  # Mask results.
  bald = entropy_marginal - marginal_entropy
  bald = jnp.where(masks, bald, NINF_SCORE)

  return bald


def get_margin_scores(logits, masks):
  """Obtain scores using margin scoring.

  Args:
    logits: the logits of the pool set.
    masks: the masks belonging to the pool set.

  Returns:
    a list of scores belonging to the pool set.
  """
  probs = jax.nn.softmax(logits)
  top2_probs = jax.lax.top_k(probs, k=2)[0]
  # top_k's documentation does not specify whether the top-k are sorted or not.
  margins = jnp.abs(top2_probs[..., 0] - top2_probs[..., 1])

  # Lower margin means higher uncertainty, so we invert the scores.
  # Then higer margin score means higher uncertainty.
  margin_scores = -margins
  margin_scores = jnp.where(masks, margin_scores, NINF_SCORE)

  return margin_scores


def get_msp_scores(logits, masks):
  """Obtain scores using maximum softmax probability scoring.

  Args:
    logits: the logits of the pool set.
    masks: the masks belonging to the pool set.

  Returns:
    a list of scores belonging to the pool set.
  """
  probs = jax.nn.softmax(logits)
  max_probs = jnp.max(probs, axis=-1)

  # High max prob means low uncertainty, so we invert the value.
  msp_scores = -max_probs
  msp_scores = jnp.where(masks, msp_scores, NINF_SCORE)

  return msp_scores


def get_uniform_scores(masks, rng):
  """Obtain scores using uniform sampling.

  Args:
    masks: the masks belonging to the pool set.
    rng: the RNG to use for uniform sampling.

  Returns:
    a list of scores belonging to the pool set.
  """
  uniform_scores = jax.random.uniform(key=rng, shape=masks.shape)
  uniform_scores = jnp.where(masks, uniform_scores, NINF_SCORE)

  return uniform_scores


def get_density_scores(*,
                       model,
                       opt_repl,
                       train_ds,
                       pool_pre_logits,
                       pool_masks,
                       config=None):
  """Obtain scores using density method.

  Args:
    model: an initialized model.
    opt_repl: the current optimizer.
    train_ds: the dataset to fit the density estimator on.
    pool_pre_logits: the pre logits (features) of the pool set.
    pool_masks: the masks belonging to the pool_pre_logits.
    config: experiment config.

  Returns:
    a list of scores belonging to the pool set.
  """
  # Fit LDA
  _, train_pre_logits, train_labels, train_masks = get_ids_logits_masks(
      model=model,
      opt_repl=opt_repl,
      ds=train_ds,
      use_pre_logits=True,
      config=config)

  # train_masks_bool [num_cores, per_core_batch_size]
  train_masks_bool = train_masks.astype(bool)
  # train_pre_logits [num_cores, per_core_batch_size, hidden_size, ens_size]
  # train_embeds [batch_size, hidden_size, ens_size]
  # batch_size = num_cores * per_core_batch_size
  train_embeds = train_pre_logits[train_masks_bool]
  train_labels = np.argmax(train_labels[train_masks_bool], axis=-1).ravel()

  use_ens = False
  if len(train_embeds.shape) == 3:
    # The output needs to the ensembled
    # embeds is of the shape [batch_size, hidden_size, ens_size]
    use_ens = True
    ens_size = train_embeds.shape[-1]

  if not use_ens:
    # Single model
    # train_embeds shape [batch_size, hidden_size]
    mean_list, cov = ood_utils.compute_mean_and_cov(train_embeds, train_labels)
  else:
    # Ensemble models
    # train_embeds shape [batch_size, hidden_size, ens_size]
    mean_list, cov = [], []
    for m in range(ens_size):
      mu, sigma = ood_utils.compute_mean_and_cov(train_embeds[..., m],
                                                 train_labels)
      mean_list.append(mu)
      cov.append(sigma)

  # Evaluate LDA on pool set
  if not use_ens:
    # Single model
    # pool_pre_logits [num_cores, per_core_batch_size, hidden_size]
    pool_pre_logits = pool_pre_logits.reshape(-1, pool_pre_logits.shape[-1])
    dists = ood_utils.compute_mahalanobis_distance(pool_pre_logits, mean_list,
                                                   cov)
    scores = np.array(jax.nn.logsumexp(-dists / 2, axis=-1))
  else:
    # Ensemble models
    # pool_pre_logits [num_cores, per_core_batch_size, hidden_size, ens_size]
    pool_pre_logits = pool_pre_logits.reshape(
        [-1] + [s for s in pool_pre_logits.shape[2:]])
    for m in range(ens_size):
      scores_list = []
      d = ood_utils.compute_mahalanobis_distance(pool_pre_logits[..., m],
                                                 mean_list[m], cov[m])
      s = np.array(jax.nn.logsumexp(-d / 2, axis=-1))
      scores_list.append(s)
    scores = np.mean(np.array(scores_list), axis=0)

  # Convert likelihood to AL score
  pool_masks_bool = np.array(pool_masks.ravel(), dtype=bool)
  scores[pool_masks_bool] = (
      scores[pool_masks_bool].max() - scores[pool_masks_bool])
  scores[~pool_masks_bool] = NINF_SCORE

  return scores


def power_score_acquisition(scores, acquisition_batch_size, beta, rng):
  """Power method for batch selection https://arxiv.org/abs/2106.12059."""
  noise = jax.random.gumbel(rng, [len(scores)])
  noised_scores = scores + noise / beta

  selected_scores, selected_indices = jax.lax.top_k(noised_scores,
                                                    acquisition_batch_size)

  return selected_scores, selected_indices


def select_acquisition_batch_indices(*,
                                     acquisition_batch_size,
                                     scores,
                                     ids,
                                     ignored_ids,
                                     power_acquisition=False,
                                     rng=None):
  """Select what data points to acquire from the pool set.

  Args:
    acquisition_batch_size: the number of data point to acquire.
    scores: acquisition scores assigned to data points.
    ids: the ids belonging to the scores.
    ignored_ids: the ids to ignore (previously acquired).
    power_acquisition: True if use power method for batch selection.
    rng: rng for power acquisition. None if not using power_acquisition.

  Returns:
    a tuple of lists with the ids to be acquired and their scores.
  """
  scores = np.array(scores.ravel())
  ids = np.array(ids.ravel())

  # Ignore already acquired ids
  # TODO(joost,andreas): vectorize this
  ids_list = ids.tolist()
  for ignored_id in ignored_ids:
    scores[ids_list.index(ignored_id)] = NINF_SCORE

  f_ent = scores[scores > NINF_SCORE]
  logging.info(msg=f'Score statistics pool set - '
               f'min: {f_ent.min()}, mean: {f_ent.mean()}, max: {f_ent.max()}')

  if power_acquisition:
    assert rng is not None, ('rng should not be None if power acquisition is '
                             'used.')
    beta = 1
    selected_scorers = power_score_acquisition(scores, acquisition_batch_size,
                                               beta, rng)
  else:
    # Use top-k otherwise.
    partitioned_scorers = np.argpartition(-scores, acquisition_batch_size)
    selected_scorers = partitioned_scorers[:acquisition_batch_size]

  selected_ids = ids[selected_scorers].tolist()
  selected_scores = scores[selected_scorers].tolist()

  logging.info(
      msg=f'Data selected - ids: {selected_ids}, with scores: {selected_scores}'
  )

  return selected_ids, selected_scores


def acquire_points(model, current_opt_repl, pool_train_ds, train_eval_ds,
                   train_subset_data_builder, acquisition_method, config,
                   rng_loop):
  """Acquire ids of the current batch."""
  pool_ids, pool_outputs, _, pool_masks = get_ids_logits_masks(
      model=model,
      opt_repl=current_opt_repl,
      ds=pool_train_ds,
      use_pre_logits=acquisition_method == 'density',
      average_logits=acquisition_method != 'bald',
      config=config)

  if acquisition_method == 'uniform':
    rng_loop, rng_acq = jax.random.split(rng_loop, 2)
    pool_scores = get_uniform_scores(pool_masks, rng_acq)
  elif acquisition_method == 'entropy':
    pool_scores = get_entropy_scores(pool_outputs, pool_masks)
  elif acquisition_method == 'margin':
    pool_scores = get_margin_scores(pool_outputs, pool_masks)
  elif acquisition_method == 'msp':
    pool_scores = get_msp_scores(pool_outputs, pool_masks)
  elif acquisition_method == 'bald':
    pool_scores = get_bald_scores(pool_outputs, pool_masks)
  elif acquisition_method == 'density':
    if train_subset_data_builder.subset_ids:
      pool_scores = get_density_scores(
          model=model,
          opt_repl=current_opt_repl,
          train_ds=train_eval_ds,
          pool_pre_logits=pool_outputs,
          pool_masks=pool_masks,
          config=config)
    else:
      rng_loop, rng_acq = jax.random.split(rng_loop, 2)
      pool_scores = get_uniform_scores(pool_masks, rng_acq)
  else:
    raise ValueError('Acquisition method not found.')

  rng_loop, rng_acq = jax.random.split(rng_loop, 2)
  acquisition_batch_ids, _ = select_acquisition_batch_indices(
      acquisition_batch_size=config.get('acquisition_batch_size', 10),
      scores=pool_scores,
      ids=pool_ids,
      ignored_ids=train_subset_data_builder.subset_ids,
      power_acquisition=config.get('power_acquisition', False),
      rng=rng_acq)

  return acquisition_batch_ids, rng_loop


def get_accuracy(*, evaluation_fn, opt_repl, ds, prefetch_to_device=1):
  """A helper function to obtain accuracy over a dataset.

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
    batch_ncorrect, _, batch_n, _ = evaluation_fn(opt_repl.target,
                                                  batch['image'],
                                                  batch['labels'],
                                                  batch['mask'])

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
             train_eval_ds,
             val_ds,
             evaluation_fn,
             early_stopping_patience,
             prefetch_to_device=1,
             profiler=None):
  """Finetunes a model on a dataset.

  Args:
    update_fn: a function that updates the model given relevant inputs.
    opt_repl: the optimizer.
    lr_fn: a function that returns the learning rate given a step.
    ds: the dataset to finetune on.
    rngs_loop: the rng for the loop.
    total_steps: the total number of fine-tuning steps to take.
    train_eval_ds: train dataset in eval mode (no augmentation or shuffling).
    val_ds: validation dataset for early stopping.
    evaluation_fn: function used for evaluation on validation set.
    early_stopping_patience: number of steps to wait before stopping training.
    prefetch_to_device: number of batches to prefetc (default: 1).
    profiler: periodic_actions.Profile.

  Returns:
    The optimizer with updated parameters and the updated rng.
  """
  iter_ds = input_utils.start_input_pipeline(ds, prefetch_to_device)
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(total_steps)), prefetch_to_device)

  best_opt_accuracy = -1
  best_step = 1

  train_val_accuracies = []

  for current_step, train_batch, lr_repl in zip(
      tqdm.trange(1, total_steps + 1), iter_ds, lr_iter):
    opt_repl, rngs_loop, _ = update_fn(opt_repl, lr_repl, train_batch['image'],
                                       train_batch['labels'], rngs_loop)
    if jax.process_index() == 0 and profiler is not None:
      profiler(current_step)
    if current_step % 5 == 0:
      train_accuracy = get_accuracy(
          evaluation_fn=evaluation_fn, opt_repl=opt_repl, ds=train_eval_ds)
      val_accuracy = get_accuracy(
          evaluation_fn=evaluation_fn, opt_repl=opt_repl, ds=val_ds)
      logging.info(
          msg=f'Current accuracy - train:{train_accuracy}, val: {val_accuracy}')
      train_val_accuracies.append((current_step, train_accuracy, val_accuracy))

      if val_accuracy >= best_opt_accuracy:
        best_step = current_step
        best_opt_accuracy = val_accuracy
        best_opt_repl = jax.device_get(opt_repl)
      else:
        logging.info(
            msg=(f'Current val accuracy {val_accuracy} '
                 f'(vs {best_opt_accuracy})'))
        if current_step - best_step >= early_stopping_patience:
          logging.info(msg='Early stopping, returning best opt_repl!')
          break

  # best_opt_repl could be unassigned, but we should error out then

  info = dict(
      best_val_accuracy=best_opt_accuracy,
      best_step=best_step,
      train_val_accuracies=train_val_accuracies)

  return best_opt_repl, rngs_loop, info


def main(config, output_dir):
  if jax.process_count() > 1:
    raise NotImplementedError
  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir,
      first_profile=10)

  logging.info(config)

  acquisition_method = config.get('acquisition_method')
  if acquisition_method == 'bald':
    assert config.model_type == 'batchensemble', 'Bald requires batch ensemble'

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)
  writer.write_hparams(dict(config))

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

  write_note(f'Initializing for {acquisition_method}')

  # Download dataset
  data_builder = tfds.builder(config.dataset)
  data_builder.download_and_prepare()

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)

  batch_size = config.batch_size
  batch_size_eval = config.get('batch_size_eval', batch_size)

  local_batch_size = batch_size // jax.process_count()
  local_batch_size_eval = batch_size_eval // jax.process_count()

  val_ds = input_utils.get_data(
      dataset=config.dataset,
      split=config.val_split,
      rng=None,
      process_batch_size=local_batch_size_eval,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_eval, available_ops=preprocess_utils.all_ops()),
      shuffle=False,
      prefetch_size=config.get('prefetch_to_host', 2),
      num_epochs=1,  # Only repeat once.
  )

  test_ds = input_utils.get_data(
      dataset=config.dataset,
      split=config.test_split,
      rng=None,
      process_batch_size=local_batch_size_eval,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_eval, available_ops=preprocess_utils.all_ops()),
      shuffle=False,
      prefetch_size=config.get('prefetch_to_host', 2),
      num_epochs=1,  # Only repeat once.
  )

  # Init model
  if config.model_type == 'deterministic':
    model_utils = deterministic_utils
    reinit_params = config.get('model_reinit_params',
                               ('head/kernel', 'head/bias'))
    model = ub.models.vision_transformer(
        num_classes=config.num_classes, **config.get('model', {}))
  elif config.model_type == 'batchensemble':
    model_utils = batchensemble_utils
    reinit_params = ('batchensemble_head/bias', 'batchensemble_head/kernel',
                     'batchensemble_head/fast_weight_alpha',
                     'batchensemble_head/fast_weight_gamma')
    model = ub.models.vision_transformer_be(
        num_classes=config.num_classes, **config.model)
  else:
    raise ValueError('Expect config.model_type to be "deterministic" or'
                     f'"batchensemble", but received {config.model_type}.')

  init = model_utils.create_init(model, config, test_ds)

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    writer.write_scalars(step=0, scalars={'num_params': num_params})

  # Load the optimizer from flax.
  opt_name = config.get('optim_name')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  write_note('Loading the model checkpoint...')
  loaded_params = checkpoint_utils.load_checkpoint(
      tree=None, path=config.model_init)
  loaded_params = checkpoint_utils.restore_from_pretrained_params(
      params_cpu,
      loaded_params,
      config.model.representation_size,
      config.model.classifier,
      reinit_params,
  )

  opt_cpu = opt_cpu.replace(target=loaded_params)
  del loaded_params, params_cpu  # Free up memory.
  # TODO(dusenberrymw): Remove this once checkpoint_utils is fixed to return
  # only CPU arrays.
  opt_cpu = jax.device_get(opt_cpu)

  update_fn = model_utils.create_update_fn(model, config)
  evaluation_fn = model_utils.create_evaluation_fn(model, config)

  # NOTE: We need this because we need an Id field of type int.
  # TODO(andreas): Rename to IdSubsetDatasetBuilder?
  pool_subset_data_builder = al_utils.SubsetDatasetBuilder(
      data_builder, subset_ids=None)

  rng, pool_ds_rng = jax.random.split(rng)

  # NOTE: below line is necessary on multi host setup
  # pool_ds_rng = jax.random.fold_in(pool_ds_rng, jax.process_index())

  pool_train_ds = input_utils.get_data(
      dataset=pool_subset_data_builder,
      split=config.train_split,
      rng=pool_ds_rng,
      process_batch_size=local_batch_size,
      preprocess_fn=preprocess_spec.parse(
          spec=config.pp_eval, available_ops=preprocess_utils.all_ops()),
      shuffle=False,
      drop_remainder=False,
      prefetch_size=config.get('prefetch_to_host', 2),
      num_epochs=1,  # Don't repeat
  )

  # Potentially acquire an initial training set.
  initial_training_set_size = config.get('initial_training_set_size', 10)

  if initial_training_set_size > 0:
    current_opt_repl = flax_utils.replicate(opt_cpu)
    pool_ids, _, _, pool_masks = get_ids_logits_masks(
        model=model, opt_repl=current_opt_repl, ds=pool_train_ds, config=config)

    rng, initial_uniform_rng = jax.random.split(rng)
    pool_scores = get_uniform_scores(pool_masks, initial_uniform_rng)

    initial_training_set_batch_ids, _ = select_acquisition_batch_indices(
        acquisition_batch_size=initial_training_set_size,
        scores=pool_scores,
        ids=pool_ids,
        ignored_ids=set(),
    )
  else:
    initial_training_set_batch_ids = []

  train_subset_data_builder = al_utils.SubsetDatasetBuilder(
      data_builder, subset_ids=set(initial_training_set_batch_ids))

  test_accuracies = []
  training_sizes = []

  rng, rng_loop = jax.random.split(rng)
  rngs_loop = flax_utils.replicate(rng_loop)
  if config.model_type == 'batchensemble':
    rngs_loop = {'dropout': rngs_loop}

  # TODO(joost,andreas): double check if below is still necessary
  # (train_split is independent of this)
  # NOTE: train_ds_rng is re-used for all train_ds creations
  rng, train_ds_rng = jax.random.split(rng)

  measurements = {}
  accumulated_steps = 0
  while True:
    current_train_ds_length = len(train_subset_data_builder.subset_ids)
    write_note(f'Training set size: {current_train_ds_length}')
    if current_train_ds_length >= config.get('max_training_set_size', 150):
      break

    current_opt_repl = flax_utils.replicate(opt_cpu)

    # Only fine-tune if there is anything to fine-tune with.
    if current_train_ds_length > 0:
      # Repeat dataset to have oversampled epochs and bootstrap more batches
      number_of_batches = current_train_ds_length / config.batch_size
      num_repeats = math.ceil(config.total_steps / number_of_batches)
      write_note(f'Repeating dataset {num_repeats} times')

      # We repeat the dataset several times, such that we can obtain batches
      # of size batch_size, even at start of training. These batches will be
      # effectively 'bootstrap' sampled, meaning they are sampled with
      # replacement from the original training set.
      repeated_train_ds = input_utils.get_data(
          dataset=train_subset_data_builder,
          split=config.train_split,
          rng=train_ds_rng,
          process_batch_size=local_batch_size,
          preprocess_fn=preprocess_spec.parse(
              spec=config.pp_train, available_ops=preprocess_utils.all_ops()),
          shuffle_buffer_size=config.shuffle_buffer_size,
          prefetch_size=config.get('prefetch_to_host', 2),
          # TODO(joost,andreas): double check if below leads to bootstrap
          # sampling.
          num_epochs=num_repeats,
      )

      # We use this dataset to evaluate how well we perform on the training set,
      # and for fitting the feature density method.
      # We need training set accuracy to evaluate if we fit well within
      # max_steps budget.
      train_eval_ds = input_utils.get_data(
          dataset=train_subset_data_builder,
          split=config.train_split,
          rng=train_ds_rng,
          process_batch_size=local_batch_size,
          preprocess_fn=preprocess_spec.parse(
              spec=config.pp_eval, available_ops=preprocess_utils.all_ops()),
          shuffle=False,
          drop_remainder=False,
          prefetch_size=config.get('prefetch_to_host', 2),
          num_epochs=1,
      )

      # NOTE: warmup and decay are not a good fit for the small training set
      # lr_fn = train_utils.create_learning_rate_schedule(config.total_steps,
      #                                                   **config.get('lr', {})
      #                                                   )
      lr_fn = lambda x: config.lr.base

      early_stopping_patience = config.get('early_stopping_patience', 15)
      current_opt_repl, rngs_loop, measurements = finetune(
          update_fn=update_fn,
          opt_repl=current_opt_repl,
          lr_fn=lr_fn,
          ds=repeated_train_ds,
          rngs_loop=rngs_loop,
          total_steps=config.total_steps,
          train_eval_ds=train_eval_ds,
          val_ds=val_ds,
          evaluation_fn=evaluation_fn,
          early_stopping_patience=early_stopping_patience,
          profiler=profiler)
      train_val_accuracies = measurements.pop('train_val_accuracies')
      current_steps = 0
      for step, train_acc, val_acc in train_val_accuracies:
        writer.write_scalars(accumulated_steps + step, {
            'train_accuracy': train_acc,
            'val_accuracy': val_acc
        })
        current_steps = step
      accumulated_steps += current_steps + 10
    else:
      train_eval_ds = None

    test_accuracy = get_accuracy(
        evaluation_fn=evaluation_fn, opt_repl=current_opt_repl, ds=test_ds)

    write_note(f'Accuracy at {current_train_ds_length}: {test_accuracy}')

    test_accuracies.append(test_accuracy)
    training_sizes.append(current_train_ds_length)

    acquisition_batch_ids, rng_loop = acquire_points(
        model, current_opt_repl, pool_train_ds, train_eval_ds,
        train_subset_data_builder, acquisition_method, config, rng_loop)
    train_subset_data_builder.subset_ids.update(acquisition_batch_ids)

    measurements.update({'test_accuracy': test_accuracy})
    writer.write_scalars(current_train_ds_length, measurements)

  write_note(f'Final acquired training ids: '
             f'{train_subset_data_builder.subset_ids}'
             f'Accuracies: {test_accuracies}')

  pool.close()
  pool.join()
  writer.close()
  # TODO(joost,andreas): save the final checkpoint
  return (train_subset_data_builder.subset_ids, test_accuracies)


if __name__ == '__main__':
  jax.config.config_with_absl()

  def _main(argv):
    del argv
    main(FLAGS.config, FLAGS.output_dir)

  app.run(_main)  # Ignore the returned values from `main`.
