# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""V-MoEs, ensembles thereof and E^3.

[Riquelme et al., 2021] https://arxiv.org/abs/2106.05974
[Allingham et al., 2021] https://arxiv.org/abs/2110.03360

The models are assumed to be already trained and we access them via their
checkpoints. This script thus only focuses on the evaluation of the models.
"""
import functools
import multiprocessing

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import preprocess_spec
import flax
import jax
from jax.experimental import pjit
import jax.numpy as jnp
import ml_collections.config_flags
import numpy as np
import robustness_metrics as rm

import tensorflow as tf
import batchensemble_utils as be_u  # local file import from baselines.jft
import data_uncertainty_utils  # local file import from baselines.jft
import input_utils  # local file import from baselines.jft
import ood_utils  # local file import from baselines.jft
import preprocess_utils  # local file import from baselines.jft
import subpopl_utils  # local file import from baselines.jft
import train_utils  # local file import from baselines.jft
import vmoe_utils  # local file import from baselines.jft

from vmoe import partitioning
from vmoe.checkpoints import partitioned
from vmoe.nn import models

# TODO(dusenberrymw): Open-source remaining imports.
fewshot = None

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




def load_checkpoint(config, mesh, efficient_ensemble_size):
  """Loads the checkpoint to evaluate."""
  if not config.model_init:
    raise ValueError(('vmoe.py expects at least one path to a ckpt to load; '
                      f'got instead config.model_init={config.model_init}.'))
  model_init = config.model_init
  if not isinstance(model_init, (tuple, list)):
    model_init = [model_init]

  restore_checkpoint = functools.partial(
      partitioned.restore_checkpoint, tree=None, axis_resources=None)
  with mesh:
    params = {p: restore_checkpoint(prefix=p) for p in model_init}
  return flax.core.freeze(params)


def _reshape_from_pmap_shape(x):
  """Reshapes from (num_devices, batch_size_per_device, ...) to (-1, ...)."""
  return x.reshape((-1,) + x.shape[2:])


def _check_pmap_and_pjit_shapes(images, labels, mask):
  """Checks the inputs have the expected pmap shape-conventions within pjit."""
  names = ('images', 'labels', 'mask')
  num_devices = jax.device_count()
  for tensor, name in zip((images, labels, mask), names):
    # pjit can see the entire data array. Since the pmap inputs have shapes
    #    (num_local_devices, local_batch_size, ...) on each host,
    # pjit has to see a first dimension equal to the *total* number of devices.
    first_dim = tensor.shape[0]
    assert first_dim == num_devices, f'{name}: {first_dim} != {num_devices}.'


def ensemble_pred_fn(single_model_pred_fn, reshape_outputs_fn, params, images,
                     loss_as_str):
  """Predicts with a V-Moe, an ensemble thereof or E^3.

  Args:
    single_model_pred_fn: Function to predict from a single model.
    reshape_outputs_fn: Function to reshape the logits and prelogits into a
      canonical format with shape (ensemble size, batch size, dimension). In
      particular, the reshape logic differs for deep and efficient ensembles.
    params: PyTree of parameters of the form:
      {
        'model_1': params_model_1,
        'model_2': params_model_2,
        ...,
        'model_M': params_model_M,
      }
      with M denoting the ensemble size.
    images: Input images to make predictions for.
    loss_as_str: A string denoting either `softmax_xent` or `sigmoid_xent`. The
      logits are aggregated according to the choice of the loss.

  Returns:
    The log probablity of the logits and pre-logits.
  """
  assert loss_as_str in ('softmax_xent', 'sigmoid_xent'), loss_as_str
  if loss_as_str == 'softmax_xent':
    ens_logits_fn = be_u.log_average_softmax_probs
  else:
    ens_logits_fn = be_u.log_average_sigmoid_probs

  outputs = [single_model_pred_fn(p, images) for p in params.values()]
  # Both ens_logits and ens_prelogits are [ens_size, batch_size, hidden_size].
  ens_logits, ens_prelogits = reshape_outputs_fn(outputs)
  ens_logits = ens_logits_fn(ens_logits)
  # ens_prelogits [batch_size, hidden_size, ens_size].
  ens_prelogits = jnp.transpose(ens_prelogits, axes=[1, 2, 0])

  return ens_logits, ens_prelogits


def main(config, output_dir):

  seed = config.get('seed', 0)
  tf.random.set_seed(seed)

  if config.get('data_dir'):
    logging.info('data_dir=%s', config.data_dir)
  logging.info('Output dir: %s', output_dir)
  tf.io.gfile.makedirs(output_dir)

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)
  write_note('Initializing...')

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

  write_note('Initializing val dataset(s)...')

  def _get_val_split(dataset, split, pp_eval, data_dir=None):
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

    val_ds = input_utils.get_data(
        dataset=dataset,
        split=split,
        rng=None,
        process_batch_size=local_batch_size_eval,
        preprocess_fn=pp_eval,
        cache=config.get('val_cache', 'batched'),
        num_epochs=1,
        repeat_after_batching=True,
        shuffle=False,
        prefetch_size=config.get('prefetch_to_host', 2),
        drop_remainder=False,
        data_dir=data_dir)

    return val_ds

  val_ds_splits = {
      'val':
          _get_val_split(
              config.dataset,
              split=config.val_split,
              pp_eval=config.pp_eval,
              data_dir=config.get('data_dir'))
  }

  if config.get('test_split'):
    val_ds_splits.update({
        'test':
            _get_val_split(
                config.dataset,
                split=config.test_split,
                pp_eval=config.pp_eval,
                data_dir=config.get('data_dir'))
    })

  if config.get('subpopl_cifar_data_file'):
    dataset_builder = input_utils.cifar_from_sql(
        sql_database=config.subpopl_cifar_data_file,
        num_classes=config.num_classes)

    subpopl_val_ds_splits = {  # pylint: disable=g-complex-comprehension
        client_id: _get_val_split(
            dataset_builder,
            split=client_id,
            pp_eval=config.pp_eval_subpopl_cifar,
            data_dir=config.subpopl_cifar_data_file)
        for client_id in dataset_builder.client_ids
    }

  if config.get('eval_on_cifar_10h'):
    cifar10_to_cifar10h_fn = data_uncertainty_utils.create_cifar10_to_cifar10h_fn(
        config.get('data_dir', None))
    preprocess_fn = preprocess_spec.parse(
        spec=config.pp_eval_cifar_10h, available_ops=preprocess_utils.all_ops())
    pp_eval = lambda ex: preprocess_fn(cifar10_to_cifar10h_fn(ex))
    val_ds_splits['cifar_10h'] = _get_val_split(
        'cifar10',
        split=config.get('cifar_10h_split') or 'test',
        pp_eval=pp_eval,
        data_dir=config.get('data_dir'))
  elif config.get('eval_on_imagenet_real'):
    imagenet_to_real_fn = data_uncertainty_utils.create_imagenet_to_real_fn()
    preprocess_fn = preprocess_spec.parse(
        spec=config.pp_eval_imagenet_real,
        available_ops=preprocess_utils.all_ops())
    pp_eval = lambda ex: preprocess_fn(imagenet_to_real_fn(ex))  # pytype: disable=wrong-arg-types
    val_ds_splits['imagenet_real'] = _get_val_split(
        'imagenet2012_real',
        split=config.get('imagenet_real_split') or 'validation',
        pp_eval=pp_eval,
        data_dir=config.get('data_dir'))

  ood_ds = {}
  if config.get('ood_datasets') and config.get('ood_methods'):
    if config.get('ood_methods'):  #  config.ood_methods is not a empty list
      logging.info('loading OOD dataset = %s', config.get('ood_datasets'))
      ood_ds, ood_ds_names = ood_utils.load_ood_datasets(
          config.dataset,
          config.ood_datasets,
          config.ood_split,
          config.pp_eval,
          config.pp_eval_ood,
          config.ood_methods,
          config.train_split,
          config.get('data_dir'),
          _get_val_split,
      )

  write_note('Initializing model...')
  model_config = config.model
  logging.info('config.model = %s', model_config)
  model_cls = getattr(models, model_config['name'])
  model = model_cls(**model_config)

  # We define the main prediction function.
  def single_model_pred_fn(params, images):
    rngs = {}  # No rngs needed since the prediction is fully deterministic.
    (logits, _), intermediates = model.apply(
        dict(params=params), images, rngs=rngs, capture_intermediates=True)
    prelogits = intermediates['intermediates']['pre_logits']['__call__'][0]
    return logits, prelogits

  efficient_ensemble_size = model_config['encoder']['moe'].get('ensemble_size')
  if efficient_ensemble_size is not None:
    reshape_outputs_fn = functools.partial(
        vmoe_utils.efficient_ensemble_reshape_outputs_fn,
        ensemble_size=efficient_ensemble_size)
  else:
    reshape_outputs_fn = vmoe_utils.deep_ensemble_reshape_outputs_fn

  pred_fn = functools.partial(ensemble_pred_fn, single_model_pred_fn,
                              reshape_outputs_fn)

  # We configure the mesh for pjit.
  num_experts = model_config['encoder']['moe']['num_experts']
  mesh = partitioning.get_auto_logical_mesh(num_experts, jax.devices())

  # We load the parameters from the checkpoint.
  write_note('Load checkpoint...')
  unpartitioned_params = load_checkpoint(config, mesh, efficient_ensemble_size)

  # We partition the params across the devices.
  variables_partition_spec = vmoe_utils.get_variables_partition_spec(
      unpartitioned_params)
  in_axis_resources = (
      variables_partition_spec,  # params.
      jax.sharding.PartitionSpec(('expert', 'replica')),  # inputs.
      jax.sharding.PartitionSpec(('expert', 'replica')),  # labels.
      jax.sharding.PartitionSpec(('expert', 'replica')),  # masks.
  )
  params = {}
  for model_key, model_params in unpartitioned_params.items():
    pjit_partition_params_fn = pjit.pjit(
        fun=lambda x: x,
        in_shardings=(
            jax.tree_map(lambda _: jax.sharding.PartitionSpec(), model_params),
        ),
        out_shardings=variables_partition_spec[model_key],
    )
    with mesh:
      params[model_key] = pjit_partition_params_fn(model_params)
  del unpartitioned_params
  params = flax.core.freeze(params)

  # We define the evaluation functions.
  def evaluation_fn(params, images, labels, mask):
    _check_pmap_and_pjit_shapes(images, labels, mask)
    images = _reshape_from_pmap_shape(images)
    labels = _reshape_from_pmap_shape(labels)
    mask = _reshape_from_pmap_shape(mask)
    # Ignore the entries with all zero labels for evaluation.
    mask *= (labels.max(axis=1) > 0).astype(labels.dtype)
    loss_as_str = config.get('loss', 'sigmoid_xent')
    logits, prelogits = pred_fn(params, images, loss_as_str)
    label_indices = config.get('label_indices')
    logging.info('!!! mask %s, label_indices %s', mask, label_indices)
    if label_indices:
      logits = logits[:, label_indices]

    # Note that logits and labels are usually of the shape [batch,num_classes].
    # But for OOD data, when num_classes_ood > num_classes_ind, we need to
    # adjust labels to labels[:, :config.num_classes] to match the shape of
    # logits. That is just to avoid shape mismatch. The output losses does not
    # have any meaning for OOD data, because OOD not belong to any IND class.
    losses = getattr(train_utils, loss_as_str)(
        logits=logits,
        labels=labels[:, :(len(label_indices) if label_indices
                           else config.num_classes)], reduction=False)
    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    # The different outer [...] are to match the pmap shapes expected after.
    loss = [jnp.sum(losses * mask, axis=0)]
    ncorrect = [jnp.sum(top1_correct * mask, axis=0)]
    n = [jnp.sum(mask, axis=0)]
    metric_args = [logits, labels, prelogits, mask]
    metric_args = [jnp.asarray([[args]]) for args in metric_args]

    return ncorrect, loss, n, metric_args

  evaluation_fn = vmoe_utils.get_pjit_eval_fn_with_mesh(
      evaluation_fn, mesh, in_axis_resources, num_outputs=4)

  def cifar_10h_evaluation_fn(params, images, labels, mask):
    _check_pmap_and_pjit_shapes(images, labels, mask)
    images = _reshape_from_pmap_shape(images)
    labels = _reshape_from_pmap_shape(labels)
    mask = _reshape_from_pmap_shape(mask)
    loss_as_str = config.get('loss', 'softmax_xent')
    logits, prelogits = pred_fn(params, images, loss_as_str)
    label_indices = config.get('label_indices')
    if label_indices:
      logits = logits[:, label_indices]

    losses = getattr(train_utils, loss_as_str)(
        logits=logits, labels=labels, reduction=False)
    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    one_hot_labels = jnp.eye(10)[jnp.argmax(labels, axis=1)]
    top1_correct = jnp.take_along_axis(
        one_hot_labels, top1_idx[:, None], axis=1)[:, 0]
    # The different outer [...] are to match the pmap shapes expected after.
    loss = [jnp.sum(losses, axis=0)]
    ncorrect = [jnp.sum(top1_correct, axis=0)]
    n = [jnp.sum(one_hot_labels, axis=0)]
    metric_args = [logits, labels, prelogits, mask]
    metric_args = [jnp.asarray([[args]]) for args in metric_args]

    return ncorrect, loss, n, metric_args

  cifar_10h_evaluation_fn = vmoe_utils.get_pjit_eval_fn_with_mesh(
      cifar_10h_evaluation_fn, mesh, in_axis_resources, num_outputs=4)

  # Setup function for computing representation.
  def representation_fn(params, images, labels, mask):
    _check_pmap_and_pjit_shapes(images, labels, mask)
    images = _reshape_from_pmap_shape(images)

    outputs = [single_model_pred_fn(p, images) for p in params.values()]
    _, prelogits = reshape_outputs_fn(outputs)
    prelogits = jnp.concatenate(prelogits, axis=1)

    # The outer [...] and reshape are to match the pmap shapes expected after.
    def reshape_to_pmap_all_gather_shape(x):
      assert x.ndim == 2, x.shape
      return [x.reshape((jax.device_count(), -1, x.shape[-1]))]

    representation = reshape_to_pmap_all_gather_shape(prelogits)
    labels = [labels]
    mask = [mask]
    return representation, labels, mask

  representation_fn = vmoe_utils.get_pjit_eval_fn_with_mesh(
      representation_fn, mesh, in_axis_resources, num_outputs=3)

  if jax.process_index() == 0:
    writer.write_hparams(dict(config))

  write_note('Initializing few-shotters...')
  fewshotter = None
  if 'fewshot' in config and fewshot is not None:
    fewshotter = fewshot.FewShotEvaluator(
        representation_fn, config.fewshot,
        config.fewshot.get('batch_size') or batch_size_eval)

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  val_loss = {val_name: -jnp.inf for val_name, _ in val_ds_splits.items()}
  fewshot_results = {'dummy': {(0, 1): -jnp.inf}}
  step = 1

  # Report validation performance.
  write_note('Evaluating on the validation set...')
  for val_name, val_ds in val_ds_splits.items():
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
    label_diversity = tf.keras.metrics.Mean()
    sample_diversity = tf.keras.metrics.Mean()
    ged = tf.keras.metrics.Mean()

    # Runs evaluation loop.
    val_iter = input_utils.start_input_pipeline(
        val_ds, config.get('prefetch_to_device', 1))
    ncorrect, loss, nseen = 0, 0, 0
    for batch in val_iter:
      if val_name == 'cifar_10h':
        batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
            cifar_10h_evaluation_fn(params, batch['image'],
                                    batch['labels'], batch['mask']))
      else:
        batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
            evaluation_fn(params, batch['image'],
                          batch['labels'], batch['mask']))
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical as they got psum'd.
      # So let's just take the first one to the host as numpy.
      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      loss += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))
      if config.get('loss', 'sigmoid_xent') != 'sigmoid_xent':
        # Here we parse batch_metric_args to compute uncertainty metrics.
        # (e.g., ECE or Calibration AUC).
        logits, labels, _, masks = batch_metric_args
        masks = np.array(masks[0], dtype=bool)
        logits = np.array(logits[0])
        probs = jax.nn.softmax(logits)
        # From one-hot to integer labels, as required by ECE.
        int_labels = np.argmax(np.array(labels[0]), axis=-1)
        int_preds = np.argmax(logits, axis=-1)
        confidence = np.max(probs, axis=-1)
        for p, c, l, d, m, label in zip(probs, confidence, int_labels,
                                        int_preds, masks, labels[0]):
          ece.add_batch(p[m, :], label=l[m])
          calib_auc.add_batch(d[m], label=l[m], confidence=c[m])
          # TODO(jereliu): Extend to support soft multi-class probabilities.
          oc_auc_0_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])
          oc_auc_1.add_batch(d[m], label=l[m], custom_binning_score=c[m])
          oc_auc_2.add_batch(d[m], label=l[m], custom_binning_score=c[m])
          oc_auc_5.add_batch(d[m], label=l[m], custom_binning_score=c[m])

          if val_name == 'cifar_10h' or val_name == 'imagenet_real':
            batch_label_diversity, batch_sample_diversity, batch_ged = data_uncertainty_utils.generalized_energy_distance(
                label[m], p[m, :], config.num_classes)
            label_diversity.update_state(batch_label_diversity)
            sample_diversity.update_state(batch_sample_diversity)
            ged.update_state(batch_ged)

    val_loss[val_name] = loss / nseen  # Keep for reproducibility tests.
    val_measurements = {
        f'{val_name}_prec@1': ncorrect / nseen,
        f'{val_name}_loss': val_loss[val_name],
    }
    if config.get('loss', 'sigmoid_xent') != 'sigmoid_xent':
      val_measurements[f'{val_name}_ece'] = ece.result()['ece']
      val_measurements[f'{val_name}_calib_auc'] = calib_auc.result()[
          'calibration_auc']
      val_measurements[f'{val_name}_oc_auc_0.5%'] = oc_auc_0_5.result()[
          'collaborative_auc']
      val_measurements[f'{val_name}_oc_auc_1%'] = oc_auc_1.result()[
          'collaborative_auc']
      val_measurements[f'{val_name}_oc_auc_2%'] = oc_auc_2.result()[
          'collaborative_auc']
      val_measurements[f'{val_name}_oc_auc_5%'] = oc_auc_5.result()[
          'collaborative_auc']
    writer.write_scalars(step, val_measurements)

    if val_name == 'cifar_10h' or val_name == 'imagenet_real':
      cifar_10h_measurements = {
          f'{val_name}_label_diversity': label_diversity.result(),
          f'{val_name}_sample_diversity': sample_diversity.result(),
          f'{val_name}_ged': ged.result(),
      }
      writer.write_scalars(step, cifar_10h_measurements)

  # OOD eval
  # Entries in the ood_ds dict include:
  # (ind_dataset, ood_dataset1, ood_dataset2, ...).
  # OOD metrics are computed using ind_dataset paired with each of the
  # ood_dataset. When Mahalanobis distance method is applied, train_ind_ds
  # is also included in the ood_ds.
  if ood_ds and config.ood_methods:
    ood_measurements = ood_utils.eval_ood_metrics(
        ood_ds,
        ood_ds_names,
        config.ood_methods,
        evaluation_fn,
        params,
        n_prefetch=config.get('prefetch_to_device', 1))
    writer.write_scalars(step, ood_measurements)

  # Perform subpopulation shift evaluation only if flag is provided.
  if config.get('subpopl_cifar_data_file'):
    subpopl_measurements = subpopl_utils.eval_subpopl_metrics(
        subpopl_val_ds_splits,
        evaluation_fn,
        params,
        n_prefetch=config.get('prefetch_to_device', 1))
    writer.write_scalars(step, scalars=subpopl_measurements)

  if 'fewshot' in config and fewshotter is not None:
    # Compute few-shot on-the-fly evaluation.
    write_note('Few-shot evaluation...')
    # Keep `results` to return for reproducibility tests.
    fewshot_results, best_l2 = fewshotter.run_all(params,
                                                  config.fewshot.datasets)

    # TODO(dusenberrymw): Remove this once fewshot.py is updated.
    def make_writer_measure_fn(step):
      def writer_measure(name, value):
        writer.write_scalars(step, {name: value})
      return writer_measure

    fewshotter.walk_results(
        make_writer_measure_fn(step), fewshot_results, best_l2)

  write_note('Done!')
  pool.close()
  pool.join()
  writer.close()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  return val_loss, fewshot_results

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
