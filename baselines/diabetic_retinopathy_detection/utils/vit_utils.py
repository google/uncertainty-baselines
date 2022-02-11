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

"""ViT evaluation utilities."""

import datetime
import functools
import logging
import os
import pathlib
from typing import Any, Dict, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy

import uncertainty_baselines as ub
import input_utils  # local file import from baselines.diabetic_retinopathy_detection
from . import eval_utils  # local file import
from . import metric_utils  # local file import
from . import results_storage_utils  # local file import
import wandb

Array = Any


def get_dataset_and_split_names(dist_shift):
  """Gets dataset and split names."""
  dataset_names = {}
  split_names = {}

  if dist_shift == 'aptos':
    dataset_names['in_domain_dataset'] = 'ub_diabetic_retinopathy_detection'
    dataset_names['ood_dataset'] = 'aptos'
    split_names['train_split'] = 'train'
    split_names['in_domain_validation_split'] = 'validation'
    split_names['ood_validation_split'] = 'validation'
    split_names['in_domain_test_split'] = 'test'
    split_names['ood_test_split'] = 'test'
  elif dist_shift == 'severity':
    dataset_names[
        'in_domain_dataset'] = 'diabetic_retinopathy_severity_shift_moderate'
    dataset_names[
        'ood_dataset'] = 'diabetic_retinopathy_severity_shift_moderate'
    split_names['train_split'] = 'train'
    split_names['in_domain_validation_split'] = 'in_domain_validation'
    split_names['ood_validation_split'] = 'ood_validation'
    split_names['in_domain_test_split'] = 'in_domain_test'
    split_names['ood_test_split'] = 'ood_test'
  else:
    raise NotImplementedError

  return dataset_names, split_names


def maybe_setup_wandb(config):
  """Potentially setup wandb."""
  if config.use_wandb:
    pathlib.Path(config.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb_args = dict(
        project=config.project,
        entity='uncertainty-baselines',
        dir=config.wandb_dir,
        reinit=True,
        name=config.exp_name,
        group=config.exp_group)
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(config, allow_val_change=True)
    output_dir = str(os.path.join(
        config.output_dir,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
  else:
    wandb_run = None
    output_dir = config.output_dir

  return wandb_run, output_dir


def write_note(note):
  if jax.process_index() == 0:
    logging.info('NOTE: %s', note)


def accumulate_gradient_with_states(
    loss_and_grad_fn,
    params,
    states,  # Allows for states.
    images,
    labels,
    accum_steps):
  """Improved version of `train_utils.accumulate_gradient()` that allows for states."""
  # This function handles the `loss_and_grad_fn` function which takes a state
  # argument and returns ((losses, states), grads).
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
    step_size = images.shape[0] // accum_steps

    # Run the first step.
    (l, s), g = loss_and_grad_fn(params, states, images[:step_size],
                                 labels[:step_size])

    # Run the rest of the steps.
    def acc_grad_and_loss(i, l_s_g):
      # Extract data for current step.
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      # Update state and accumulate gradient.
      l, s, g = l_s_g
      (li, si), gi = loss_and_grad_fn(params, s, imgs, lbls)
      return (l + li, si, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, s, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, s, g))
    l, g = jax.tree_map(lambda x: x / accum_steps, (l, g))
    return (l, s), g
  else:
    return loss_and_grad_fn(params, states, images, labels)


def get_eval_split(dataset,
                   split,
                   preproc_fn,
                   config,
                   batch_size_eval,
                   local_batch_size_eval):
  """Gets evaluation split."""
  data_dir = config.get('data_dir')

  # We do ceil rounding such that we include the last incomplete batch.
  n_eval_img = input_utils.get_num_examples(
      dataset,
      split=split,
      process_batch_size=local_batch_size_eval,
      drop_remainder=False,
      data_dir=data_dir)
  eval_steps = int(np.ceil(n_eval_img / batch_size_eval))
  logging.info('Running evaluation for %d steps for %s, %s', eval_steps,
               dataset, split)
  eval_ds = input_utils.get_data(
      dataset=dataset,
      split=split,
      rng=None,
      process_batch_size=local_batch_size_eval,
      preprocess_fn=preproc_fn,
      cache=False,
      repeat_after_batching=True,
      shuffle=False,
      prefetch_size=config.get('prefetch_to_host', 2),
      drop_remainder=False,
      data_dir=data_dir)
  eval_iter = input_utils.start_input_pipeline(
      eval_ds, config.get('prefetch_to_device', 1))
  return eval_iter, eval_steps


def initialize_deterministic_model(config):
  logging.info('config.model = %s', config.get('model'))
  model = ub.models.vision_transformer(
      num_classes=config.num_classes, **config.get('model', {}))
  return {
      'model': model
  }


def initialize_sngp_model(config):
  """Initializes SNGP model."""
  # Specify Gaussian process layer configs.
  use_gp_layer = config.get('use_gp_layer', True)
  gp_config = config.get('gp_layer', {})
  gp_layer_kwargs = get_gp_kwargs(gp_config)

  # Process ViT backbone model configs.
  vit_kwargs = config.get('model')

  model = ub.models.vision_transformer_gp(
      num_classes=config.num_classes,
      use_gp_layer=use_gp_layer,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)

  return {
      'model': model,
      'use_gp_layer': use_gp_layer
  }


def initialize_batchensemble_model(config):
  """Initialize BatchEnsemble model."""
  model = ub.models.PatchTransformerBE(
      num_classes=config.num_classes, **config.model)
  return {
    'model': model,
    'ens_size': config.model.transformer.ens_size
  }


VIT_MODEL_INIT_MAP = {
    'deterministic': initialize_deterministic_model,
    'sngp': initialize_sngp_model,
    'batchensemble': initialize_batchensemble_model
}


def initialize_model(model_type, config):
  if model_type not in VIT_MODEL_INIT_MAP.keys():
    raise NotImplementedError(f'No initialization method yet implemented '
                              f'for model_type {model_type}.')
  return VIT_MODEL_INIT_MAP[model_type](config)


def get_gp_kwargs(gp_config):
  """Extract keyword argument parameters for the Gaussian process layer."""
  covmat_momentum = gp_config.get('covmat_momentum', 0.999)

  # Extracts model parameter.
  logging.info('gp_config.covmat_momentum = %s', covmat_momentum)
  covmat_momentum = None if covmat_momentum < 0. else covmat_momentum
  covmat_kwargs = dict(momentum=covmat_momentum)

  # Assembles into kwargs dictionary.
  gp_layer_kwargs = dict(covmat_kwargs=covmat_kwargs)

  return gp_layer_kwargs


def evaluate_vit_predictions(
    dataset_split_to_containers: Dict[str, Dict[str, Array]],
    is_deterministic: bool,
    num_bins: int = 15,
    return_per_pred_results: bool = False
) -> Union[Dict[str, Dict[str, Array]],
           Tuple[Dict[str, Dict[str, Array]], Dict[str, Dict[str, Array]]]]:
  """Compute evaluation metrics given ViT predictions.

  Args:
    dataset_split_to_containers: Dictionary where each dataset is a dictionary
      with array-like predictions, ground truth, and uncertainty estimates.
    is_deterministic: Whether the model is a single deterministic network.
      In this case, we cannot capture epistemic uncertainty.
    num_bins: Number of bins to use with expected calibration error.
    return_per_pred_results: Whether to return per-prediction results.

  Returns:
    Tuple of dicts if return_per_pred_results else only the second.
      first Dict:
        for each dataset, per-prediction results (e.g., each prediction,
        ground-truth, loss, retention arrays).
      second Dict:
        for each dataset, `np.array` predictions, ground truth,
        and uncertainty estimates.
  """
  eval_results = results_storage_utils.add_joint_dicts(
      dataset_split_to_containers, is_deterministic=is_deterministic)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = eval_utils.compute_loss_and_accuracy_arrs_for_all_datasets(
      eval_results)

  # Compute all metrics for each dataset --
  # Robustness, Open Set Recognition, Retention AUC
  metrics_results = eval_utils.compute_metrics_for_all_datasets(
      eval_results, use_precomputed_arrs=False, ece_num_bins=num_bins,
      compute_retention_auc=True,
      verbose=False)

  # Log metrics
  metric_utils.log_vit_validation_metrics(metrics_results)

  if return_per_pred_results:
    return eval_results, metrics_results
  else:
    return metrics_results


# We want all parameters to be created in host RAM, not on any device, they'll
# be sent there later as needed, otherwise we already encountered two
# situations where we allocate them twice.
@functools.partial(jax.jit, backend='cpu')
def init_deterministic_params(image_size,
                              local_batch_size,
                              model_dict,
                              rng,
                              config):
  """Initializes deterministic."""
  model = model_dict['model']
  logging.info('image_size = %s', image_size)
  dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
  params = flax.core.unfreeze(model.init(rng, dummy_input,
                                         train=False))['params']

  # Set bias in the head to a low value, such that loss is small initially.
  params['head']['bias'] = jnp.full_like(
      params['head']['bias'], config.get('init_head_bias', 0))

  # init head kernel to all zeros for fine-tuning
  if config.get('model_init'):
    params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

  return params


# We want all parameters to be created in host RAM, not on any device, they'll
# be sent there later as needed, otherwise we already encountered two
# situations where we allocate them twice.
@functools.partial(jax.jit, backend='cpu')
def init_sngp_params(image_size,
                     local_batch_size,
                     model_dict,
                     rng,
                     config):
  """Initializes SNGP."""
  model, use_gp_layer = model_dict['model'], model_dict['use_gp_layer']
  logging.info('image_size = %s', image_size)
  dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
  variables = model.init(rng, dummy_input, train=False)
  # Split model parameters into trainable and untrainable collections.
  states, params = variables.pop('params')
  del variables

  # Set bias in the head to a low value, such that loss is small initially.
  params = flax.core.unfreeze(params)
  if use_gp_layer:
    # Modify the head parameter in the GP head.
    params['head']['output_layer']['bias'] = jnp.full_like(
        params['head']['output_layer']['bias'],
        config.get('init_head_bias', 0))
  else:
    params['head']['bias'] = jnp.full_like(
        params['head']['bias'], config.get('init_head_bias', 0))

  return params, states


MODEL_TYPE_TO_INIT_PARAMS_FN = {
    'deterministic': init_deterministic_params,
    'sngp': init_sngp_params
}


def init_evaluation_datasets(use_validation,
                             use_test,
                             dataset_names,
                             split_names,
                             config,
                             preproc_fn,
                             batch_size_eval,
                             local_batch_size_eval):
  """Sets up evaluation datasets."""
  data_dir = config.get('data_dir')
  def get_dataset(dataset_name, split_name):
    base_dataset = ub.datasets.get(
        dataset_name, split=split_name, data_dir=data_dir)
    dataset_builder = base_dataset._dataset_builder  # pylint:disable=protected-access
    return get_eval_split(
        dataset_builder,
        split_name,
        preproc_fn,
        config,
        batch_size_eval,
        local_batch_size_eval)

  datasets = {}
  if use_validation:
    datasets['in_domain_validation'] = get_dataset(
        dataset_name=dataset_names['in_domain_dataset'],
        split_name=split_names['in_domain_validation_split'])
    datasets['ood_validation'] = get_dataset(
        dataset_name=dataset_names['ood_dataset'],
        split_name=split_names['ood_validation_split'])
  if use_test:
    datasets['in_domain_test'] = get_dataset(
        dataset_name=dataset_names['in_domain_dataset'],
        split_name=split_names['in_domain_test_split'])
    datasets['ood_test'] = get_dataset(
        dataset_name=dataset_names['ood_dataset'],
        split_name=split_names['ood_test_split'])

  return datasets


def entropy(pk, qk=None, base=None, axis=0):
  """Calculate the entropy of a distribution for given probability values.

  If only probabilities `pk` are given, the entropy is calculated as
  ``S = -sum(pk * log(pk), axis=axis)``.
  If `qk` is not None, then compute the Kullback-Leibler divergence
  ``S = sum(pk * log(pk / qk), axis=axis)``.
  This routine will normalize `pk` and `qk` if they don't sum to 1.

  Args:
    pk: sequence
        Defines the (discrete) distribution. ``pk[i]`` is the (possibly
        unnormalized) probability of event ``i``.
    qk: sequence, optional
        Sequence against which the relative entropy is computed. Should be in
        the same format as `pk`.
    base: float, optional
        The logarithmic base to use, defaults to ``e`` (natural logarithm).
    axis: int, optional
        The axis along which the entropy is calculated. Default is 0.

  Returns:
    The calculated entropy.
  """
  if base is not None and base <= 0:
    raise ValueError('`base` must be a positive number or `None`.')

  pk = np.asarray(pk)
  pk = 1.0*pk / np.sum(pk, axis=axis, keepdims=True)
  if qk is None:
    vec = scipy.special.entr(pk)
  else:
    qk = np.asarray(qk)
    pk, qk = np.broadcast_arrays(pk, qk)
    qk = 1.0*qk / np.sum(qk, axis=axis, keepdims=True)
    vec = scipy.special.rel_entr(pk, qk)
  s = np.sum(vec, axis=axis)
  if base is not None:
    s /= np.log(base)
  return s
