import functools
import logging

import flax
import jax
import numpy as np
from absl import logging
from clu import parameter_overview
from experiments.config.imagenet21k_vit_large32_finetune import (
  get_config as vit_l32_i21k_config)

import input_utils  # local file import
import uncertainty_baselines as ub
from eval_utils import (
  add_joint_dicts, compute_loss_and_accuracy_arrs_for_all_datasets,
  compute_metrics_for_all_datasets)
from experiments.config.drd_vit_base16 import (
  get_config as vit_b16_no_pretrain_config)
from experiments.config.imagenet21k_vit_base16_finetune import (
  get_config as vit_b16_i21k_config)
from experiments.config.imagenet21k_vit_base16_sngp_finetune import (
  get_config as sngp_vit_b16_i21k_config)
from metric_utils import log_vit_validation_metrics

# Mapping from (model_type, vit_model_size, pretrain_dataset) to config.
VIT_CONFIG_MAP = {
  ('deterministic', 'B/16', 'imagenet21k'): vit_b16_i21k_config,
  ('deterministic', 'B/16', None): vit_b16_no_pretrain_config,  # No pretraining
  ('deterministic', 'L/32', 'imagenet21k'): vit_l32_i21k_config,
  ('sngp', 'B/16', 'imagenet21k'): sngp_vit_b16_i21k_config
}

import pathlib
import wandb
import datetime
import os


def get_dataset_and_split_names(dist_shift):
  dataset_names = {}
  split_names = {}

  if dist_shift == 'aptos':
    dataset_names['in_domain_dataset'] = 'ub_diabetic_retinopathy_detection'
    dataset_names['ood_dataset'] = 'aptos'
    split_names['train_split'] = 'train'
    split_names['in_domain_val_split'] = 'validation'
    split_names['ood_val_split'] = 'validation'
    split_names['in_domain_test_split'] = 'test'
    split_names['ood_test_split'] = 'test'
  elif dist_shift == 'severity':
    dataset_names[
      'in_domain_dataset'] = 'diabetic_retinopathy_severity_shift_moderate'
    dataset_names[
      'ood_dataset'] = 'diabetic_retinopathy_severity_shift_moderate'
    split_names['train_split'] = 'train'
    split_names['in_domain_val_split'] = 'in_domain_validation'
    split_names['ood_val_split'] = 'ood_validation'
    split_names['in_domain_test_split'] = 'in_domain_test'
    split_names['ood_test_split'] = 'ood_test'
  else:
    raise NotImplementedError

  return dataset_names, split_names


def maybe_setup_wandb(flags):
  # Wandb Setup
  if flags.use_wandb:
    pathlib.Path(flags.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb_args = dict(
        project=flags.project,
        entity='uncertainty-baselines',
        dir=flags.wandb_dir,
        reinit=True,
        name=flags.exp_name,
        group=flags.exp_group)
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(flags, allow_val_change=True)
    output_dir = str(os.path.join(
        flags.output_dir,
        datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
  else:
    wandb_run = None
    output_dir = flags.output_dir

  return wandb_run, output_dir

def write_note(note):
  if jax.process_index() == 0:
    logging.info('NOTE: %s', note)


def get_vit_config(model_type, vit_model_size, pretrain_dataset):
  config_key = (model_type, vit_model_size, pretrain_dataset)
  if config_key not in VIT_CONFIG_MAP:
    raise NotImplementedError(f'No config found for key: {config_key}')

  write_note(f'Retrieving ViT config with key: {config_key}')
  return VIT_CONFIG_MAP[config_key]()


# Utility functions.
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

VIT_MODEL_INIT_MAP = {
  'deterministic': initialize_deterministic_model,
  'sngp': initialize_sngp_model
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
  dataset_split_to_containers,
  is_deterministic,
  num_bins=15,
  return_per_pred_results=False
):
  """Compute evaluation metrics given ViT predictions.

  Args:
    dataset_split_to_containers: Dict, for each dataset, contains `np.array`
      predictions, ground truth, and uncertainty estimates.
    is_deterministic: bool, is the model a single deterministic network.
      In this case, we cannot capture epistemic uncertainty.
    num_bins: int, number of bins to use with expected calibration error.
    return_per_pred_results: bool,
  Returns:
    Union[Tuple[Dict, Dict], Dict]
      If return_per_pred_results, return two Dicts. Else, return only the second.
      first Dict:
        for each dataset, per-prediction results (e.g., each prediction,
        ground-truth, loss, retention arrays).
      second Dict:
        for each dataset, contains `np.array` predictions, ground truth,
        and uncertainty estimates.
  """
  eval_results = add_joint_dicts(
    dataset_split_to_containers, is_deterministic=is_deterministic)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = compute_loss_and_accuracy_arrs_for_all_datasets(eval_results)

  # Compute all metrics for each dataset --
  # Robustness, Open Set Recognition, Retention AUC
  metrics_results = compute_metrics_for_all_datasets(
    eval_results, use_precomputed_arrs=False, ece_num_bins=num_bins,
    compute_retention_auc=True,
    verbose=False)

  # Log metrics
  log_vit_validation_metrics(metrics_results)

  if return_per_pred_results:
    return eval_results, metrics_results
  else:
    return metrics_results

import jax.numpy as jnp


# We want all parameters to be created in host RAM, not on any device, they'll
# be sent there later as needed, otherwise we already encountered two
# situations where we allocate them twice.
@functools.partial(jax.jit, backend='cpu')
def init_deterministic_params(image_size,
                              local_batch_size,
                              model_dict,
                              rng,
                              config):
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
    write_note('Initializing val dataset(s)...')
    datasets['in_domain_validation'] = get_dataset(
        dataset_name=dataset_names['in_domain_dataset'],
        split_name=split_names['in_domain_validation_split'])
    datasets['ood_validation'] = get_dataset(
        dataset_name=dataset_names['ood_dataset'],
        split_name=split_names['ood_validation_split'])
  if use_test:
    write_note('Initializing test dataset(s)...')
    datasets['in_domain_test'] = get_dataset(
        dataset_name=dataset_names['in_domain_dataset'],
        split_name=split_names['in_domain_test_split'])
    datasets['ood_test'] = get_dataset(
        dataset_name=dataset_names['ood_dataset'],
        split_name=split_names['ood_test_split'])

  return datasets
