import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import contextlib
import pathlib
import pickle

import jax
import numpy as np
import tensorflow as tf
import torch
import wandb
from absl import app
from absl import flags
from absl import logging

import utils  # local file import

# Logging and hyperparameter tuning.
flags.DEFINE_string('wandb_dir', 'wandb', 'Directory where wandb logs go.')
flags.DEFINE_string('project', 'ub-debug', 'Wandb project name.')
flags.DEFINE_string('exp_name', None, 'Give experiment a name.')
flags.DEFINE_string('exp_group', None, 'Give experiment a group name.')

# Data load / output flags.
# TODO: update with uncertainty wrappers
flags.DEFINE_string(
    'model_type', None,
    'The type of model being loaded and evaluated. This is used to retrieve '
    'the correct wrapper for obtaining uncertainty estimates, as implemented '
    'in TODO.py.')
flags.mark_flag_as_required('model_type')
# flags.DEFINE_string(
#     'checkpoint_dir', None,  # eg. '/tmp/diabetic_retinopathy_detection/dropout'
#     'The directory from which the trained model weights are retrieved.')
# flags.mark_flag_as_required('checkpoint_dir')
# flags.DEFINE_string(
#     'output_dir',
#     '/tmp/diabetic_retinopathy_detection/evaluation_results',
#     'The directory where the evaluation summaries and DataFrame results '
#     'are stored.')
flags.DEFINE_string('data_dir', 'gs://ub-data/retinopathy',
                    'Path to evaluation data.')
# TODO: fix bug that requires us to specify True here
flags.DEFINE_bool('use_validation', True,
                  'Whether to use a validation split.')
flags.DEFINE_bool('use_test', True, 'Whether to use a test split.')
flags.DEFINE_bool('cache_eval_datasets', False, 'Caches eval datasets.')
flags.DEFINE_string(
  'dr_decision_threshold', None,
  ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
   "binary classification task. Only affects the APTOS dataset partitioning. "
   "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
   "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))
flags.mark_flag_as_required('dr_decision_threshold')

# OOD Dataset flags.
flags.DEFINE_string(
  'distribution_shift', None,
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.DEFINE_bool(
  'load_train_split', False,
  "Should always be disabled - we are evaluating in this script.")

# General model flags.
flags.DEFINE_integer(
  'k', None,
  "Number of ensemble members.")

flags.DEFINE_integer(
    'num_eval_seeds', 10,
    'Number of random repetitions to use in evaluation.')
flags.DEFINE_bool('model_is_torch', False,
                  'If model to be loaded is a Torch model.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The per-core validation/test batch size.')

# Robustness and Uncertainty evaluation flags.
flags.DEFINE_string(
  'tuning_domain', None,
  "If tuning was done on `indomain` or `joint` (ID + OOD).")
flags.DEFINE_integer(
    'num_mc_samples', 5,
    'Number of Monte Carlo samples to use for prediction, if applicable.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')
flags.DEFINE_bool(
  'use_distribution_strategy', False,
  'If specified, use a distribution strategy.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg

  # Wandb init
  pathlib.Path(FLAGS.wandb_dir).mkdir(parents=True, exist_ok=True)
  wandb_args = dict(
      project=FLAGS.project,
      entity="uncertainty-baselines",
      dir=FLAGS.wandb_dir,
      reinit=True,
      name=FLAGS.exp_name,
      group=FLAGS.exp_group)
  wandb_run = wandb.init(**wandb_args)
  wandb.config.update(FLAGS, allow_val_change=True)

  model_type = FLAGS.model_type
  k = FLAGS.k
  use_ensemble = k > 1
  dist_shift = FLAGS.distribution_shift
  tuning_domain = FLAGS.tuning_domain
  n_samples = FLAGS.num_mc_samples
  checkpoint_dir = (
      f'gs://drd-final-eval/{dist_shift}/{model_type}_k{k}_{tuning_domain}')
  output_dir = (
      f'gs://drd-final-results/{dist_shift}/'
      f'{model_type}_k{k}_{tuning_domain}_mc{n_samples}')


  tf.io.gfile.makedirs(output_dir)
  logging.info(
    'Saving robustness and uncertainty evaluation results to %s',
    output_dir)

  # If model is Torch, set seed/device
  use_torch = FLAGS.model_is_torch
  strategy = None

  if use_torch:
    torch.manual_seed(FLAGS.seed)

    # Resolve CUDA device(s)
    if FLAGS.use_gpu and torch.cuda.is_available():
      print('Running model with CUDA.')
      device = 'cuda:0'
      # torch.backends.cudnn.benchmark = True
    else:
      print('Running model on CPU.')
      device = 'cpu'
  elif FLAGS.use_distribution_strategy:
    strategy = utils.init_distribution_strategy(
      FLAGS.force_use_cpu, FLAGS.use_gpu, FLAGS.tpu)
    use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  model_type = FLAGS.model_type
  eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores
  num_mc_samples = FLAGS.num_mc_samples

  # Get the wrapper function which will produce uncertainty estimates for
  # our choice of method and Y/N ensembling.
  uncertainty_estimator_fn = utils.get_uncertainty_estimator(
    model_type, use_ensemble=use_ensemble,
    use_tf=FLAGS.use_distribution_strategy)

  # Load in evaluation datasets.
  datasets, steps = utils.load_dataset(
    None, eval_batch_size, flags=FLAGS, strategy=strategy,
    load_for_eval=(not FLAGS.use_distribution_strategy))
  datasets = {
    'in_domain_test': datasets['in_domain_test'],
    'ood_test': datasets['ood_test']
  }

  if FLAGS.use_bfloat16 and not use_torch:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  # * Load Checkpoints *
  ensemble_str = 'ensemble' if use_ensemble else 'model'

  if use_torch:
    logging.info(f'Loading Torch ResNet-50 {model_type} {ensemble_str}.')
    raise NotImplementedError
  else:
    if strategy is None:
      logging.info('Not using a distribution strategy.')
      strategy_scope = contextlib.nullcontext()
    else:
      strategy_scope = strategy.scope()

    with strategy_scope:
      logging.info(f'Loading Keras ResNet-50 {model_type} {ensemble_str}.')
      if "fsvi" in model_type:
        model = utils.load_fsvi_jax_checkpoints(
              checkpoint_dir=checkpoint_dir, load_ensemble=use_ensemble,
                return_epoch=False)
      else:
        model = utils.load_keras_checkpoints(
            checkpoint_dir=checkpoint_dir, load_ensemble=use_ensemble,
            return_epoch=False
          )
      logging.info('Successfully loaded.')
      # logging.info(f'Loaded model from epoch {epoch}.')

      # Wrap models: apply sigmoid on logits, use mixed precision,
      # and cast to NumPy array for use with generic Uncertainty Utils.
      if "fsvi" in model_type:
        if use_ensemble:
          estimator = [m.model for m in model]
        else:
          estimator = model.model
      else:
        if use_ensemble:
          estimator = [
            utils.wrap_retinopathy_estimator(
              loaded_model, use_mixed_precision=FLAGS.use_bfloat16,
              numpy_outputs=not FLAGS.use_distribution_strategy)
            for loaded_model in model]
          k = len(estimator)
        else:
          estimator = utils.wrap_retinopathy_estimator(
            model, use_mixed_precision=FLAGS.use_bfloat16,
            numpy_outputs=not FLAGS.use_distribution_strategy)
          k = None

  estimator_args = {}

  is_deterministic = model_type == 'deterministic'

  if not is_deterministic:
    # Argument for stochastic forward passes
    estimator_args['num_samples'] = num_mc_samples
  if "fsvi" in model_type:
    if use_ensemble:
      estimator_args["params"] = [m.params for m in model]
      estimator_args["state"] = [m.state for m in model]
    else:
      estimator_args["params"] = model.params
      estimator_args["state"] = model.state

  scalar_results_arr = []

  # Evaluation Loop
  for eval_seed in range(FLAGS.num_eval_seeds):
    logging.info(f'Evaluating with eval_seed: {eval_seed}.')

    # Set seeds
    tf.random.set_seed(eval_seed)
    np.random.seed(eval_seed)
    if "fsvi" in model_type:
      estimator_args["rng_key"] = jax.random.PRNGKey(eval_seed)

    per_pred_results, scalar_results = utils.eval_model_numpy(
      datasets, steps, estimator, estimator_args, uncertainty_estimator_fn,
      eval_batch_size, is_deterministic,
      distribution_shift=FLAGS.distribution_shift,
      num_bins=FLAGS.num_bins, np_input="fsvi" in model_type)

    # Store scalar results
    scalar_results_arr.append(scalar_results)

    # Save all predictions, ground truths, uncertainty measures, etc.
    # as NumPy arrays, for use with the plotting module.
    utils.save_per_prediction_results(
      output_dir, epoch=eval_seed,
      per_prediction_results=per_pred_results, verbose=True)

  # Scalar results stored as pd.DataFrame
  utils.merge_and_store_scalar_results(
    scalar_results_arr, output_dir=output_dir)
  logging.info('Wrote out scalar results.')

  wandb_run.finish()

# def get_per_level_histogram():

if __name__ == '__main__':
  app.run(main)
