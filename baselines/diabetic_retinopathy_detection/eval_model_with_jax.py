import os
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import contextlib
import pathlib

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
  'k', 1,
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
flags.DEFINE_bool(
  'single_model_multi_train_seeds', False,
  'If true, then evaluate a single model with multiple train seeds instead of multiple '
  'evaluation seeds. If this option is true, then all the checkpoints in the'
  '`checkpoint_dir` will be loaded')
flags.DEFINE_integer(
  'seed', 0,
  'Used as evaluation seed when `single_model_multi_train_seeds` is True.')
flags.DEFINE_integer(
  'k_ensemble_members', 3,
  'The number of models to sample without replacement from a directory of ensemble checkpoints.')
flags.DEFINE_integer(
  'ensemble_sampling_repetitions', 0,
  'The number of times to sample a subset of models from a directory of ensemble checkpoints.')
flags.DEFINE_string('chkpt_bucket', 'drd-final-eval-multi-seeds',
  'The name of the bucket containing checkpoints.')
flags.DEFINE_string('output_bucket', 'drd-final-results-multi-seeds',
  'The name of the output bucket.')
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
  assert not ("fsvi" in model_type and FLAGS.use_distribution_strategy), \
    "This script doesn't support running FSVI in parallel yet."
  single_model_multi_train_seeds = FLAGS.single_model_multi_train_seeds
  k = FLAGS.k
  N = FLAGS.ensemble_sampling_repetitions
  sample_from_ensemble = N > 0
  use_ensemble = k > 1 or sample_from_ensemble
  assert not (use_ensemble and single_model_multi_train_seeds), "can not both use ensemble and" \
                                                                " single_model_multi_train_seeds"
  k_ensemble_members = FLAGS.k_ensemble_members
  dist_shift = FLAGS.distribution_shift
  tuning_domain = FLAGS.tuning_domain
  n_samples = FLAGS.num_mc_samples
  checkpoint_dir = (
      f'gs://{FLAGS.chkpt_bucket}/{dist_shift}/{model_type}_k{k}_{tuning_domain}')
  output_suffix = "single" if single_model_multi_train_seeds else "ensemble"
  output_dir = (
      f'gs://{FLAGS.output_bucket}/{dist_shift}/'
      f'{model_type}_k{k}_{tuning_domain}_mc{n_samples}/{output_suffix}')

  if ("radial" in model_type or "rank1" in model_type) \
          and dist_shift == "severity" \
          and tuning_domain == "joint":
    logging.info(f"Doesn't have to run {model_type}, {dist_shift}, {tuning_domain}!")
    wandb_run.finish()
    return


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

  eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores
  num_mc_samples = FLAGS.num_mc_samples

  # Get the wrapper function which will produce uncertainty estimates for
  # our choice of method and Y/N ensembling.
  model_type_str = "variational_inference" if model_type == "vi" else model_type
  uncertainty_estimator_fn = utils.get_uncertainty_estimator(
    model_type_str, use_ensemble=use_ensemble,
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
      model_str = "Jax" if "fsvi" in model_type else "Keras"
      logging.info(f'Loading {model_str} ResNet-50 {model_type} {ensemble_str}.')
      if "fsvi" in model_type:
        model = utils.load_fsvi_jax_checkpoints(
              checkpoint_dir=checkpoint_dir,
              load_ensemble=use_ensemble or single_model_multi_train_seeds,
              return_epoch=False)
      else:
        model = utils.load_keras_checkpoints(
            checkpoint_dir=checkpoint_dir,
            load_ensemble=use_ensemble or single_model_multi_train_seeds,
            return_epoch=False
          )
      logging.info('Successfully loaded.')

      if sample_from_ensemble:
        assert len(model) >= 6, "Running in sample_from_ensemble mode, but only found " \
                                f"{len(model)} checkpoints in folder {checkpoint_dir}"
      # logging.info(f'Loaded model from epoch {epoch}.')

      # Wrap models: apply sigmoid on logits, use mixed precision,
      # and cast to NumPy array for use with generic Uncertainty Utils.
      if "fsvi" in model_type:
        if use_ensemble or single_model_multi_train_seeds:
          estimator = [m.model for m in model]
        else:
          estimator = model.model
      else:
        if use_ensemble or single_model_multi_train_seeds:
          estimator = [
            utils.wrap_retinopathy_estimator(
              loaded_model, use_mixed_precision=FLAGS.use_bfloat16,
              numpy_outputs=not FLAGS.use_distribution_strategy)
            for loaded_model in model]
        else:
          estimator = utils.wrap_retinopathy_estimator(
            model, use_mixed_precision=FLAGS.use_bfloat16,
            numpy_outputs=not FLAGS.use_distribution_strategy)

  assert not sample_from_ensemble or len(estimator) >= k_ensemble_members, \
    f"The number of models in the ensemble ({len(estimator)}) " \
    f"is smaller than the number of samples we want to draw {k_ensemble_members}," \
    f" it is not possible to sample without replacement."
  estimator_args = {}

  is_deterministic_single_model = model_type == 'deterministic' and not use_ensemble

  if model_type != 'deterministic':
    # Argument for stochastic forward passes
    # we don't sample MC samples for either a single deterministic model or deep ensemble
    estimator_args['num_samples'] = num_mc_samples
  if "fsvi" in model_type:
    if use_ensemble or single_model_multi_train_seeds:
      estimator_args["params"] = [m.params for m in model]
      estimator_args["state"] = [m.state for m in model]
    else:
      estimator_args["params"] = model.params
      estimator_args["state"] = model.state

  scalar_results_arr = []

  def set_seeds(eval_seed):
    logging.info(f'Evaluating with eval_seed: {eval_seed}.')

    # Set seeds
    tf.random.set_seed(eval_seed)
    np.random.seed(eval_seed)

  def iter_step(eval_seed, estimator_args, estimator, scalar_results_arr, iter_id):
    if "fsvi" in model_type:
      estimator_args["rng_key"] = jax.random.PRNGKey(eval_seed)

    per_pred_results, scalar_results = utils.eval_model_numpy(
      datasets, steps, estimator, estimator_args, uncertainty_estimator_fn,
      eval_batch_size, is_deterministic_single_model,
      distribution_shift=FLAGS.distribution_shift,
      num_bins=FLAGS.num_bins, np_input="fsvi" in model_type)

    # Store scalar results
    scalar_results_arr.append(scalar_results)

    # Save all predictions, ground truths, uncertainty measures, etc.
    # as NumPy arrays, for use with the plotting module.
    utils.save_per_prediction_results(
      output_dir, epoch=iter_id,
      per_prediction_results=per_pred_results, verbose=True,
      allow_overwrite=True,
    )

  # Evaluation Loop
  if single_model_multi_train_seeds:
    for model_index in range(len(estimator)):
      logging.info(f"Evaluating the {model_index}-th trained model")
      set_seeds(FLAGS.seed)
      if "fsvi" in model_type:
        _estimator_args = {
          "num_samples": estimator_args["num_samples"],
          "params": estimator_args["params"][model_index],
          "state": estimator_args["state"][model_index],
        }
      else:
        _estimator_args = estimator_args
      iter_step(
        eval_seed=FLAGS.seed,
        estimator_args=_estimator_args,
        estimator=estimator[model_index],
        scalar_results_arr=scalar_results_arr,
        iter_id=model_index,
      )
  elif sample_from_ensemble:
    for rep_index in range(N):
      logging.info(f"Evaluating by sampling {k_ensemble_members} models from an ensemble "
                   f"of {len(estimator)} models, currently at repetition {rep_index}/{N}")
      set_seeds(FLAGS.seed)
      sampled_indices = np.random.choice(len(estimator), size=k_ensemble_members, replace=False)
      sampled_estimator = [estimator[ind] for ind in sampled_indices]
      if "fsvi" in model_type:
        _estimator_args = {
          "num_samples": estimator_args["num_samples"],
          "params": [estimator_args["params"][ind] for ind in sampled_indices],
          "state": [estimator_args["state"][ind] for ind in sampled_indices],
        }
      else:
        _estimator_args = estimator_args
      iter_step(
        eval_seed=FLAGS.seed,
        estimator_args=_estimator_args,
        estimator=sampled_estimator,
        scalar_results_arr=scalar_results_arr,
        iter_id=rep_index,
      )
  else:
    for eval_seed in range(FLAGS.num_eval_seeds):
      set_seeds(eval_seed)
      iter_step(
        eval_seed=eval_seed,
        estimator_args=estimator_args,
        estimator=estimator,
        scalar_results_arr=scalar_results_arr,
        iter_id=eval_seed,
      )

  # Scalar results stored as pd.DataFrame
  utils.merge_and_store_scalar_results(
    scalar_results_arr, output_dir=output_dir,
    allow_overwrite=True,
  )
  logging.info('Wrote out scalar results.')

  wandb_run.finish()

# def get_per_level_histogram():

if __name__ == '__main__':
  app.run(main)
