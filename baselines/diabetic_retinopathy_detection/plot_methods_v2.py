from time import time

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

import utils  # local file import

# Data load / output flags.
# TODO: update with uncertainty wrappers
from baselines.diabetic_retinopathy_detection.utils.load_utils import fast_load_dataset_to_model_results

flags.DEFINE_string(
    'results_dir', None,  # eg. 'gs://drd-final-results/all-ensembles/'
    'The directory where model outputs (e.g., predictions, uncertainty '
    'estimates, ground truth values, retention curves are stored).'
    'We expect that subdirectories in this dir will be named with the format '
    '{model_type}_k{k}_{tuning_domain}_mc{n_samples} where `model_type` is '
    'the method used, `k` is the size of the ensemble, `tuning_domain` '
    'specifies if tuning was done on ID or ID+OOD metrics, and `n_samples` is '
    'the number of MC samples. Each of those directories should follow the '
    'format output by `eval_model_backup.py`.')
flags.mark_flag_as_required('results_dir')
flags.DEFINE_string(
    'output_dir',
    '/tmp/diabetic_retinopathy_detection/plots',
    'The directory where the plots are stored.')

# OOD Dataset flags.
flags.DEFINE_string(
  'distribution_shift', None,
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.mark_flag_as_required('distribution_shift')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info(
    'Saving robustness and uncertainty plots to %s', FLAGS.output_dir)

  distribution_shift = FLAGS.distribution_shift
  num_bins = FLAGS.num_bins
  results_dir = FLAGS.results_dir

  logging.info(f'Plotting for distribution shift {distribution_shift}.')

  start = time()
  # Contains a defaultdict for each dataset
  # Each dataset has a map from (model_type, k, tuning_domain, num_mc_samples)
  # to a final dict.
  # This dict has the below keys. The values are lists of np.arrays, one
  # array for each random seed.
  dataset_to_model_results = fast_load_dataset_to_model_results(
    results_dir=results_dir, invalid_cache=False
  )
  print("-"* 100)
  logging.info(f"used {time() - start:.2f} seconds to load")

  # use this
  # dataset_to_model_results
  # utils.plot_retention_curves(
  #   distribution_shift_name=distribution_shift,
  #   dataset_to_model_results=dataset_to_model_results, plot_dir='.')

  utils.plot_roc_curves(
    distribution_shift_name=distribution_shift,
    dataset_to_model_results=dataset_to_model_results, plot_dir='roc-plots')


if __name__ == '__main__':
  app.run(main)
