"""
Loads and packages the data for the aptos dataset.
"""

from absl import app
from absl import flags
from absl import logging

import uncertainty_baselines as ub

flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  data_dir = FLAGS.data_dir

  # APTOS has two splits: validation and test.

  # Load `aptos` validation data
  aptos_validation_builder = ub.datasets.get(
    "aptos",
    split='validation',
    data_dir=data_dir,
    download_data=True)
  logging.info('Shuffling and packaging `aptos` validation data.')
  aptos_validation_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/aptos/')

  # Load `aptos` test data
  aptos_test_builder = ub.datasets.get(
    "aptos",
    split='test',
    data_dir=data_dir,
    download_data=True)
  logging.info('Shuffling and packaging `aptos` test data.')
  aptos_test_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/aptos/')
  logging.info('Finished packaging `aptos` test data.')

  logging.info('Finished packaging `aptos` data.')


if __name__ == '__main__':
  app.run(main)
