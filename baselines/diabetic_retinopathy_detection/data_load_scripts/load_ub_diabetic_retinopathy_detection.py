"""
Loads and packages the data for the ub_diabetic_retinopathy_detection dataset.
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

  # Load `ub_diabetic_retinopathy_detection` train data
  dataset_train_builder = ub.datasets.get(
    "ub_diabetic_retinopathy_detection",
    split='train',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `ub_diabetic_retinopathy_detection` train data.')
  dataset_train_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/downloads/')
  logging.info(
    'Finished packaging `ub_diabetic_retinopathy_detection` train data.')

  # Load `ub_diabetic_retinopathy_detection` test data
  dataset_test_builder = ub.datasets.get(
    "ub_diabetic_retinopathy_detection",
    split='test',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `ub_diabetic_retinopathy_detection` test data.')
  dataset_test_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/downloads/')
  logging.info(
    'Finished packaging `ub_diabetic_retinopathy_detection` test data.')

  logging.info('Finished packaging `ub_diabetic_retinopathy_detection` data.')

if __name__ == '__main__':
  app.run(main)
