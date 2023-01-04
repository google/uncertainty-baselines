"""
Loads and packages the data for the
diabetic_retinopathy_severity_shift_mild dataset.
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

  # The DiabeticRetinopathySeverityShiftMild dataset has
  #   five splits:
  #     - train
  #     - in_domain_validation
  #     - ood_validation
  #     - in_domain_test
  #     - ood_test

  # Load `diabetic_retinopathy_severity_shift_mild` train data
  dataset_train_builder = ub.datasets.get(
    "diabetic_retinopathy_severity_shift_mild",
    split='train',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `diabetic_retinopathy_severity_shift_mild` '
    'train data.')
  dataset_train_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/diabetic_retinopathy_severity_shift_mild/')
  logging.info(
    'Finished packaging `diabetic_retinopathy_severity_shift_mild` '
    'train data.')

  # Load `diabetic_retinopathy_severity_shift_mild` in_domain_validation data
  dataset_in_domain_validation_builder = ub.datasets.get(
    "diabetic_retinopathy_severity_shift_mild",
    split='in_domain_validation',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `diabetic_retinopathy_severity_shift_mild` in_domain_validation data.')
  dataset_in_domain_validation_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/diabetic_retinopathy_severity_shift_mild/')
  logging.info(
    'Finished packaging `diabetic_retinopathy_severity_shift_mild` in_domain_validation data.')

  # Load `diabetic_retinopathy_severity_shift_mild` ood_validation data
  dataset_ood_validation_builder = ub.datasets.get(
    "diabetic_retinopathy_severity_shift_mild",
    split='ood_validation',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `diabetic_retinopathy_severity_shift_mild` ood_validation data.')
  dataset_ood_validation_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/diabetic_retinopathy_severity_shift_mild/')
  logging.info(
    'Finished packaging `diabetic_retinopathy_severity_shift_mild` ood_validation data.')

  # Load `diabetic_retinopathy_severity_shift_mild` in_domain_test data
  dataset_in_domain_test_builder = ub.datasets.get(
    "diabetic_retinopathy_severity_shift_mild",
    split='in_domain_test',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `diabetic_retinopathy_severity_shift_mild` in_domain_test data.')
  dataset_in_domain_test_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/diabetic_retinopathy_severity_shift_mild/')
  logging.info(
    'Finished packaging `diabetic_retinopathy_severity_shift_mild` in_domain_test data.')

  # Load `diabetic_retinopathy_severity_shift_mild` ood_test data
  dataset_ood_test_builder = ub.datasets.get(
    "diabetic_retinopathy_severity_shift_mild",
    split='ood_test',
    data_dir=data_dir,
    download_data=True)
  logging.info(
    'Shuffling and packaging `diabetic_retinopathy_severity_shift_mild` ood_test data.')
  dataset_ood_test_builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/diabetic_retinopathy_severity_shift_mild/')
  logging.info(
    'Finished packaging `diabetic_retinopathy_severity_shift_mild` ood_test data.')

  logging.info('Finished packaging `diabetic_retinopathy_severity_shift_mild` data.')

if __name__ == '__main__':
  app.run(main)
