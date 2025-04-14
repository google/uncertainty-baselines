"""
Loads and packages the data for the RETINA Benchmark.
"""

from absl import app
from absl import flags
from absl import logging

import uncertainty_baselines as ub

flags.DEFINE_string(
  'data_dir', None, 
  'Path to data folder, which contain subfolders called '
  '`ub_diabetic_retinopathy_detection` and `aptos`, containing the raw data for '
  'EyePACS and APTOS 2019 respectively. See README.md for further information.')
flags.DEFINE_string(
  'prepare_mode', 'all', 'Determine which dataset(s) to prepare.')
flags.register_validator(
  'prepare_mode',
  lambda value: value in ['all', 'eyepacs', 'aptos', 'severity'],
  message='--prepare_mode must be one of [all, eyepacs, aptos, severity].')
FLAGS = flags.FLAGS

# Supported datasets.
_UB_DIABETIC_RETINOPATHY_DETECTION ="ub_diabetic_retinopathy_detection"
_APTOS = "aptos"
_DIABETIC_RETINOPATHY_SEVERITY_SHIFT_MILD = (
  "diabetic_retinopathy_severity_shift_mild")

# Splits for each dataset.
_SPLITS = {
  _UB_DIABETIC_RETINOPATHY_DETECTION: ['train', 'test'],
  _APTOS: ['validation', 'test'],
  _DIABETIC_RETINOPATHY_SEVERITY_SHIFT_MILD: [
    'train', 'in_domain_validation',
    'ood_validation', 'in_domain_test', 'ood_test']
}

_DATASET_NAMES_BY_MODE = {
  'all': list(_SPLITS.keys()),
  'eyepacs': [_UB_DIABETIC_RETINOPATHY_DETECTION],
  'aptos': [_APTOS],
  'severity': [_DIABETIC_RETINOPATHY_SEVERITY_SHIFT_MILD]
}

def _download_and_prepare_dataset(
    dataset_name: str,
    split: str,
    data_dir: str
) -> None:
  builder = ub.datasets.get(dataset_name=dataset_name, split=split,
                            data_dir=data_dir, download_data=True)
  builder._dataset_builder.download_and_prepare(
    download_dir=f'{data_dir}/{dataset_name}/')

def main(argv):
  del argv  # unused arg
  data_dir = FLAGS.data_dir
  dataset_names = _DATASET_NAMES_BY_MODE[FLAGS.prepare_mode]
  for dataset_name in dataset_names:
    for split in _SPLITS[dataset_name]:
      _download_and_prepare_dataset(
        dataset_name=dataset_name, split=split, data_dir=data_dir)
      logging.info(f'Finished packaging `{dataset_name}` {split} data.')

if __name__ == '__main__':
  app.run(main)
