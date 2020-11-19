# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

# Lint as: python3
"""Dataset getter utility."""

import json
import logging
from typing import Any, Dict, List, Optional
import warnings

from uncertainty_baselines.datasets.base import BaseDataset
from uncertainty_baselines.datasets.cifar import Cifar100Dataset
from uncertainty_baselines.datasets.cifar import Cifar10Dataset
from uncertainty_baselines.datasets.clinc_intent import ClincIntentDetectionDataset
from uncertainty_baselines.datasets.criteo import CriteoDataset
from uncertainty_baselines.datasets.diabetic_retinopathy_detection import DiabeticRetinopathyDetectionDataset
from uncertainty_baselines.datasets.genomics_ood import GenomicsOodDataset
from uncertainty_baselines.datasets.glue import GlueDatasets
from uncertainty_baselines.datasets.imagenet import ImageNetDataset
from uncertainty_baselines.datasets.mnist import MnistDataset
from uncertainty_baselines.datasets.mnli import MnliDataset
from uncertainty_baselines.datasets.movielens import MovieLensDataset
from uncertainty_baselines.datasets.places import Places365Dataset
from uncertainty_baselines.datasets.random import RandomGaussianImageDataset
from uncertainty_baselines.datasets.random import RandomRademacherImageDataset
from uncertainty_baselines.datasets.svhn import SvhnDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsIdentitiesDataset
from uncertainty_baselines.datasets.toxic_comments import WikipediaToxicityDataset


try:
  from uncertainty_baselines.datasets.speech_commands import SpeechCommandsDataset  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')
  SpeechCommandsDataset = None

  
_DATASETS = {
    'cifar100': Cifar100Dataset,
    'cifar10': Cifar10Dataset,
    'civil_comments': CivilCommentsDataset,
    'civil_comments_identities': CivilCommentsIdentitiesDataset,
    'clinic_intent': ClincIntentDetectionDataset,
    'criteo': CriteoDataset,
    'diabetic_retinopathy_detection': DiabeticRetinopathyDetectionDataset,
    'imagenet': ImageNetDataset,
    'mnist': MnistDataset,
    'mnli': MnliDataset,
    'movielens': MovieLensDataset,
    'places365': Places365Dataset,
    'random_gaussian': RandomGaussianImageDataset,
    'random_rademacher': RandomRademacherImageDataset,
    'speech_commands': SpeechCommandsDataset,
    'svhn': SvhnDataset,
    'glue/cola': GlueDatasets['glue/cola'],
    'glue/sst2': GlueDatasets['glue/sst2'],
    'glue/mrpc': GlueDatasets['glue/mrpc'],
    'glue/qqp': GlueDatasets['glue/qqp'],
    'glue/qnli': GlueDatasets['glue/qnli'],
    'glue/rte': GlueDatasets['glue/rte'],
    'glue/wnli': GlueDatasets['glue/wnli'],
    'glue/stsb': GlueDatasets['glue/stsb'],
    'wikipedia_toxicity': WikipediaToxicityDataset,
    'genomics_ood': GenomicsOodDataset,
}


def get_dataset_names() -> List[str]:
  return list(_DATASETS.keys())


def get(dataset_name: str,
    batch_size: int,
    eval_batch_size: int,
    data_dir: Optional[str] = None,
    **hyperparameters: Dict[str, Any]) -> BaseDataset:
  """Gets a dataset builder by name.

  Args:
    dataset_name: Name of the dataset builder class.
    batch_size: the training batch size.
    eval_batch_size: the validation/test batch size.
    data_dir: optional dir to save TFDS data to. If none then the local
      filesystem is used. Required for using TPUs on Cloud.
    **hyperparameters: dict of possible kwargs to be passed to the dataset
      constructor.

  Returns:
    A dataset builder class with a method .build(split) which can be called to
    get the tf.data.Dataset, which has elements that are a dict with keys
    'features' and 'labels'.

  Raises:
    ValueError: If dataset_name is unrecognized.
  """
  logging.info(
      'Building dataset %s with additional kwargs:\n%s',
      dataset_name,
      json.dumps(hyperparameters, indent=2, sort_keys=True))
  if dataset_name not in _DATASETS:
    raise ValueError('Unrecognized dataset name: {!r}'.format(dataset_name))

  dataset_class = _DATASETS[dataset_name]
  return dataset_class(batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      data_dir=data_dir,
      **hyperparameters)
