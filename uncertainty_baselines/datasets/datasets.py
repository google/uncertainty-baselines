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
from typing import Any, Dict, List

from uncertainty_baselines.datasets.base import BaseDataset
from uncertainty_baselines.datasets.cifar import Cifar100Dataset
from uncertainty_baselines.datasets.cifar import Cifar10Dataset
from uncertainty_baselines.datasets.clinc_intent import ClincIntentDetectionDataset
from uncertainty_baselines.datasets.criteo import CriteoDataset
from uncertainty_baselines.datasets.genomics_ood import GenomicsOodDataset
from uncertainty_baselines.datasets.glue import GlueDatasets
from uncertainty_baselines.datasets.imagenet import ImageNetDataset
from uncertainty_baselines.datasets.mnist import MnistDataset
from uncertainty_baselines.datasets.mnli import MnliDataset
from uncertainty_baselines.datasets.random import RandomGaussianImageDataset
from uncertainty_baselines.datasets.random import RandomRademacherImageDataset
from uncertainty_baselines.datasets.svhn import SvhnDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsIdentitiesDataset
from uncertainty_baselines.datasets.toxic_comments import WikipediaToxicityDataset


def get_dataset_names() -> List[str]:
  return [
      'cifar100',
      'cifar10',
      'civil_comments',
      'civil_comments_identities',
      'clinic_intent',
      'criteo',
      'imagenet',
      'mnist',
      'mnli',
      'random_gaussian',
      'random_rademacher',
      'svhn',
      'glue/cola',
      'glue/sst2',
      'glue/mrpc',
      'glue/qqp',
      'glue/qnli',
      'glue/rte',
      'glue/wnli',
      'glue/stsb',
      'wikipedia_toxicity',
      'genomics_ood',
  ]


def get(
    dataset_name: str,
    batch_size: int,
    eval_batch_size: int,
    **hyperparameters: Dict[str, Any]) -> BaseDataset:
  """Gets a dataset builder by name.

  Args:
    dataset_name: Name of the dataset builder class.
    batch_size: the training batch size.
    eval_batch_size: the validation/test batch size.
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
  elif dataset_name == 'cifar100':
    dataset_class = Cifar100Dataset
  elif dataset_name == 'cifar10':
    dataset_class = Cifar10Dataset
  elif dataset_name == 'civil_comments':
    dataset_class = CivilCommentsDataset
  elif dataset_name == 'civil_comments_identities':
    dataset_class = CivilCommentsIdentitiesDataset
  elif dataset_name == 'clinic_intent':
    dataset_class = ClincIntentDetectionDataset
  elif dataset_name == 'criteo':
    dataset_class = CriteoDataset
  elif dataset_name == 'imagenet':
    dataset_class = ImageNetDataset
  elif dataset_name == 'mnist':
    dataset_class = MnistDataset
  elif dataset_name == 'mnli':
    dataset_class = MnliDataset
  elif dataset_name == 'random_gaussian':
    dataset_class = RandomGaussianImageDataset
  elif dataset_name == 'random_rademacher':
    dataset_class = RandomRademacherImageDataset
  elif dataset_name == 'svhn':
    dataset_class = SvhnDataset
  elif 'glue/' in dataset_name:
    dataset_class = GlueDatasets[dataset_name]
  elif dataset_name == 'wikipedia_toxicity':
    dataset_class = WikipediaToxicityDataset
  elif dataset_name == 'genomics_ood':
    dataset_class = GenomicsOodDataset
  else:
    raise ValueError('Unrecognized dataset name: {!r}'.format(dataset_name))

  return dataset_class(
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      **hyperparameters)
