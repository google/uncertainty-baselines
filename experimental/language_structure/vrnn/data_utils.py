# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Data utils to ease the dataset loading."""

import os

SUPPORTED_DATASETS = [
    'simdial',
    'simdial-weather',
    'simdial-bus',
    'simdial-movie',
    'simdial-restaurant',
    'multiwoz_synth',
    'sgd_synth',
    'sgd',
    'sgd_domain_adapation',
]

_DATASET_MAX_SEQ_LENGTH = {
    'simdial': 40,
    'simdial-weather': 40,
    'simdial-bus': 40,
    'simdial-movie': 40,
    'simdial-restaurant': 40,
    'multiwoz_synth': 42,
    'sgd_synth': 76,
    'sgd': 79,
    'sgd_domain_adapation': 79,
}

_DATASET_MAX_DIALOG_LENGTH = {
    'simdial': 13,
    'simdial-weather': 13,
    'simdial-bus': 13,
    'simdial-movie': 13,
    'simdial-restaurant': 13,
    'multiwoz_synth': 7,
    'sgd_synth': 24,
    'sgd': 25,
    'sgd_domain_adapation': 25,
}

_DATASET_NUM_LATENT_STATES = {
    'simdial': 52,
    'simdial-weather': 12,
    'simdial-bus': 14,
    'simdial-movie': 22,
    'simdial-restaurant': 20,
    'multiwoz_synth': 10,
    'sgd_synth': 39,
    'sgd': 75,
    'sgd_domain_adapation': 75,
}

_DATASET_NUM_DOMIANS = {
    'simdial': 5,
    'simdial-weather': 2,
    'simdial-bus': 2,
    'simdial-movie': 2,
    'simdial-restaurant': 2,
    'multiwoz_synth': 7,
    'sgd_synth': 33,
    'sgd': 18,
    'sgd_domain_adapation': 18,
}

_DATASET_BASEDIR_PATH = {dataset: '' for dataset in SUPPORTED_DATASETS}



def _check_dataset_supported(dataset: str):
  if dataset not in SUPPORTED_DATASETS:
    raise ValueError('dataset must be one of {}. Found {}.'.format(
        ','.join(SUPPORTED_DATASETS), dataset))




def get_dataset_max_seq_length(dataset: str) -> int:
  _check_dataset_supported(dataset)
  return _DATASET_MAX_SEQ_LENGTH[dataset]


def get_dataset_max_dialog_length(dataset: str) -> int:
  _check_dataset_supported(dataset)
  return _DATASET_MAX_DIALOG_LENGTH[dataset]


def get_dataset_num_latent_states(dataset: str) -> int:
  _check_dataset_supported(dataset)
  return _DATASET_NUM_LATENT_STATES[dataset]


def get_dataset_num_domains(dataset: str) -> int:
  _check_dataset_supported(dataset)
  return _DATASET_NUM_DOMIANS[dataset]
