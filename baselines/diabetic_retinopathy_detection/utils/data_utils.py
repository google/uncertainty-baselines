# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Diabetic Retinopathy Data Loading utils."""

import logging

import tensorflow_datasets as tfds

import uncertainty_baselines as ub


def load_kaggle_severity_shift_dataset(
    train_batch_size, eval_batch_size, flags, strategy
):
  """
  Partitioning of the Kaggle/EyePACS Diabetic Retinopathy dataset to
  hold out certain clinical severity levels as OOD.

  :param flags:
  :param strategy:
  :param load_train_split:
  :return:
  """
  assert flags.use_validation
  data_dir = flags.data_dir
  load_train_split = flags.load_train_split

  # Using the decision threshold between severity levels 0 and 1, we define
  # the in-domain (training) task as
  # Binary classification between examples with severity levels 0 and 1
  # We consider examples with levels 2,3,4 as OOD
  # This leaves a few thousand examples previously segmented in the Kaggle
  # training set (the 2,3,4 ones) which we now group into the OOD test set.

  # We have split sizes:
  split_to_num_examples = {
    'train': 28253,
    'in_domain_validation': 8850,
    'ood_validation': 2056,
    'in_domain_test': 34445,
    'ood_test': 15098
  }
  split_to_batch_size = {
    'train': train_batch_size,
    'in_domain_validation': eval_batch_size,
    'ood_validation': eval_batch_size,
    'in_domain_test': eval_batch_size,
    'ood_test': eval_batch_size
  }

  split_to_steps_per_epoch = {
    split: num_examples // split_to_batch_size[split]
    for split, num_examples in split_to_num_examples.items()
  }
  splits_to_return = [
    'in_domain_validation', 'ood_validation']
  if load_train_split:
    splits_to_return = ['train'] + splits_to_return
  if flags.use_test:
    splits_to_return = splits_to_return + ['in_domain_test', 'ood_test']

  split_to_dataset = {}
  for split in splits_to_return:
    dataset_builder = ub.datasets.get(
      'diabetic_retinopathy_severity_shift', split=split, data_dir=data_dir)
    dataset = dataset_builder.load(batch_size=split_to_batch_size[split])

    if strategy is not None:
      dataset = strategy.experimental_distribute_dataset(dataset)

    split_to_dataset[split] = dataset

  return split_to_dataset, split_to_steps_per_epoch


def load_kaggle_aptos_country_shift_dataset(
    train_batch_size, eval_batch_size, flags, strategy
):
  """
  Full Kaggle/EyePACS Diabetic Retinopathy dataset, including OOD
  validation/test sets (APTOS).

  Optionally exclude train split (e.g., loading for evaluation).

  :param flags:
  :param strategy:
  :return:
  """
  data_dir = flags.data_dir
  load_train_split = flags.load_train_split

  # * Load Steps Per Epoch for Each Dataset *
  split_to_steps_per_epoch = {}

  # As per the Kaggle challenge, we have split sizes for the EyePACS subsets:
  # train: 35,126
  # validation: 10,906
  # test: 42,670
  ds_info = tfds.builder('diabetic_retinopathy_detection').info
  if load_train_split:
    split_to_steps_per_epoch['train'] = (
      ds_info.splits['train'].num_examples // train_batch_size)
  split_to_steps_per_epoch['in_domain_validation'] = (
    ds_info.splits['validation'].num_examples // eval_batch_size)
  split_to_steps_per_epoch['in_domain_test'] = (
    ds_info.splits['test'].num_examples // eval_batch_size)

  # APTOS Evaluation Data
  split_to_steps_per_epoch['ood_validation'] = 733 // eval_batch_size
  split_to_steps_per_epoch['ood_test'] = 2929 // eval_batch_size

  # * Load Datasets *
  split_to_dataset = {}

  # Load validation data
  dataset_validation_builder = ub.datasets.get(
    'diabetic_retinopathy_detection', split='validation', data_dir=data_dir,
    is_training=not flags.use_validation,
    decision_threshold=flags.dr_decision_threshold)
  validation_batch_size = (
    eval_batch_size if flags.use_validation else train_batch_size)
  dataset_validation = dataset_validation_builder.load(
    batch_size=validation_batch_size)

  # If `flags.use_validation`, then we distribute the validation dataset
  # independently and add as a separate dataset.
  # Otherwise, we concatenate it with the training data below.
  if flags.use_validation:
    # Load APTOS validation dataset
    aptos_validation_builder = ub.datasets.get(
      'aptos', split='validation', data_dir=data_dir,
      decision_threshold=flags.dr_decision_threshold)
    dataset_ood_validation = aptos_validation_builder.load(
      batch_size=eval_batch_size)

    if strategy is not None:
      dataset_validation = strategy.experimental_distribute_dataset(
        dataset_validation)
      dataset_ood_validation = strategy.experimental_distribute_dataset(
        dataset_ood_validation)

    split_to_dataset['in_domain_validation'] = dataset_validation
    split_to_dataset['ood_validation'] = dataset_ood_validation

  if load_train_split:
    # Load EyePACS train data
    dataset_train_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='train', data_dir=data_dir,
      decision_threshold=flags.dr_decision_threshold)
    dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

    if not flags.use_validation:
      raise NotImplementedError(
        'Existing bug involving the number of steps not being adjusted after '
        'concatenating the validation dataset. Needs verifying.')
      # Note that this will not create any mixed batches of
      # train and validation images.
      dataset_train = dataset_train.concatenate(dataset_validation)

    if strategy is not None:
      dataset_train = strategy.experimental_distribute_dataset(dataset_train)

    split_to_dataset['train'] = dataset_train

  if flags.use_test:
    print(flags.use_test)
    print('Flags -- using test datasets.')
    # In-Domain Test
    dataset_test_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='test', data_dir=data_dir,
      decision_threshold=flags.dr_decision_threshold)
    dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
    if strategy is not None:
      dataset_test = strategy.experimental_distribute_dataset(dataset_test)

    split_to_dataset['in_domain_test'] = dataset_test

    # OOD (APTOS) Test
    aptos_test_builder = ub.datasets.get(
      'aptos', split='test', data_dir=data_dir,
      decision_threshold=flags.dr_decision_threshold)
    dataset_ood_test = aptos_test_builder.load(batch_size=eval_batch_size)
    if strategy is not None:
      dataset_ood_test = strategy.experimental_distribute_dataset(
        dataset_ood_test)

    split_to_dataset['ood_test'] = dataset_ood_test

  return split_to_dataset, split_to_steps_per_epoch


def load_dataset(train_batch_size, eval_batch_size, flags, strategy):
  distribution_shift = flags.distribution_shift

  if distribution_shift == 'severity':
    datasets, steps = load_kaggle_severity_shift_dataset(
      train_batch_size, eval_batch_size,
      flags=flags, strategy=strategy)
  elif distribution_shift == 'aptos' or distribution_shift is None:
    datasets, steps = load_kaggle_aptos_country_shift_dataset(
      train_batch_size, eval_batch_size,
      flags=flags, strategy=strategy)
  else:
    raise NotImplementedError(
      'Only support `severity` and `aptos` dataset partitions '
      '(None defaults to APTOS).')

  logging.info(f'Successfully loaded the following dataset splits from the '
               f'{distribution_shift} shift dataset: {list(datasets.keys())}')
  return datasets, steps
