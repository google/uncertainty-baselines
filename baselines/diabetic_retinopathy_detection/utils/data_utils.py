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
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import logging

import tensorflow_datasets as tfds

import uncertainty_baselines as ub


def load_kaggle_severity_shift_dataset(train_batch_size,
                                       eval_batch_size,
                                       flags,
                                       strategy,
                                       load_for_eval=False):
  """Partitioning of the Kaggle/EyePACS Diabetic Retinopathy dataset to hold out certain clinical severity levels as OOD.

  Optionally exclude train split (e.g., loading for evaluation) in flags.
  See runscripts for more information on loading options.

  Args:
    train_batch_size: int.
    eval_batch_size: int.
    flags: FlagValues, runscript flags.
    strategy: tf.distribute strategy, used to distribute datasets.
    load_for_eval: Bool, if True, does not truncate the last batch (for
      standardized evaluation).

  Returns:
    Dict of datasets, Dict of number of steps per dataset.
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
  #
  # We have split sizes:
  if flags.dr_decision_threshold == 'mild':
    split_to_num_examples = {
        'in_domain_validation': 8850,
        'ood_validation': 2056,
        'in_domain_test': 34445,
        'ood_test': 15098
    }
    if train_batch_size is not None:
      split_to_num_examples['train'] = 28253
  elif flags.dr_decision_threshold == 'moderate':
    split_to_num_examples = {
        'in_domain_validation': 10429,
        'ood_validation': 477,
        'in_domain_test': 40727,
        'ood_test': 3524
    }
    if train_batch_size is not None:
      split_to_num_examples['train'] = 33545
  else:
    raise NotImplementedError(
        f'Unknown decision threshold {flags.dr_decision_threshold}.')

  split_to_batch_size = {
      'in_domain_validation': eval_batch_size,
      'ood_validation': eval_batch_size,
      'in_domain_test': eval_batch_size,
      'ood_test': eval_batch_size
  }

  if train_batch_size is not None:
    split_to_batch_size['train'] = train_batch_size

  split_to_steps_per_epoch = {
      split: num_examples // split_to_batch_size[split]
      for split, num_examples in split_to_num_examples.items()
  }
  splits_to_return = ['in_domain_validation', 'ood_validation']
  if load_train_split:
    splits_to_return = ['train'] + splits_to_return
  if flags.use_test:
    splits_to_return = splits_to_return + ['in_domain_test', 'ood_test']

  dataset_name = (
      f'diabetic_retinopathy_severity_shift_{flags.dr_decision_threshold}')
  split_to_dataset = {}
  for split in splits_to_return:
    dataset_builder = ub.datasets.get(
        dataset_name,
        split=split,
        data_dir=data_dir,
        cache=(flags.cache_eval_datasets and split != 'train'),
        drop_remainder=not load_for_eval,
        builder_config=f'{dataset_name}/{flags.preproc_builder_config}')
    dataset = dataset_builder.load(batch_size=split_to_batch_size[split])

    if strategy is not None:
      dataset = strategy.experimental_distribute_dataset(dataset)

    split_to_dataset[split] = dataset

  return split_to_dataset, split_to_steps_per_epoch


def load_kaggle_aptos_country_shift_dataset(train_batch_size,
                                            eval_batch_size,
                                            flags,
                                            strategy,
                                            load_for_eval=False):
  """Full Kaggle/EyePACS Diabetic Retinopathy dataset, including OOD validation/test sets (APTOS).

  Optionally exclude train split (e.g., loading for evaluation) in flags.
  See runscripts for more information on loading options.

  Args:
    train_batch_size: int.
    eval_batch_size: int.
    flags: FlagValues, runscript flags.
    strategy: tf.distribute strategy, used to distribute datasets.
    load_for_eval: Bool, if True, does not truncate the last batch.

  Returns:
    Dict of datasets, Dict of number of steps per dataset.
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

  dr_dataset_name = 'ub_diabetic_retinopathy_detection'

  # Load validation data
  dataset_validation_builder = ub.datasets.get(
      dr_dataset_name,
      split='validation',
      data_dir=data_dir,
      is_training=not flags.use_validation,
      decision_threshold=flags.dr_decision_threshold,
      cache=flags.cache_eval_datasets,
      drop_remainder=not load_for_eval,
      builder_config=f'{dr_dataset_name}/{flags.preproc_builder_config}')
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
        'aptos',
        split='validation',
        data_dir=data_dir,
        decision_threshold=flags.dr_decision_threshold,
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'aptos/{flags.preproc_builder_config}')
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
        dr_dataset_name,
        split='train',
        data_dir=data_dir,
        decision_threshold=flags.dr_decision_threshold,
        builder_config=f'{dr_dataset_name}/{flags.preproc_builder_config}')
    dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

    if not flags.use_validation:
      # TODO(nband): investigate validation dataset concat bug
      # Note that this will not create any mixed batches of
      # train and validation images.
      # dataset_train = dataset_train.concatenate(dataset_validation)
      raise NotImplementedError(
          'Existing bug involving the number of steps not being adjusted after '
          'concatenating the validation dataset. Needs verifying.')

    if strategy is not None:
      dataset_train = strategy.experimental_distribute_dataset(dataset_train)

    split_to_dataset['train'] = dataset_train

  if flags.use_test:
    # In-Domain Test
    dataset_test_builder = ub.datasets.get(
        dr_dataset_name,
        split='test',
        data_dir=data_dir,
        decision_threshold=flags.dr_decision_threshold,
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'{dr_dataset_name}/{flags.preproc_builder_config}')
    dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
    if strategy is not None:
      dataset_test = strategy.experimental_distribute_dataset(dataset_test)

    split_to_dataset['in_domain_test'] = dataset_test

    # OOD (APTOS) Test
    aptos_test_builder = ub.datasets.get(
        'aptos',
        split='test',
        data_dir=data_dir,
        decision_threshold=flags.dr_decision_threshold,
        cache=flags.cache_eval_datasets,
        drop_remainder=not load_for_eval,
        builder_config=f'aptos/{flags.preproc_builder_config}')
    dataset_ood_test = aptos_test_builder.load(batch_size=eval_batch_size)
    if strategy is not None:
      dataset_ood_test = strategy.experimental_distribute_dataset(
          dataset_ood_test)

    split_to_dataset['ood_test'] = dataset_ood_test

  return split_to_dataset, split_to_steps_per_epoch


def load_dataset(train_batch_size,
                 eval_batch_size,
                 flags,
                 strategy,
                 load_for_eval=False):
  """Retrieve the in-domain and OOD datasets for a given distributional shift task in diabetic retinopathy.

  Optionally exclude train split (e.g., loading for evaluation) in flags.
  See runscripts for more information on loading options.

  Args:
    train_batch_size: int.
    eval_batch_size: int.
    flags: FlagValues, runscript flags.
    strategy: tf.distribute strategy, used to distribute datasets.
    load_for_eval: Bool, if True, does not truncate the last batch.

  Returns:
    Dict of datasets, Dict of number of steps per dataset.
  """
  distribution_shift = flags.distribution_shift

  if distribution_shift == 'severity':
    datasets, steps = load_kaggle_severity_shift_dataset(
        train_batch_size,
        eval_batch_size,
        flags=flags,
        strategy=strategy,
        load_for_eval=load_for_eval)
  elif distribution_shift == 'aptos' or distribution_shift is None:
    datasets, steps = load_kaggle_aptos_country_shift_dataset(
        train_batch_size,
        eval_batch_size,
        flags=flags,
        strategy=strategy,
        load_for_eval=load_for_eval)
  else:
    raise NotImplementedError(
        'Only support `severity` and `aptos` dataset partitions '
        '(None defaults to APTOS).')

  logging.info(f'Datasets using builder config {flags.preproc_builder_config}.')
  logging.info(f'Successfully loaded the following dataset splits from the '
               f'{distribution_shift} shift dataset: {list(datasets.keys())}')
  return datasets, steps
