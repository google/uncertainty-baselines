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

"""Utility function for BERT models."""
import json
import os
import re

from typing import Any, Dict, List, Mapping, Optional, Tuple
from absl import flags
from absl import logging

import numpy as np
import robustness_metrics as rm
import tensorflow as tf

from tensorflow_addons import metrics as tfa_metrics
import metrics  # local file import from baselines.toxic_comments as tc_metrics
from uncertainty_baselines.datasets import toxic_comments as ds

from tensorflow.core.protobuf import trackable_object_graph_pb2  # pylint: disable=g-direct-tensorflow-import
from official.nlp import optimization
from official.nlp.bert import configs

IND_DATA_CLS = ds.WikipediaToxicityDataset
OOD_DATA_CLS = ds.CivilCommentsDataset


# Number of positive examples for each datasets.
# For `civil_comments`, the positive examples are based on threshold = 0.7.
NUM_POS_EXAMPLES = {
    'wikipedia_toxicity_subtypes': {'train': 15294, 'test': 6090},
    'civil_comments': {'train': 45451, 'test': 2458},
    'civil_comments/CivilCommentsIdentities': {'train': 12246, 'test': 634},
    'ind': {'train': 15294, 'test': 6090},
    'ood': {'train': 45451, 'test': 2458},
    'ood_identity': {'train': 12246, 'test': 634},
    'gender': {'test': 126, 'validation': 131, 'train': 2607},
    'sexual_orientation': {'test': 19, 'validation': 26, 'train': 596},
    'religion': {'test': 85, 'validation': 75, 'train': 1557},
    'race': {'test': 101, 'validation': 112, 'train': 2070},
    'disability': {'test': 10, 'validation': 12, 'train': 243},
    'male': {'test': 62, 'validation': 80, 'train': 1396},
    'female': {'test': 70, 'validation': 66, 'train': 1531},
    'transgender': {'test': 5, 'validation': 8, 'train': 91},
    'other_gender': {'test': 0, 'validation': 0, 'train': 0},
    'heterosexual': {'test': 3, 'validation': 1, 'train': 27},
    'homosexual_gay_or_lesbian': {'test': 17, 'validation': 25, 'train': 583},
    'bisexual': {'test': 0, 'validation': 0, 'train': 6},
    'other_sexual_orientation': {'test': 0, 'validation': 0, 'train': 1},
    'christian': {'test': 38, 'validation': 23, 'train': 536},
    'jewish': {'test': 11, 'validation': 16, 'train': 252},
    'muslim': {'test': 43, 'validation': 44, 'train': 959},
    'hindu': {'test': 0, 'validation': 1, 'train': 10},
    'buddhist': {'test': 0, 'validation': 1, 'train': 10},
    'atheist': {'test': 3, 'validation': 2, 'train': 34},
    'other_religion': {'test': 0, 'validation': 0, 'train': 5},
    'black': {'test': 41, 'validation': 46, 'train': 977},
    'white': {'test': 72, 'validation': 77, 'train': 1358},
    'asian': {'test': 4, 'validation': 3, 'train': 93},
    'latino': {'test': 3, 'validation': 1, 'train': 50},
    'other_race_or_ethnicity': {'test': 0, 'validation': 0, 'train': 8},
    'physical_disability': {'test': 0, 'validation': 0, 'train': 0},
    'intellectual_or_learning_disability': {
        'test': 0, 'validation': 0, 'train': 0},
    'psychiatric_or_mental_illness': {
        'test': 10, 'validation': 12, 'train': 243},
    'other_disability': {'test': 0, 'validation': 0, 'train': 0}}

# Number of examples for each identity dataset splits.
# The examples are selected based on threshold=0.5.
NUM_EXAMPLES = {
    'gender': {'test': 3694, 'validation': 3704, 'train': 75662},
    'sexual_orientation': {'test': 518, 'validation': 517, 'train': 10777},
    'religion': {'test': 3027, 'validation': 2948, 'train': 57772},
    'race': {'test': 1771, 'validation': 1803, 'train': 36230},
    'disability': {'test': 200, 'validation': 217, 'train': 4129},
    'male': {'test': 1907, 'validation': 2030, 'train': 40024},
    'female': {'test': 2433, 'validation': 2351, 'train': 50536},
    'transgender': {'test': 117, 'validation': 112, 'train': 2208},
    'other_gender': {'test': 0, 'validation': 0, 'train': 3},
    'heterosexual': {'test': 56, 'validation': 62, 'train': 1030},
    'homosexual_gay_or_lesbian': {
        'test': 486, 'validation': 484, 'train': 10230},
    'bisexual': {'test': 16, 'validation': 8, 'train': 171},
    'other_sexual_orientation': {'test': 0, 'validation': 0, 'train': 3},
    'christian': {'test': 1869, 'validation': 1808, 'train': 35491},
    'jewish': {'test': 383, 'validation': 397, 'train': 7237},
    'muslim': {'test': 975, 'validation': 901, 'train': 19659},
    'hindu': {'test': 25, 'validation': 19, 'train': 467},
    'buddhist': {'test': 24, 'validation': 22, 'train': 471},
    'atheist': {'test': 138, 'validation': 120, 'train': 1269},
    'other_religion': {'test': 3, 'validation': 1, 'train': 79},
    'black': {'test': 693, 'validation': 689, 'train': 13864},
    'white': {'test': 1128, 'validation': 1191, 'train': 23843},
    'asian': {'test': 165, 'validation': 177, 'train': 3550},
    'latino': {'test': 86, 'validation': 73, 'train': 1425},
    'other_race_or_ethnicity': {'test': 5, 'validation': 6, 'train': 148},
    'physical_disability': {'test': 1, 'validation': 1, 'train': 30},
    'intellectual_or_learning_disability': {
        'test': 3, 'validation': 5, 'train': 42},
    'psychiatric_or_mental_illness': {
        'test': 197, 'validation': 212, 'train': 4075},
    'other_disability': {'test': 0, 'validation': 0, 'train': 1}}

IDENTITY_LABELS = ('male', 'female', 'transgender', 'other_gender',
                   'heterosexual', 'homosexual_gay_or_lesbian', 'bisexual',
                   'other_sexual_orientation', 'christian', 'jewish', 'muslim',
                   'hindu', 'buddhist', 'atheist', 'other_religion', 'black',
                   'white', 'asian', 'latino', 'other_race_or_ethnicity',
                   'physical_disability', 'intellectual_or_learning_disability',
                   'psychiatric_or_mental_illness', 'other_disability')

IDENTITY_TYPES = ('gender', 'sexual_orientation', 'religion', 'race',
                  'disability')

CHALLENGE_DATASET_NAMES = ('bias', 'uncertainty', 'noise', 'all')

# Data flags
flags.DEFINE_enum(
    'dataset_type', 'tfrecord', ['tfrecord', 'csv', 'tfds'],
    'Must be one of ["tfrecord", "csv", "tfds"]. If "tfds", data will be loaded'
    ' from TFDS. Otherwise it will be loaded from local directory.')
flags.DEFINE_bool(
    'use_cross_validation', False,
    'Whether to use cross validation for training and evaluation protocol.'
    ' If True, then divide the official train split into further train and eval'
    ' sets, and evaluate on both the eval set and the official test splits.')
flags.DEFINE_integer(
    'num_folds', 10,
    'The number of folds to be used for cross-validation training. Ignored if'
    ' use_cross_validation is False.')
flags.DEFINE_list(
    'train_fold_ids', ['1', '2', '3', '4', '5'],
    'The ids of folds to use for training, the rest of the folds will be used'
    ' for cross-validation eval. Ignored if use_cross_validation is False.')
flags.DEFINE_string(
    'train_cv_split_name', 'train',
    'The name of the split to create cross-validation training data from.')
flags.DEFINE_string(
    'test_cv_split_name', 'train',
    'The name of the split to create cross-validation testing data from.')

flags.DEFINE_enum(
    'train_on_multi_task_label', 'bias',
    ['', 'bias', 'uncertainty', 'noise'],
    'The type of additional multi-task labels (a binary label for whether the'
    ' model is likely to make a mistake on this example) to the training '
    'output. If empty then do not perform multi-task training.')
flags.DEFINE_float(
    'multi_task_label_threshold', .35,
    'The threshold on bias score to be used for generating bias labels.'
    ' The bias label is generated as '
    '`I(multi_task_score > multi_task_label_threshold)`.')
flags.DEFINE_float(
    'multi_task_loss_weight', .1,
    'Non-negative float for the weight to apply to the bias label. If negative'
    ' then no bias will be added.')
flags.DEFINE_bool(
    'train_on_identity_subgroup_data', False,
    'Whether to add minority examples (CivilCommentsIdentity) to the training'
    ' data.')
flags.DEFINE_bool(
    'test_on_identity_subgroup_data', True,
    'Whether to add minority examples (CivilCommentsIdentity) to the testing'
    ' data.')
flags.DEFINE_bool(
    'test_on_challenge_data', True,
    'Whether to add challenge examples (biased, noisy or uncertain examples) to'
    ' the testing data.')
flags.DEFINE_bool(
    'eval_collab_metrics', False,
    'Whether to compute collaboration effectiveness by score type.')

flags.DEFINE_string(
    'in_dataset_dir', None,
    'Path to in-domain dataset (WikipediaToxicityDataset).')
flags.DEFINE_string(
    'ood_dataset_dir', None,
    'Path to out-of-domain dataset (CivilCommentsDataset).')
flags.DEFINE_string(
    'identity_dataset_dir', None,
    'Path to out-of-domain dataset with identity labels '
    '(CivilCommentsIdentitiesDataset).')

# Model flags
flags.DEFINE_string('model_family', 'bert',
                    'Types of model to use. Can be either TextCNN or BERT.')

# Model flags, BERT.
flags.DEFINE_string(
    'bert_dir', None,
    'Directory to BERT pre-trained checkpoints and config files.')
flags.DEFINE_string(
    'bert_ckpt_dir', None, 'Directory to BERT pre-trained checkpoints. '
    'If None then then default to {bert_dir}/bert_model.ckpt.')
flags.DEFINE_string(
    'bert_config_dir', None, 'Directory to BERT config files. '
    'If None then then default to {bert_dir}/bert_config.json.')
flags.DEFINE_string(
    'bert_tokenizer_tf_hub_url',
    'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
    'TF Hub URL to BERT tokenizer.')


# Evaluation flags.
flags.DEFINE_integer('num_ece_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_integer(
    'num_approx_bins', 1000,
    'Number of bins for approximating collaborative and abstention metrics.')
flags.DEFINE_list(
    'fractions',
    ['0.0', '0.001', '0.005', '0.01', '0.02', '0.05', '0.1', '0.15', '0.2'],
    'A list of fractions of total examples to send to '
    'the moderators (up to 1).')
flags.DEFINE_string('output_dir', '/tmp/toxic_comments', 'Output directory.')
flags.DEFINE_float(
    'ece_label_threshold', 0.7,
    'Threshold used to convert toxicity score into binary labels for computing '
    'Expected Calibration Error (ECE). Default is 0.7 which is the threshold '
    'value recommended by Jigsaw Conversation AI team.')

# Prediction mode.
flags.DEFINE_bool('prediction_mode', False, 'Whether to predict only.')
flags.DEFINE_string(
    'eval_checkpoint_dir', None,
    'The top-level directory to restore the model weights from'
    ' for prediction mode.')
flags.DEFINE_string(
    'checkpoint_name', None, 'The sub-directory to load the checkpoint from for'
    ' prediction mode. If provided then the model will load'
    ' from `{checkpoint_dir}/{checkpoint_name}`, otherwise it'
    ' will load from `{checkpoint_dir}/`.')
flags.DEFINE_bool('identity_prediction', False, 'Whether to do prediction on '
                  'each identity dataset in prediction mode.')
flags.DEFINE_string('identity_specific_dataset_dir', None,
                    'Path to specific out-of-domain dataset with identity '
                    'labels (CivilCommentsIdentitiesDataset).')
flags.DEFINE_string('identity_type_dataset_dir', None,
                    'Path to specific out-of-domain dataset with identity '
                    'types (CivilCommentsIdentitiesDataset).')
flags.DEFINE_string('challenge_dataset_dir', None,
                    'Path to challenge eval datasets that are stored in CSV '
                    'format and under the directory '
                    '{challenge_dataset_dir}/challenge_eval_{dataset_name}')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS



@flags.multi_flags_validator(
    ['prediction_mode', 'eval_checkpoint_dir'],
    message='`eval_checkpoint_dir` should be provided in prediction mode')
def _check_checkpoint_dir_for_prediction_mode(flags_dict):
  return  not flags_dict['prediction_mode'] or (
      flags_dict['eval_checkpoint_dir'] is not None)


@flags.multi_flags_validator(
    ['identity_prediction', 'identity_specific_dataset_dir',
     'identity_type_dataset_dir'], message='`identity_specific_dataset_dir` '
    'and `identity_type_dataset_dir` should be provided when '
    '`identity_prediction` is True.')
def _check_dataset_dir_for_identity_prediction(flags_dict):
  return not flags_dict['identity_prediction'] or (
      flags_dict['identity_specific_dataset_dir'] is not None and
      flags_dict['identity_type_dataset_dir'] is not None)


def get_num_examples(dataset_builder, dataset_name, split_name):
  """Extracts number of examples in a dataset."""
  custom_num_examples = dataset_builder._dataset_builder.info.metadata.get(  # pylint:disable=protected-access
      'num_examples')

  if dataset_name in IDENTITY_LABELS + IDENTITY_TYPES:
    return NUM_EXAMPLES[dataset_name][split_name]
  elif custom_num_examples and custom_num_examples.get(split_name, False):
    # Return custom number of examples if it exists. This is usually true for
    # custom datasets in csv format.
    return custom_num_examples[split_name]
  else:
    # Use official `num_examples` for non-identity-specific and non-custom
    # TFDS datasets (i.e., 'ind', 'ood', 'ood_identity' and 'cv_*' data).
    return dataset_builder.num_examples


def make_train_and_test_dataset_builders(in_dataset_dir,
                                         ood_dataset_dir,
                                         identity_dataset_dir,
                                         train_dataset_type,
                                         test_dataset_type,
                                         use_cross_validation,
                                         num_folds,
                                         train_fold_ids,
                                         return_train_split_name=False,
                                         cv_split_name='train',
                                         train_on_identity_subgroup_data=False,
                                         train_on_multi_task_label='bias',
                                         multi_task_label_threshold=0.,
                                         test_on_identity_subgroup_data=False,
                                         test_on_challenge_data=False,
                                         identity_type_dataset_dir=None,
                                         identity_specific_dataset_dir=None,
                                         challenge_dataset_dir=None,
                                         **ds_kwargs):
  """Defines train and evaluation datasets."""
  maybe_get_train_dir = lambda ds_dir: None if train_dataset_type == 'tfds' else ds_dir
  maybe_get_test_dir = lambda ds_dir: None if test_dataset_type == 'tfds' else ds_dir

  def get_identity_dir(name):
    parent_dir = identity_type_dataset_dir if name in IDENTITY_TYPES else identity_specific_dataset_dir
    return os.path.join(parent_dir, name)

  def get_challenge_dir(name):
    return os.path.join(challenge_dataset_dir, f'challenge_eval_{name}')

  if use_cross_validation and train_dataset_type == 'tfrecord':
    raise ValueError('Cannot use local data when in cross_validation mode.'
                     'Please set `train_dataset_type` to "tfds" or "csv".')

  # Create training data and optionally cross-validation eval sets.
  train_split_name = 'train'
  eval_split_name = 'validation'

  if use_cross_validation:
    cv_train_split_name, cv_eval_split_name = make_cv_train_and_eval_splits(
        num_folds,
        train_fold_ids,
        use_tfds_format=train_dataset_type == 'tfds',
        split_name=cv_split_name)

    if train_dataset_type == 'csv':
      # Overwrite `in_dataset_dir` to point to a split-specific directory.
      in_dataset_dir = os.path.join(in_dataset_dir, cv_train_split_name)
    elif train_dataset_type == 'tfds':
      # Update split name to TFDS' reading-instruction format.
      train_split_name = cv_train_split_name
      eval_split_name = cv_eval_split_name
    else:
      raise ValueError(
          '`train_dataset_type` must be one of ("csv", "tfds") when'
          f'use_cross_validation=True. Got {train_dataset_type}.')

    cv_eval_dataset_builder = IND_DATA_CLS(
        split=eval_split_name,
        dataset_type=train_dataset_type,
        data_dir=maybe_get_train_dir(in_dataset_dir),
        **ds_kwargs)

  train_dataset_builder = IND_DATA_CLS(
      split=train_split_name,
      is_training=True,
      dataset_type=train_dataset_type,
      data_dir=maybe_get_train_dir(in_dataset_dir),
      multi_task_labels=train_on_multi_task_label,
      multi_task_label_threshold=multi_task_label_threshold,
      **ds_kwargs)

  # Optionally, add identity specific examples to training data.
  if train_on_identity_subgroup_data:
    identity_train_dataset_builders = {}
    for dataset_name in IDENTITY_TYPES:
      identity_data_dir = get_identity_dir(dataset_name)
      identity_train_dataset_builders[
          dataset_name] = ds.CivilCommentsIdentitiesDataset(
              split='train', data_dir=identity_data_dir, **ds_kwargs)

  # Create testing data.
  ind_dataset_builder = IND_DATA_CLS(
      split='test',
      dataset_type=test_dataset_type,
      data_dir=maybe_get_test_dir(in_dataset_dir),
      **ds_kwargs)
  ood_dataset_builder = OOD_DATA_CLS(
      split='test',
      dataset_type=test_dataset_type,
      data_dir=maybe_get_test_dir(ood_dataset_dir),
      **ds_kwargs)
  ood_identity_dataset_builder = ds.CivilCommentsIdentitiesDataset(
      split='test',
      dataset_type=test_dataset_type,
      data_dir=maybe_get_test_dir(identity_dataset_dir),
      **ds_kwargs)

  # Optionally, add identity specific examples to testing data.
  if test_on_identity_subgroup_data:
    identity_test_dataset_builders = {}
    for dataset_name in IDENTITY_LABELS + IDENTITY_TYPES:
      # Add to eval only if number of test examples is large enough (>100).
      if NUM_EXAMPLES[dataset_name]['test'] > 100:
        identity_data_dir = get_identity_dir(dataset_name)
        identity_test_dataset_builders[
            dataset_name] = ds.CivilCommentsIdentitiesDataset(
                split='test',
                dataset_type='tfrecord',
                data_dir=identity_data_dir,
                **ds_kwargs)

  # Optionally, add challenge eval sets.
  if test_on_challenge_data:
    challenge_test_dataset_builders = {}
    for dataset_name in CHALLENGE_DATASET_NAMES:
      identity_data_dir = get_challenge_dir(dataset_name)
      challenge_test_dataset_builders[dataset_name] = ds.CivilCommentsDataset(
          split='test',
          dataset_type='csv',
          data_dir=identity_data_dir,
          multi_task_labels=train_on_multi_task_label,
          multi_task_label_threshold=multi_task_label_threshold,
          **ds_kwargs)

  # Gather training dataset builders into dictionaries.
  train_dataset_builders = {
      'train': train_dataset_builder
  }
  if train_on_identity_subgroup_data:
    train_dataset_builders.update(identity_train_dataset_builders)

  # Gather test dataset builders into dictionaries.
  test_dataset_builders = {
      'ind': ind_dataset_builder,
      'ood': ood_dataset_builder,
      'ood_identity': ood_identity_dataset_builder,
  }

  if test_on_identity_subgroup_data:
    test_dataset_builders.update(identity_test_dataset_builders)

  if test_on_challenge_data:
    test_dataset_builders.update(challenge_test_dataset_builders)

  if use_cross_validation:
    test_dataset_builders['cv_eval'] = cv_eval_dataset_builder

  if return_train_split_name:
    return train_dataset_builders, test_dataset_builders, train_split_name

  return train_dataset_builders, test_dataset_builders


def make_prediction_dataset_builders(add_identity_datasets,
                                     identity_dataset_dir,
                                     add_cross_validation_datasets,
                                     cv_dataset_dir,
                                     num_folds,
                                     train_fold_ids,
                                     cv_dataset_type='tfds',
                                     cv_split_name='train',
                                     **ds_kwargs):
  """Adds additional test datasets for prediction mode."""
  get_identity_dir = lambda name: os.path.join(identity_dataset_dir, name)
  maybe_get_cv_dir = (
      lambda name: None  # pylint:disable=g-long-lambda
      if cv_dataset_type == 'tfds' else os.path.join(cv_dataset_dir, name))

  def _standardize_split_name(name):
    """Maps split name to a canonical name (e.g., 'train_0_1' to 'train')."""
    for standard_name in ds.DATA_SPLIT_NAMES:
      if standard_name in name:
        return standard_name

    raise ValueError(
        f'split name `{name}` does not match any of the offical split '
        f'names {ds.DATA_SPLIT_NAMES}.')

  dataset_builders = {}

  # Adds identity dataset that has > 100 observations.
  if add_identity_datasets:
    for dataset_name in IDENTITY_LABELS + IDENTITY_TYPES:
      if NUM_EXAMPLES[dataset_name]['test'] > 100:
        identity_data_dir = get_identity_dir(dataset_name)
        dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test',
            dataset_type='tfrecord',
            data_dir=identity_data_dir,
            **ds_kwargs)

  # Adds cross validation folds for evaluation.
  if add_cross_validation_datasets:
    # make_cv_train_and_eval_splits returns a 5-tuple when
    # `return_individual_folds=True`.
    _, _, train_fold_names, eval_fold_names, eval_fold_ids = make_cv_train_and_eval_splits(  # pylint: disable=unbalanced-tuple-unpacking
        num_folds,
        train_fold_ids,
        split_name=cv_split_name,
        use_tfds_format=(cv_dataset_type == 'tfds'),
        return_individual_folds=True)

    for cv_fold_id, cv_fold_name in zip(train_fold_ids, train_fold_names):
      dataset_builders[f'cv_train_fold_{cv_fold_id}'] = IND_DATA_CLS(
          split=_standardize_split_name(cv_fold_name),
          dataset_type=cv_dataset_type,
          is_training=False,
          data_dir=maybe_get_cv_dir(cv_fold_name),
          **ds_kwargs)

    for cv_fold_id, cv_fold_name in zip(eval_fold_ids, eval_fold_names):
      dataset_builders[f'cv_eval_fold_{cv_fold_id}'] = IND_DATA_CLS(
          split=_standardize_split_name(cv_fold_name),
          dataset_type=cv_dataset_type,
          is_training=False,
          data_dir=maybe_get_cv_dir(cv_fold_name),
          **ds_kwargs)

  return dataset_builders


def build_datasets(train_dataset_builders, test_dataset_builders,
                   batch_size, test_batch_size, per_core_batch_size):
  """Builds train and test datasets."""
  train_datasets = {}
  test_datasets = {}
  train_steps_per_epoch = {}
  test_steps_per_eval = {}

  for dataset_name, dataset_builder in train_dataset_builders.items():
    train_datasets[dataset_name] = dataset_builder.load(
        batch_size=per_core_batch_size)
    train_num_examples = get_num_examples(
        dataset_builder, dataset_name, split_name='train')
    train_steps_per_epoch[dataset_name] = train_num_examples // batch_size

  for dataset_name, dataset_builder in test_dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=test_batch_size)
    # Set `split_name` to dataset_builder.split in case we want to eval on
    # `csv` data that is generated as training data. This will be the case
    # for k-fold cross validation.
    test_num_examples = get_num_examples(
        dataset_builder, dataset_name, split_name=dataset_builder.split)
    test_steps_per_eval[dataset_name] = test_num_examples // test_batch_size

  return train_datasets, test_datasets, train_steps_per_epoch, test_steps_per_eval


def make_cv_train_and_eval_splits(num_folds,
                                  train_fold_ids,
                                  split_name='train',
                                  use_tfds_format=True,
                                  return_individual_folds=False):
  """Defines the train and evaluation splits for cross validation."""
  fold_ranges = np.linspace(0, 100, num_folds + 1, dtype=int)
  if use_tfds_format:
    # Uses the format for TFDS splicing.
    split_fn = lambda k: f'{split_name}[{fold_ranges[k]}%:{fold_ranges[k+1]}%]'
  else:
    split_fn = lambda k: f'{split_name}_{fold_ranges[k]}_{fold_ranges[k+1]}'

  all_splits = [split_fn(k) for k in range(num_folds)]

  # Make train and eval fold IDs.
  if isinstance(train_fold_ids, str):
    # If train_fold_ids is a comma-separated string of "{ID1},{ID2},{ID3},.."
    # split it into list.
    train_fold_ids = train_fold_ids.split(',')

  train_fold_ids = [int(fold_id) for fold_id in train_fold_ids]
  eval_fold_ids = list(set(range(num_folds)) - set(train_fold_ids))
  eval_fold_ids.sort()

  # Collect split names for train and eval folds.
  train_split_list = [all_splits[fold_id] for fold_id in train_fold_ids]
  eval_split_list = [all_splits[fold_id] for fold_id in eval_fold_ids]

  # Make the train and eval split names to be used by TFDS loader.
  train_split = '+'.join(train_split_list)
  eval_split = '+'.join(eval_split_list)

  if not return_individual_folds:
    return train_split, eval_split

  return train_split, eval_split, train_split_list, eval_split_list, eval_fold_ids


def save_prediction(data, path):
  """Save the data as numpy array to the path."""
  with (tf.io.gfile.GFile(path + '.npy', 'w')) as test_file:
    np.save(test_file, np.array(data))


def create_class_weight(train_dataset_builders=None,
                        test_dataset_builders=None):
  """Creates a dictionary of class weights for computing eval metrics."""

  def generate_weight(num_positive_examples, num_examples):
    pos_fraction = num_positive_examples / num_examples
    neg_fraction = 1 - pos_fraction
    return [0.5 / neg_fraction, 0.5 / pos_fraction]

  class_weight = {}
  if train_dataset_builders:
    for dataset_name, dataset_builder in train_dataset_builders.items():
      num_positive_examples = NUM_POS_EXAMPLES.get(dataset_name, None)
      if not num_positive_examples:
        # Equal weight if not an official split.
        class_weight['train/{}'.format(dataset_name)] = [0.5, 0.5]
      else:
        class_weight['train/{}'.format(dataset_name)] = generate_weight(
            num_positive_examples['train'], dataset_builder.num_examples)

  if test_dataset_builders:
    for dataset_name, dataset_builder in test_dataset_builders.items():
      num_positive_examples = NUM_POS_EXAMPLES.get(dataset_name, None)
      if not num_positive_examples:
        # Equal weight if not an official split.
        class_weight['test/{}'.format(dataset_name)] = [0.5, 0.5]
      else:
        class_weight['test/{}'.format(dataset_name)] = generate_weight(
            NUM_POS_EXAMPLES[dataset_name]['test'],
            dataset_builder.num_examples)

  return class_weight


def create_train_and_test_metrics(test_datasets,
                                  num_classes,
                                  num_ece_bins,
                                  ece_label_threshold,
                                  eval_collab_metrics,
                                  num_approx_bins,
                                  log_eval_time=True,
                                  train_on_multi_task_label=False):
  """Creates metrics for train and test eval."""
  # Train metrics.
  metrics = {
      'train/negative_log_likelihood':
          tf.keras.metrics.Mean(),
      'train/accuracy':
          tf.keras.metrics.Accuracy(),
      'train/accuracy_weighted':
          tf.keras.metrics.Accuracy(),
      'train/auroc':
          tf.keras.metrics.AUC(),
      'train/loss':
          tf.keras.metrics.Mean(),
      'train/ece':
          rm.metrics.ExpectedCalibrationError(num_bins=num_ece_bins),
      'train/precision':
          tf.keras.metrics.Precision(),
      'train/recall':
          tf.keras.metrics.Recall(),
      'train/f1':
          tfa_metrics.F1Score(
              num_classes=num_classes,
              average='micro',
              threshold=ece_label_threshold),
  }

  if train_on_multi_task_label:
    metrics.update({
        'train/multi_task_accuracy':
            tf.keras.metrics.Accuracy(),
        'train/multi_task_aupr':
            tf.keras.metrics.AUC(curve='PR'),
        'train/multi_task_auroc':
            tf.keras.metrics.AUC(curve='ROC'),
        'train/multi_task_precision':
            tf.keras.metrics.Precision(),
        'train/multi_task_recall':
            tf.keras.metrics.Recall(),
        'train/multi_task_f1':
            tfa_metrics.F1Score(
                num_classes=num_classes, average='micro', threshold=0.5),
    })

  # Main test metrics.
  metrics.update({
      'test/negative_log_likelihood':
          tf.keras.metrics.Mean(),
      'test/auroc':
          tf.keras.metrics.AUC(curve='ROC'),
      'test/aupr':
          tf.keras.metrics.AUC(curve='PR'),
      'test/brier':
          tf.keras.metrics.MeanSquaredError(),
      'test/brier_weighted':
          tf.keras.metrics.MeanSquaredError(),
      'test/ece':
          rm.metrics.ExpectedCalibrationError(num_bins=num_ece_bins),
      'test/acc':
          tf.keras.metrics.Accuracy(),
      'test/acc_weighted':
          tf.keras.metrics.Accuracy(),
      'test/precision':
          tf.keras.metrics.Precision(),
      'test/recall':
          tf.keras.metrics.Recall(),
      'test/f1':
          tfa_metrics.F1Score(
              num_classes=num_classes,
              average='micro',
              threshold=ece_label_threshold),
      'test/calibration_auroc':
          tc_metrics.CalibrationAUC(
              curve='ROC', correct_pred_as_pos_label=False),
      'test/calibration_auprc':
          tc_metrics.CalibrationAUC(
              curve='PR', correct_pred_as_pos_label=False)
  })

  if log_eval_time:
    metrics['test/eval_time'] = tf.keras.metrics.Mean()

  if train_on_multi_task_label:
    metrics.update({
        'test/multi_task_accuracy':
            tf.keras.metrics.Accuracy(),
        'test/multi_task_aupr':
            tf.keras.metrics.AUC(curve='PR'),
        'test/multi_task_auroc':
            tf.keras.metrics.AUC(curve='ROC'),
        'test/multi_task_precision':
            tf.keras.metrics.Precision(),
        'test/multi_task_recall':
            tf.keras.metrics.Recall(),
        'test/multi_task_f1':
            tfa_metrics.F1Score(
                num_classes=num_classes, average='micro', threshold=0.5),
    })

  # Main collaborative metrics.
  if eval_collab_metrics:
    for policy in ('uncertainty', 'toxicity'):
      metrics.update({
          'test_{}/calibration_auroc'.format(policy):
              tc_metrics.CalibrationAUC(curve='ROC'),
          'test_{}/calibration_auprc'.format(policy):
              tc_metrics.CalibrationAUC(curve='PR')
      })

      for fraction in FLAGS.fractions:
        metrics.update({
            'test_{}/collab_acc_{}'.format(policy, fraction):
                rm.metrics.OracleCollaborativeAccuracy(
                    fraction=float(fraction), num_bins=num_approx_bins),
            'test_{}/abstain_prec_{}'.format(policy, fraction):
                tc_metrics.AbstainPrecision(
                    abstain_fraction=float(fraction),
                    num_approx_bins=num_approx_bins),
            'test_{}/abstain_recall_{}'.format(policy, fraction):
                tc_metrics.AbstainRecall(
                    abstain_fraction=float(fraction),
                    num_approx_bins=num_approx_bins),
            'test_{}/collab_auroc_{}'.format(policy, fraction):
                tc_metrics.OracleCollaborativeAUC(
                    oracle_fraction=float(fraction), num_bins=num_approx_bins),
            'test_{}/collab_auprc_{}'.format(policy, fraction):
                tc_metrics.OracleCollaborativeAUC(
                    oracle_fraction=float(fraction),
                    curve='PR',
                    num_bins=num_approx_bins),
        })

  # Dataset-specific test metrics.
  for dataset_name in test_datasets.keys():
    if dataset_name != 'ind':
      metrics.update({
          'test/nll_{}'.format(dataset_name):
              tf.keras.metrics.Mean(),
          'test/auroc_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='ROC'),
          'test/aupr_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='PR'),
          'test/brier_{}'.format(dataset_name):
              tf.keras.metrics.MeanSquaredError(),
          'test/brier_weighted_{}'.format(dataset_name):
              tf.keras.metrics.MeanSquaredError(),
          'test/ece_{}'.format(dataset_name):
              rm.metrics.ExpectedCalibrationError(num_bins=num_ece_bins),
          'test/acc_{}'.format(dataset_name):
              tf.keras.metrics.Accuracy(),
          'test/acc_weighted_{}'.format(dataset_name):
              tf.keras.metrics.Accuracy(),
          'test/precision_{}'.format(dataset_name):
              tf.keras.metrics.Precision(),
          'test/recall_{}'.format(dataset_name):
              tf.keras.metrics.Recall(),
          'test/f1_{}'.format(dataset_name):
              tfa_metrics.F1Score(
                  num_classes=num_classes,
                  average='micro',
                  threshold=ece_label_threshold),
          'test/calibration_auroc_{}'.format(dataset_name):
              tc_metrics.CalibrationAUC(
                  curve='ROC', correct_pred_as_pos_label=False),
          'test/calibration_auprc_{}'.format(dataset_name):
              tc_metrics.CalibrationAUC(
                  curve='PR', correct_pred_as_pos_label=False)
      })

    if log_eval_time:
      metrics['test/eval_time_{}'.format(
          dataset_name)] = tf.keras.metrics.Mean()

    if train_on_multi_task_label and dataset_name in CHALLENGE_DATASET_NAMES:
      metrics.update({
          'test/multi_task_accuracy_{}'.format(dataset_name):
              tf.keras.metrics.Accuracy(),
          'test/multi_task_aupr_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='PR'),
          'test/multi_task_auroc_{}'.format(dataset_name):
              tf.keras.metrics.AUC(curve='ROC'),
          'test/multi_task_precision_{}'.format(dataset_name):
              tf.keras.metrics.Precision(),
          'test/multi_task_recall_{}'.format(dataset_name):
              tf.keras.metrics.Recall(),
          'test/multi_task_f1_{}'.format(dataset_name):
              tfa_metrics.F1Score(
                  num_classes=num_classes, average='micro', threshold=0.5),
      })

    if eval_collab_metrics:
      for policy in ('uncertainty', 'toxicity'):
        metrics.update({
            'test_{}/calibration_auroc_{}'.format(policy, dataset_name):
                tc_metrics.CalibrationAUC(curve='ROC'),
            'test_{}/calibration_auprc_{}'.format(policy, dataset_name):
                tc_metrics.CalibrationAUC(curve='PR'),
        })

        for fraction in FLAGS.fractions:
          metrics.update({
              'test_{}/collab_acc_{}_{}'.format(policy, fraction, dataset_name):
                  rm.metrics.OracleCollaborativeAccuracy(
                      fraction=float(fraction), num_bins=num_approx_bins),
              'test_{}/abstain_prec_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.AbstainPrecision(
                      abstain_fraction=float(fraction),
                      num_approx_bins=num_approx_bins),
              'test_{}/abstain_recall_{}_{}'.format(policy, fraction,
                                                    dataset_name):
                  tc_metrics.AbstainRecall(
                      abstain_fraction=float(fraction),
                      num_approx_bins=num_approx_bins),
              'test_{}/collab_auroc_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.OracleCollaborativeAUC(
                      oracle_fraction=float(fraction),
                      num_bins=num_approx_bins),
              'test_{}/collab_auprc_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                  tc_metrics.OracleCollaborativeAUC(
                      oracle_fraction=float(fraction),
                      curve='PR',
                      num_bins=num_approx_bins),
          })

  return metrics


def make_test_metrics_update_fn(dataset_name,
                                sample_weight,
                                num_classes,
                                labels,
                                probs,
                                multi_task_labels,
                                multi_task_probs,
                                negative_log_likelihood,
                                eval_time=None,
                                ece_label_threshold=0.5,
                                train_on_multi_task_label=False,
                                eval_collab_metrics=False):
  """Makes an update function for test step metrics."""
  # Cast labels to discrete for ECE computation.
  ece_labels = tf.cast(labels > ece_label_threshold, tf.float32)
  one_hot_labels = tf.one_hot(tf.cast(ece_labels, tf.int32), depth=num_classes)
  ece_probs = tf.concat([1. - probs, probs], axis=1)
  pred_labels = tf.math.argmax(ece_probs, axis=-1)
  auc_probs = tf.squeeze(probs, axis=1)

  # Use normalized binary predictive variance as the confidence score.
  # Since the prediction variance p*(1-p) is within range (0, 0.25),
  # normalize it by maximum value so the confidence is between (0, 1).
  calib_confidence = 1. - probs * (1. - probs) / .25

  # Compute bias prediction results.
  update_multi_task_metrics = (
      train_on_multi_task_label and isinstance(multi_task_labels, tf.Tensor) and
      isinstance(multi_task_probs, tf.Tensor))
  if update_multi_task_metrics:
    multi_task_ece_probs = tf.concat([1. - multi_task_probs, multi_task_probs],
                                     axis=1)
    multi_task_preds = tf.math.argmax(multi_task_ece_probs, axis=-1)
    multi_task_one_hot_labels = tf.one_hot(
        tf.cast(multi_task_labels, tf.int32), depth=num_classes)
    multi_task_auc_probs = tf.squeeze(multi_task_probs, axis=1)

  def update_fn(metrics):
    if dataset_name == 'ind':
      metrics['test/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['test/auroc'].update_state(labels, auc_probs)
      metrics['test/aupr'].update_state(labels, auc_probs)
      metrics['test/brier'].update_state(labels, auc_probs)
      metrics['test/brier_weighted'].update_state(
          tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
      metrics['test/ece'].add_batch(ece_probs, label=ece_labels)
      metrics['test/acc'].update_state(ece_labels, pred_labels)
      metrics['test/acc_weighted'].update_state(
          ece_labels, pred_labels, sample_weight=sample_weight)
      metrics['test/precision'].update_state(ece_labels, pred_labels)
      metrics['test/recall'].update_state(ece_labels, pred_labels)
      metrics['test/f1'].update_state(one_hot_labels, ece_probs)
      metrics['test/calibration_auroc'].update_state(ece_labels, pred_labels,
                                                     calib_confidence)
      metrics['test/calibration_auprc'].update_state(ece_labels, pred_labels,
                                                     calib_confidence)

      if eval_time:
        metrics['test/eval_time'].update_state(eval_time)

      if update_multi_task_metrics:
        metrics['test/multi_task_accuracy'].update_state(
            multi_task_labels, multi_task_preds)
        metrics['test/multi_task_auroc'].update_state(multi_task_labels,
                                                      multi_task_auc_probs)
        metrics['test/multi_task_aupr'].update_state(multi_task_labels,
                                                     multi_task_auc_probs)
        metrics['test/multi_task_precision'].update_state(
            multi_task_labels, multi_task_preds)
        metrics['test/multi_task_recall'].update_state(multi_task_labels,
                                                       multi_task_preds)
        metrics['test/multi_task_f1'].update_state(multi_task_one_hot_labels,
                                                   multi_task_ece_probs)

      if eval_collab_metrics:
        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.reshape(confidence, [-1])

          metrics['test_{}/calibration_auroc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)
          metrics['test_{}/calibration_auprc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}'.format(policy, fraction)].add_batch(
                ece_probs,
                label=ece_labels,
                custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/abstain_recall_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/collab_auroc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

    else:
      metrics['test/nll_{}'.format(dataset_name)].update_state(
          negative_log_likelihood)
      metrics['test/auroc_{}'.format(dataset_name)].update_state(
          labels, auc_probs)
      metrics['test/aupr_{}'.format(dataset_name)].update_state(
          labels, auc_probs)
      metrics['test/brier_{}'.format(dataset_name)].update_state(
          labels, auc_probs)
      metrics['test/brier_weighted_{}'.format(dataset_name)].update_state(
          tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
      metrics['test/ece_{}'.format(dataset_name)].add_batch(
          ece_probs, label=ece_labels)
      metrics['test/acc_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels)
      metrics['test/acc_weighted_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels, sample_weight=sample_weight)
      metrics['test/precision_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels)
      metrics['test/recall_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels)
      metrics['test/f1_{}'.format(dataset_name)].update_state(
          one_hot_labels, ece_probs)
      metrics['test/calibration_auroc_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels, calib_confidence)
      metrics['test/calibration_auprc_{}'.format(dataset_name)].update_state(
          ece_labels, pred_labels, calib_confidence)

      if eval_time:
        metrics['test/eval_time_{}'.format(dataset_name)].update_state(
            eval_time)

      if update_multi_task_metrics:
        metrics['test/multi_task_accuracy_{}'.format(
            dataset_name)].update_state(multi_task_labels, multi_task_preds)
        metrics['test/multi_task_auroc_{}'.format(dataset_name)].update_state(
            multi_task_labels, multi_task_auc_probs)
        metrics['test/multi_task_aupr_{}'.format(dataset_name)].update_state(
            multi_task_labels, multi_task_auc_probs)
        metrics['test/multi_task_precision_{}'.format(
            dataset_name)].update_state(multi_task_labels, multi_task_preds)
        metrics['test/multi_task_recall_{}'.format(dataset_name)].update_state(
            multi_task_labels, multi_task_preds)
        metrics['test/multi_task_f1_{}'.format(dataset_name)].update_state(
            multi_task_one_hot_labels, multi_task_ece_probs)

      if eval_collab_metrics:
        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.reshape(confidence, [-1])

          metrics['test_{}/calibration_auroc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)
          metrics['test_{}/calibration_auprc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}_{}'.format(
                policy, fraction, dataset_name)].add_batch(
                    ece_probs,
                    label=ece_labels,
                    custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/abstain_recall_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/collab_auroc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

  return update_fn


def create_config(config_dir: str) -> configs.BertConfig:
  """Load a BERT config object from directory."""
  with tf.io.gfile.GFile(config_dir) as config_file:
    bert_config = json.load(config_file)
  return configs.BertConfig(**bert_config)


def create_feature_and_label(inputs):
  """Creates features and labels from model inputs."""
  # Extracts data and squeezes redundant dimensions.
  def _may_squeeze_dimension(batch_tensor):
    """Remove redundant dimensions in the input data."""
    if len(batch_tensor.shape) == 2:
      return batch_tensor
    return tf.squeeze(batch_tensor, axis=1)

  input_ids = _may_squeeze_dimension(inputs['input_ids'])
  input_mask = _may_squeeze_dimension(inputs['input_mask'])
  segment_ids = _may_squeeze_dimension(inputs['segment_ids'])

  # Process labels.
  labels = inputs['labels']
  additional_labels = {}
  for additional_label in IDENTITY_LABELS:
    if additional_label in inputs:
      additional_labels[additional_label] = inputs[additional_label]
  # labels = tf.stack([labels, 1. - labels], axis=-1)

  return [input_ids, input_mask, segment_ids], labels, additional_labels


def create_optimizer(
    initial_lr: float,
    steps_per_epoch: int,
    epochs: int,
    warmup_proportion: float,
    end_lr: float = 0.0,
    optimizer_type: str = 'adamw',
    beta_1: float = 0.9) -> tf.keras.optimizers.Optimizer:
  """Creates a BERT optimizer with learning rate schedule."""
  num_train_steps = steps_per_epoch * epochs
  num_warmup_steps = int(num_train_steps * warmup_proportion)
  return optimization.create_optimizer(
      initial_lr,
      num_train_steps,
      num_warmup_steps,
      end_lr=end_lr,
      optimizer_type=optimizer_type,
      beta_1=beta_1)


def resolve_bert_ckpt_and_config_dir(bert_model_type, bert_dir, bert_config_dir,
                                     bert_ckpt_dir):
  """Resolves BERT checkpoint and config file directories."""

  missing_ckpt_or_config_dir = not (bert_ckpt_dir and bert_config_dir)
  if missing_ckpt_or_config_dir:
    if not bert_dir:
      raise ValueError('bert_dir cannot be empty.')

    if not bert_config_dir:
      bert_config_dir = os.path.join(bert_dir, 'bert_config.json')

    if not bert_ckpt_dir:
      bert_ckpt_dir = os.path.join(bert_dir, 'bert_model.ckpt')
  return bert_config_dir, bert_ckpt_dir


def load_bert_weight_from_ckpt(
    bert_model: tf.keras.Model,
    bert_ckpt_dir: str,
    repl_patterns: Optional[Dict[str, str]] = None
) -> Tuple[tf.keras.Model, Mapping[str, str], Mapping[str, Any]]:
  """Loads checkpoint weights and match to model weights if applicable.

  Args:
    bert_model: The BERT Encoder model whose weights to load from checkpoints.
    bert_ckpt_dir: Path to BERT pre-trained checkpoints.
    repl_patterns: A mapping of regex string patterns and their replacements. To
      be used to update checkpoint weight names so they match those in
      bert_model (e.g., via re.sub(pattern, repl, weight_name))

  Returns:
    bert_model: The BERT Encoder model with loaded weights.
    names_to_keys: A dict mapping of weight name to checkpoint keys.
    keys_to_weights:  A dict mapping of checkpoint keys to weight values.
  """
  # Load a dict mapping of weight names to their corresponding checkpoint keys.
  names_to_keys = object_graph_key_mapping(bert_ckpt_dir)
  if repl_patterns:
    # Update weight names so they match those in bert_model
    names_to_keys = {
        update_weight_name(repl_patterns, weight_name): weight_key
        for weight_name, weight_key in names_to_keys.items()
    }

  # Load a dict mapping of checkpoint keys to weight values.
  logging.info('Loading weights from checkpoint: %s', bert_ckpt_dir)
  keys_to_weights = load_ckpt_keys_to_weight_mapping(bert_ckpt_dir)

  # Arranges the pre-trained weights in the order of model weights.
  init_weight_list = match_ckpt_weights_to_model(bert_model, names_to_keys,
                                                 keys_to_weights)

  # Load weights into model.
  bert_model.set_weights(init_weight_list)

  return bert_model, names_to_keys, keys_to_weights


def load_ckpt_keys_to_weight_mapping(ckpt_path: str) -> Mapping[str, Any]:
  """Loads weight values and their checkpoint keys from BERT checkpoint."""
  init_vars = tf.train.list_variables(ckpt_path)

  keys_to_weights = {}
  for name, _ in init_vars:
    var = tf.train.load_variable(ckpt_path, name)
    keys_to_weights[name] = var

  return keys_to_weights


def match_ckpt_weights_to_model(
    model: tf.keras.Model,
    names_to_keys: Mapping[str, str],
    keys_to_weights: Mapping[str, Any]) -> List[Any]:
  """Produces a list of checkpoint weights in the order specified by model."""
  init_weight_list = []

  for weight in model.weights:
    # Look up weight name in checkpoint weight names.
    weight_name = weight.name.replace(':0', '')
    ckpt_key = names_to_keys.get(weight_name, None)

    if ckpt_key:
      init_weight = keys_to_weights[ckpt_key]
    else:
      logging.info(
          '"%s" not found in checkpoint. '
          'Using randomly initialized values.', weight_name)
      init_weight = weight.numpy()

    init_weight_list.append(init_weight)

  return init_weight_list


def update_weight_name(repl_patterns: Dict[str, str], weight_name: str) -> str:
  """Updates weight names according a dictionary of replacement patterns."""
  # Create a regular expression from all of the dictionary keys
  regex = re.compile('|'.join(map(re.escape, repl_patterns.keys())))

  # For each match, look up the corresponding value in the repl_patterns dict.
  return regex.sub(lambda match: repl_patterns[match.group(0)], weight_name)


def object_graph_key_mapping(checkpoint_path: str) -> Dict[str, str]:
  """Return name to key mappings from the checkpoint."""
  reader = tf.train.load_checkpoint(checkpoint_path)
  object_graph_string = reader.get_tensor('_CHECKPOINTABLE_OBJECT_GRAPH')
  object_graph_proto = trackable_object_graph_pb2.TrackableObjectGraph()
  object_graph_proto.ParseFromString(object_graph_string)
  names_to_keys = {}
  for node in object_graph_proto.nodes:
    for attribute in node.attributes:
      names_to_keys[attribute.full_name] = attribute.checkpoint_key
  return names_to_keys
