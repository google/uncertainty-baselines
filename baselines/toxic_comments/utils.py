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
import tensorflow as tf

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
    # Return custom number of examples if it exists.
    return custom_num_examples[split_name]
  else:
    # Use official `num_examples` for non-identity-specific and non-custom
    # TFDS datasets (i.e., 'ind', 'ood', 'ood_identity' and 'cv_*' data).
    return dataset_builder.num_examples


def make_train_and_test_dataset_builders(in_dataset_dir,
                                         ood_dataset_dir,
                                         identity_dataset_dir,
                                         use_local_data,
                                         use_cross_validation,
                                         num_folds,
                                         train_fold_ids,
                                         return_train_split_name=False,
                                         train_on_identity_subgroup_data=False,
                                         test_on_identity_subgroup_data=False,
                                         identity_type_dataset_dir=None,
                                         identity_specific_dataset_dir=None,
                                         **ds_kwargs):
  """Defines train and evaluation datasets."""
  maybe_get_dir = lambda ds_dir: ds_dir if use_local_data else None

  def get_identity_dir(name):
    parent_dir = identity_type_dataset_dir if name in IDENTITY_TYPES else identity_specific_dataset_dir
    return os.path.join(parent_dir, name)

  if use_cross_validation and use_local_data:
    raise ValueError('Cannot use local data when in cross_validation mode.'
                     'Please set `use_local_data` to False.')

  # Create training data and optionally cross-validation eval sets.
  train_split_name = 'train'
  if use_cross_validation:
    train_split_name, eval_split_name = make_cv_train_and_eval_splits(
        num_folds, train_fold_ids)
    cv_eval_dataset_builder = IND_DATA_CLS(split=eval_split_name, **ds_kwargs)

  train_dataset_builder = IND_DATA_CLS(
      split=train_split_name,
      is_training=True,
      data_dir=maybe_get_dir(in_dataset_dir),
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
      split='test', data_dir=maybe_get_dir(in_dataset_dir), **ds_kwargs)
  ood_dataset_builder = OOD_DATA_CLS(
      split='test', data_dir=maybe_get_dir(ood_dataset_dir), **ds_kwargs)
  ood_identity_dataset_builder = ds.CivilCommentsIdentitiesDataset(
      split='test', data_dir=maybe_get_dir(identity_dataset_dir), **ds_kwargs)

  # Optionally, add identity specific examples to testing data.
  if test_on_identity_subgroup_data:
    identity_test_dataset_builders = {}
    for dataset_name in IDENTITY_LABELS + IDENTITY_TYPES:
      # Add to eval only if number of test examples is large enough (>100).
      if NUM_EXAMPLES[dataset_name]['test'] > 100:
        identity_data_dir = get_identity_dir(dataset_name)
        identity_test_dataset_builders[
            dataset_name] = ds.CivilCommentsIdentitiesDataset(
                split='test', data_dir=identity_data_dir, **ds_kwargs)

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

  if use_cross_validation:
    test_dataset_builders['cv_eval'] = cv_eval_dataset_builder

  if return_train_split_name:
    return train_dataset_builders, test_dataset_builders, train_split_name

  return train_dataset_builders, test_dataset_builders


def make_prediction_dataset_builders(add_identity_datasets,
                                     use_cross_validation, identity_dataset_dir,
                                     use_local_data, num_folds, train_fold_ids,
                                     **ds_kwargs):
  """Adds additional test datasets for prediction mode."""
  maybe_get_identity_dir = (
      lambda name: os.path.join(identity_dataset_dir, name)  # pylint: disable=g-long-lambda
      if use_local_data else None)

  dataset_builders = {}

  # Adds identity dataset that has > 100 observations.
  if add_identity_datasets:
    for dataset_name in IDENTITY_LABELS + IDENTITY_TYPES:
      if NUM_EXAMPLES[dataset_name]['test'] > 100:
        identity_data_dir = maybe_get_identity_dir(dataset_name)
        dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test', data_dir=identity_data_dir, **ds_kwargs)

  # Adds cross validation folds for evaluation.
  if use_cross_validation:
    # make_cv_train_and_eval_splits returns a 5-tuple when
    # `return_individual_folds=True`.
    _, _, train_fold_names, eval_fold_names, eval_fold_ids = make_cv_train_and_eval_splits(  # pylint: disable=unbalanced-tuple-unpacking
        num_folds,
        train_fold_ids,
        return_individual_folds=True)

    for cv_fold_id, cv_fold_name in zip(train_fold_ids, train_fold_names):
      dataset_builders[
          f'cv_train_fold_{cv_fold_id}'] = IND_DATA_CLS(
              split=cv_fold_name, **ds_kwargs)

    for cv_fold_id, cv_fold_name in zip(eval_fold_ids, eval_fold_names):
      dataset_builders[
          f'cv_eval_fold_{cv_fold_id}'] = IND_DATA_CLS(
              split=cv_fold_name, **ds_kwargs)

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
    test_num_examples = get_num_examples(
        dataset_builder, dataset_name, split_name='test')
    test_steps_per_eval[dataset_name] = test_num_examples // test_batch_size

  return train_datasets, test_datasets, train_steps_per_epoch, test_steps_per_eval


def make_cv_train_and_eval_splits(num_folds,
                                  train_fold_ids,
                                  return_individual_folds=False):
  """Defines the train and evaluation splits for cross validation."""
  fold_ranges = np.linspace(0, 100, num_folds + 1, dtype=int)
  all_splits = [
      f'train[{fold_ranges[k]}%:{fold_ranges[k+1]}%]' for k in range(num_folds)
  ]

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
