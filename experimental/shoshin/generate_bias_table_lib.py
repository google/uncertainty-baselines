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

"""Utilities for Introspective Active Sampling.

Library of utilities for the Introspecive Active Sampling method. Includes a
function to generate a table mapping example ID to bias label, which can be
used to train the bias output head.
"""

import os
from typing import List, Optional

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

import data  # local file import from experimental.shoshin


EXAMPLE_ID_KEY = 'example_id'
BIAS_LABEL_KEY = 'bias_label'
# Subdirectory for models trained on splits in FLAGS.output_dir.
COMBOS_SUBDIR = 'combos'


def load_existing_bias_table(path_to_table: str):
  """Loads bias table from file."""
  df = pd.read_csv(path_to_table)
  key_tensor = np.array([eval(x).decode('UTF-8') for    #  pylint:disable=eval-used
                         x in df[EXAMPLE_ID_KEY].to_list()])
  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(
          key_tensor, dtype=tf.string),
      values=tf.convert_to_tensor(
          df[BIAS_LABEL_KEY].to_numpy(), dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def get_example_id_to_bias_label_table(
    dataloader: data.Dataloader,
    combos_dir: str, trained_models: List[tf.keras.Model],
    num_splits: int,
    bias_percentile_threshold: float,
    bias_value_threshold: Optional[float] = None,
    save_dir: Optional[str] = None,
    save_table: Optional[bool] = True
) -> tf.lookup.StaticHashTable:
  """Generates a lookup table mapping example ID to bias label.

  Args:
    dataloader: Dataclass object containing training and validation data.
    combos_dir: Directory of model checkpoints by the combination of data splits
      used in training.
    trained_models: List of trained models.
    num_splits: Total number of slices that data was split into.
    bias_percentile_threshold: Float representing the percentile of bias
      values to give a label of 1 (and 0 for all others).
    bias_value_threshold: Float representing the bias value threshold, above
      which examples will receive a bias label of 1 (and 0 if below). Use
      percentile by default.
    save_dir: Directory in which bias table will be saved as CSV.
    save_table: Boolean for whether or not to save table.

  Returns:
    A lookup table mapping example ID to bias label.
  """
  example_ids_all = []
  bias_values_all = []
  bias_labels_all = []
  for split_idx in range(num_splits):
    # For each split of data,
    # 1. Get the models that included this split (as in-domain training data).
    # 2. Get the models that excluded this split (as out-of-distribution data).
    # 3. Calculate the bias value and, using the threshold, bias label.
    id_predictions_all = []
    ood_predictions_all = []
    labels = list(dataloader.train_splits[split_idx].map(
        lambda feats, label, example_id: label).as_numpy_iterator())
    labels += list(dataloader.val_splits[split_idx].map(
        lambda feats, label, example_id: label).as_numpy_iterator())
    labels = np.concatenate(labels)
    for combo_idx, combo in enumerate(tf.io.gfile.listdir(combos_dir)):
      if str(split_idx) in combo:
        model = trained_models[combo_idx]
        id_predictions_train = model.predict(dataloader.train_splits[split_idx])
        id_predictions_val = model.predict(dataloader.val_splits[split_idx])
        id_predictions = tf.concat(
            [id_predictions_train['main'], id_predictions_val['main']], axis=0)
        id_predictions = tf.gather_nd(
            id_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        id_predictions_all.append(id_predictions)
      else:
        model = trained_models[combo_idx]
        ood_predictions_train = model.predict(
            dataloader.train_splits[split_idx])
        ood_predictions_val = model.predict(dataloader.val_splits[split_idx])
        ood_predictions = tf.concat(
            [ood_predictions_train['main'], ood_predictions_val['main']],
            axis=0)
        ood_predictions = tf.gather_nd(
            ood_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        ood_predictions_all.append(ood_predictions)

    example_ids = list(dataloader.train_splits[split_idx].map(
        lambda feats, label, example_id: example_id).as_numpy_iterator())
    example_ids += list(dataloader.val_splits[split_idx].map(
        lambda feats, label, example_id: example_id).as_numpy_iterator())
    example_ids = np.concatenate(example_ids)
    example_ids_all.append(example_ids)
    id_predictions_avg = np.average(np.stack(id_predictions_all), axis=0)
    ood_predictions_avg = np.average(np.stack(ood_predictions_all), axis=0)
    bias_values = np.absolute(
        np.subtract(id_predictions_avg, ood_predictions_avg))
    # Calculate bias labels using value threshold by default.
    # If percentile is specified, use percentile instead.
    if bias_percentile_threshold:
      threshold = np.percentile(bias_values, bias_percentile_threshold)
    else:
      threshold = bias_value_threshold
    bias_labels = tf.math.greater(bias_values, threshold)
    bias_values_all.append(bias_values)
    bias_labels_all.append(bias_labels)

  example_ids_all = np.concatenate(example_ids_all)
  bias_values_all = np.squeeze(np.concatenate(bias_values_all))
  bias_labels_all = np.squeeze(np.concatenate(bias_labels_all))
  logging.info('# of examples: %s', example_ids_all.shape[0])
  logging.info('# of bias labels: %s', bias_labels_all.shape[0])
  logging.info('# of non-zero bias labels: %s',
               tf.math.count_nonzero(bias_labels_all).numpy())

  if save_table:
    df = pd.DataFrame({
        'example_id': example_ids_all,
        'bias': bias_values_all,
        'bias_label': bias_labels_all
    })
    df.to_csv(
        os.path.join(save_dir, 'bias_table.csv'),
        index=False)

  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(example_ids_all, dtype=tf.string),
      values=tf.convert_to_tensor(bias_labels_all, dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def get_example_id_to_predictions_table(
    dataloader: data.Dataloader,
    trained_models: List[tf.keras.Model],
    has_bias: bool,
    split: Optional[str] = 'train',
    save_dir: Optional[str] = None,
    save_table: Optional[bool] = True) -> pd.DataFrame:
  """Generates a lookup table mapping example ID to bias label.

  Args:
    dataloader: Dataclass object containing training and validation data.
    trained_models: List of trained models.
    has_bias: Do the trained models have a bias prediction head
    split: Which split of the dataset to use ('train'/'val'/'test')
    save_dir: Directory in which predictions table will be saved as CSV.
    save_table: Boolean for whether or not to save table.

  Returns:
    A pandas dataframe mapping example ID to all label and bias predictions.
  """
  table_name = 'predictions_table'
  ds = dataloader.train_ds
  if split != 'train':
    ds = dataloader.eval_ds[split]
    table_name += '_' + split
  labels = list(
      ds.map(
          lambda feats, label, example_id: label).as_numpy_iterator())
  labels = np.concatenate(labels)
  predictions_all = []
  if has_bias:
    bias_predictions_all = []
  for idx, model in enumerate(trained_models):
    model = trained_models[idx]
    predictions = model.predict(ds)
    predictions_all.append(predictions['main'][..., 1])
    if has_bias:
      bias_predictions_all.append(predictions['bias'][..., 1])
  example_ids = list(ds.map(
      lambda feats, label, example_id: example_id).as_numpy_iterator())
  example_ids = np.concatenate(example_ids)
  predictions_all = np.stack(predictions_all)
  if has_bias:
    bias_predictions_all = np.stack(bias_predictions_all)

  logging.info('# of examples in prediction table is: %s', example_ids.shape[0])

  dict_values = {'example_id': example_ids}
  for i in range(predictions_all.shape[0]):
    dict_values[f'predictions_label_{i}'] = predictions_all[i]
    if has_bias:
      dict_values[f'predictions_bias_{i}'] = bias_predictions_all[i]
  df = pd.DataFrame(dict_values)
  if save_table:
    df.to_csv(os.path.join(save_dir, table_name + '.csv'), index=False)
  return df


# Helper functions to process hash tables


def filter_ids_fn(hash_table, value=1):
  """Filter dataset based on whether ids take a certain value in hash table."""
  def filter_fn(feats, label, example_ids):
    del feats, label
    return hash_table.lookup(example_ids) == value
  return filter_fn


