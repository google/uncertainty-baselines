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
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf


def compute_ids_to_sample(
    sampling_score: str,
    predictions_df: pd.DataFrame,
    num_samples: int,) -> List[str]:
  """Compute ids to actively sample new labels for.

  Args:
    sampling_score: Which score to use for sampling. Currently supported
      options are 'ensemble_uncertainty', 'ensemble_margin', 'bias'
    predictions_df: Dataframe with columns `example_ids` and
      `predictions_label_{i}` `predictions_bias_{i}` for i in range(k)
    num_samples: Number of samples

  Returns:
    List of example ids to sample based on sampling score
  """
  prediction_label_cols = filter(lambda x: 'label' in x, predictions_df.columns)
  prediction_bias_cols = filter(lambda x: 'bias' in x, predictions_df.columns)
  if sampling_score == 'ensemble_uncertainty':
    sample_avg = predictions_df[prediction_label_cols].mean(axis=1).to_numpy()
    uncertainty = np.minimum(sample_avg, 1-sample_avg)
    predictions_df['sampling_score'] = uncertainty
  elif sampling_score == 'bias':
    sample_avg = predictions_df[prediction_bias_cols].mean(axis=1).to_numpy()
    predictions_df['sampling_score'] = sample_avg
  predictions_df = predictions_df.sort_values(
      by='sampling_score', ascending=False)
  return predictions_df.head(num_samples)['example_id'].to_numpy()


def sample_and_split_ids(
    ids_train: List[str],
    predictions_df: pd.DataFrame,
    sampling_score: str,
    num_samples_per_round: int,
    num_splits: int,
    save_dir: str,
    save_ids: bool,
    ) -> List[pd.DataFrame]:
  """Computes ids to sample for next round and generates new training splits.

  Args:
    ids_train: ids of examples used for training so far
    predictions_df: A dataframe containing the predictions of the two-head
      models for all the training samples.
    sampling_score: The score used to rank candidates for active learning.
    num_samples_per_round: Number of new samples to add in each round of
      active learning.
    num_splits: Number of splits to generate after active sampling.
    save_dir: The director where the splits are to be saved
    save_ids: A boolean indicating whether to save the ids
  Returns:
    A list of pandas dataframes, each containing a list of example ids to be
    included in a split for the next round of training.
  """
  predictions_df = predictions_df[~predictions_df['example_id'].isin(ids_train)]
  ids_to_sample = compute_ids_to_sample(
      sampling_score, predictions_df,
      num_samples_per_round)
  ids_to_sample = np.concatenate([ids_to_sample, ids_train], axis=0)
  tf.io.gfile.makedirs(save_dir)

  # Randomly permute and split set of ids to sample
  n_sample = ids_to_sample.size
  order = np.random.permutation(n_sample)
  split_idx = 0
  num_data_per_split = int(n_sample / num_splits)
  split_dfs = []
  for i in range(num_splits):
    ids_i = ids_to_sample[order[split_idx:min(split_idx + num_data_per_split,
                                              n_sample - 1)]]
    split_idx += ids_i.size
    df = pd.DataFrame({'example_id': ids_i})
    split_dfs.append(df)
    if save_ids:
      df.to_csv(
          os.path.join(save_dir, f'ids_{i}.csv'),
          index=False)
  return split_dfs


def convert_ids_to_table(
    ids_dir: str,) -> List[tf.lookup.StaticHashTable]:
  """Gets static hash table representing ids in each file in ids_dir."""
  ids_tables = []

  # ids_dir is populated by the sample_and_split_ids function above
  for ids_file in tf.io.gfile.listdir(ids_dir):
    ids_i = pd.read_csv(os.path.join(ids_dir, ids_file))['example_id']
    ids_i = np.array([eval(x).decode('UTF-8') for x in ids_i.to_list()])  #  pylint:disable=eval-used
    keys = tf.convert_to_tensor(ids_i, dtype=tf.string)
    values = tf.ones(shape=keys.shape, dtype=tf.int64)
    init = tf.lookup.KeyValueTensorInitializer(
        keys=keys,
        values=values,
        key_dtype=tf.string,
        value_dtype=tf.int64)
    ids_tables.append(tf.lookup.StaticHashTable(init, default_value=0))
  return ids_tables
