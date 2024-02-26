# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

r"""Library for evaluating active sampling.
"""

import os
from typing import Mapping

import numpy as np
import pandas as pd
import tensorflow as tf
import data  # local file import from experimental.shoshin


def merge_subgroup_labels(
    ds: tf.data.Dataset,
    table: pd.DataFrame,
    batch_size: int,
):
  """Merge table with subroup labels from ds."""
  ids = np.concatenate(list(
      ds.map(lambda example: example['example_id']).batch(
          batch_size).as_numpy_iterator())).tolist()
  ids = list(map(lambda x: x.decode('UTF-8'), ids))
  subgroup_labels = np.concatenate(list(
      ds.map(lambda example: example['subgroup_label']).batch(
          batch_size).as_numpy_iterator())).tolist()
  labels = np.concatenate(list(
      ds.map(lambda example: example['label']).batch(
          batch_size).as_numpy_iterator())).tolist()
  df_a = pd.DataFrame({
      'example_id': ids, 'subgroup_label': subgroup_labels,
      'label': labels})
  table = table[table['example_id'].isin(ids)]
  return pd.merge(table, df_a, on=['example_id'])


def _process_table(table: pd.DataFrame, prediction: bool):
  """Modify table to have cleaned up example ids and predictions."""
  table['example_id'] = table['example_id'].map(
      lambda x: eval(x).decode('UTF-8'))  #  pylint:disable=eval-used
  if prediction:
    prediction_label_cols = filter(lambda x: 'label' in x, table.columns)
    prediction_bias_cols = filter(lambda x: 'bias' in x, table.columns)
    table['bias'] = table[prediction_bias_cols].mean(axis=1)
    table['label_prediction'] = table[prediction_label_cols].mean(axis=1)
  return table


def evaluate_active_sampling(
    num_rounds: int,
    output_dir: str,
    dataloader: data.Dataloader,
    batch_size: int,
    num_subgroups: int,
    ) -> pd.DataFrame:
  """Evaluates model for subgroup representation vs number of rounds."""
  round_idx = []
  subgroup_ids = []
  num_samples = []
  prob_representation = []
  for idx in range(num_rounds):
    ds = dataloader.train_ds
    bias_table = pd.read_csv(
        os.path.join(
            os.path.join(output_dir, f'round_{idx}'), 'bias_table.csv'))
    predictions_merge = merge_subgroup_labels(ds, bias_table, batch_size)
    for subgroup_id in range(num_subgroups):
      prob_i = (predictions_merge['subgroup_label']
                == subgroup_id).sum() / len(predictions_merge)
      round_idx.append(idx)
      subgroup_ids.append(subgroup_id)
      num_samples.append(len(predictions_merge))
      prob_representation.append(prob_i)
  return pd.DataFrame({
      'num_samples': num_samples,
      'prob_representation': prob_representation,
      'round_idx': round_idx,
      'subgroup_ids': subgroup_ids,
  })


def evaluate_model(
    round_idx: int,
    output_dir: str,
    dataloader: data.Dataloader,
    batch_size: int,
    ) -> Mapping[str, pd.DataFrame]:
  """Evaluates model for subgroup representation vs number of rounds."""
  bias_table = pd.read_csv(
      os.path.join(
          os.path.join(output_dir, f'round_{round_idx}'), 'bias_table.csv'))
  bias_table = _process_table(bias_table, False)
  predictions_table = pd.read_csv(
      os.path.join(
          os.path.join(output_dir, f'round_{round_idx}'),
          'predictions_table.csv'))
  predictions_table = _process_table(predictions_table, True)
  predictions_merge = {}
  predictions_merge['train_bias'] = merge_subgroup_labels(
      dataloader.train_ds, bias_table, batch_size)
  predictions_merge['train_predictions'] = merge_subgroup_labels(
      dataloader.train_ds, predictions_table, batch_size)
  for (ds_name, ds) in dataloader.eval_ds.items():
    predictions_table = _process_table(pd.read_csv(
        os.path.join(
            os.path.join(output_dir, f'round_{round_idx}'),
            f'predictions_table_{ds_name}.csv')), True)
    predictions_merge[f'{ds_name}_predictions'] = merge_subgroup_labels(
        ds, predictions_table, batch_size)
    predictions_merge[f'{ds_name}_bias'] = merge_subgroup_labels(
        ds, bias_table, batch_size)
  return predictions_merge
