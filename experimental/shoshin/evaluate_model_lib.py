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

r"""Library for evaluating active sampling.
"""

import os

import numpy as np
import pandas as pd
import data  # local file import from experimental.shoshin


def evaluate_active_sampling(
    num_rounds: int,
    output_dir: str,
    dataloader: data.Dataloader,
    batch_size: int,
    ) -> pd.DataFrame:
  """Evaluates model for subgroup representation vs number of rounds."""
  num_samples = []
  probabilities = []
  for idx in range(num_rounds):
    ds = dataloader.train_ds
    bias_table = pd.read_csv(
        os.path.join(
            os.path.join(output_dir, f'round_{idx}'), 'bias_table.csv'))
    bias_table['example_id'] = bias_table['example_id'].map(
        lambda x: eval(x).decode('UTF-8'))  #  pylint:disable=eval-used
    ids = np.concatenate(list(
        ds.map(lambda example: example['example_id']).batch(
            batch_size).as_numpy_iterator())).tolist()
    ids = list(map(lambda x: x.decode('UTF-8'), ids))
    subgroup_labels = list(
        ds.map(lambda example: example['subgroup_label']).batch(
            batch_size).as_numpy_iterator())
    subgroup_labels = np.concatenate(subgroup_labels).tolist()
    df_a = pd.DataFrame({'example_id': ids, 'subgroup_label': subgroup_labels})
    bias_table = bias_table[bias_table['example_id'].isin(ids)]
    predictions_merge = pd.merge(bias_table, df_a, on=['example_id'])
    prob_one = (predictions_merge['subgroup_label']
                == 1).sum() / len(predictions_merge)
    num_samples.append(len(predictions_merge))
    probabilities.append(prob_one)
  return pd.DataFrame({
      'num_samples': num_samples,
      'prob_representation': probabilities
  })
