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

from typing import List

import numpy as np
import pandas as pd


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
  return predictions_df.head(num_samples)['example_id'].to_list()
