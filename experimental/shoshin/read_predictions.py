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

"""Read predictions from experiment and compute bias label."""

import pandas as pd

from google3.learning.deepmind.researchdata import datatables


def read_predictions(xid):
  """Read predictions from experiment."""
  reader = datatables.Reader(f'/datatable/xid/{xid}/predictions')
  df = reader.read()
  df = pd.DataFrame(df, columns=df.keys())
  df_agg = df.groupby(by=['id', 'in_sample']).agg('mean').reset_index()
  df_insample = df_agg.loc[df_agg['in_sample'], :]
  df_outsample = df_agg.loc[~df_agg['in_sample'], :]
  df_result = pd.merge(df_insample, df_outsample, on=['id'])
  df_result = df_result.loc[:, ['id', 'prediction_x', 'prediction_y']]
  df_result = df_result.rename(columns={
      'prediction_x': 'prediction_insample',
      'prediction_y': 'prediction_outsample'
  })
  df_result = df_result.set_index('id')
  return df_result
