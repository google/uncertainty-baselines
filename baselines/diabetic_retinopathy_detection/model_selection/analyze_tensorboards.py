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

"""Analyze tensorboards."""
import pandas as pd


def main():
  results_df = pd.read_csv('results.tsv', sep='\t')

  in_domain_metric = 'in_domain_validation/auroc'
  joint_metric = 'joint_validation/balanced_selective_prediction_accuracy_auc'
  in_domain_df = results_df[results_df['tag'] == in_domain_metric]
  joint_df = results_df[results_df['tag'] == joint_metric]

  for df_type, df in zip(['ID', 'OOD'], [in_domain_df, joint_df]):
    max_val_and_run_id = []
    for run in list(set(df['run'])):
      run_df = df[df['run'] == run]

      run_df.reset_index(inplace=True)
      max_idx = run_df['value'].idxmax()
      max_row = run_df.iloc[max_idx]
      # print(max_row)
      max_val_and_run_id.append((max_row.value, max_row.run, max_row.step))

    print(df_type)
    for val, run_id, step in sorted(max_val_and_run_id, reverse=True):
      print(f'{run_id}, {step}, {val}')


if __name__ == '__main__':
  main()
