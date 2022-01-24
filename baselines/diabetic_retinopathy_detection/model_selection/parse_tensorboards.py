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

"""This script details how experimental results can be obtained from a directory of TensorBoard results, each contained in a subdirectory.

This is done by writing to a public TensorBoard.

Steps:
  1. Create the TensorBoard; e.g., `tensorboard dev upload --logdir '.'`
  2. Check the experiment ID, which will be printed from the above command.
  3. Execute this script as `python parse_tensorboards.py {experiment_id}`,
    which writes out the result as `results.tsv`.
  4. Optionally, delete the public TensorBoard with
    `tensorboard dev delete --experiment_id {experiment_id}`
"""
import sys
import pandas as pd
import tensorboard as tb


def main(experiment_id):
  experiment = tb.data.experimental.ExperimentFromDev(experiment_id)
  df = experiment.get_scalars()
  df.to_csv(path_or_buf='results.tsv', sep='\t', index=False)
  print(len(set(df['run'])), len(pd.read_csv('results.tsv', sep='\t')))


if __name__ == '__main__':
  exp_id = sys.argv[1]
  main(exp_id)
