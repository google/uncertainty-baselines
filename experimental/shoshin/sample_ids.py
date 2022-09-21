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

r"""Binary executable for generating ids to sample in next round.

This file serves as a binary to compute the ids of samples to be included in
next round of training in an active learning loop.

Usage:
# pylint: disable=line-too-long

  ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/sample_ids.py \
      --adhoc_import_modules=uncertainty_baselines \
      -- \
      --xm_runlocal \
      --logtostderr \
      --dataset_name=cardiotoxicity \
      --config.output_dir=...(directory containing bias and predictions table)\
      --config.ids_dir=...(directory where to save computed ids)

# pylint: enable=line-too-long

Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

import os

from absl import app
from absl import flags
from ml_collections import config_flags
import pandas as pd
import tensorflow as tf
import sampling_policies  # local file import from experimental.shoshin
from configs import base_config  # local file import from experimental.shoshin


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:

  config = FLAGS.config
  base_config.check_flags(config)

  bias_table = pd.read_csv(os.path.join(config.output_dir, 'bias_table.csv'))
  predictions_table = pd.read_csv(os.path.join(config.output_dir,
                                               'predictions_table.csv'))
  tf.io.gfile.makedirs(config.ids_dir)
  _ = sampling_policies.sample_and_split_ids(
      bias_table['example_id'].to_numpy(),
      predictions_table,
      config.active_sampling.sampling_score,
      config.active_sampling.num_samples_per_round,
      config.data.num_splits,
      config.ids_dir,
      True)

if __name__ == '__main__':
  app.run(main)
