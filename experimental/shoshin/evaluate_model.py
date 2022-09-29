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

r"""Binary executable for evaluating active sampling.

Usage:
# pylint: disable=line-too-long

  ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/evaluate_model.py \
      --adhoc_import_modules=uncertainty_baselines \
      -- \
      --xm_runlocal \
      --config=...\
      --config.output_dir=...
      --logtostderr

# pylint: enable=line-too-long

Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import data  # local file import from experimental.shoshin
import evaluate_model_lib  # local file import from experimental.shoshin
from configs import base_config  # local file import from experimental.shoshin


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:

  config = FLAGS.config
  base_config.check_flags(config)

  dataset_builder = data.get_dataset(config.data.name)
  dataloader = dataset_builder(config.data.num_splits, 1,
                               config.data.subgroup_ids,
                               config.data.subgroup_proportions, False)
  df_eval = evaluate_model_lib.evaluate_active_sampling(
      config.active_sampling.num_samples_per_round, config.output_dir,
      dataloader, config.data.batch_size)
  for (it, row) in df_eval.rows:
    logging.info(
        'Round %d, %d Samples, Subgroup representation %f',
        it, row['num_samples'], row['prob_representation'])

if __name__ == '__main__':
  app.run(main)
