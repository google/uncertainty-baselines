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

r"""Binary executable for generating tracin table.

This file serves as a binary to calculate tracin values and create a lookup table
that maps from example ID to tracin label.

Usage:
# pylint: disable=line-too-long

  ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/generate_tracin_table.py \
      --adhoc_import_modules=uncertainty_baselines \
      -- \
      --xm_runlocal \
      --logtostderr \
      --config=third_party/py/uncertainty_baselines/experimental/shoshin/configs/waterbirds_resnet_tracin_config.py

# pylint: enable=line-too-long

Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import data  # local file import from experimental.shoshin
import generate_bias_table_lib  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import sampling_policies  # local file import from experimental.shoshin
from configs import base_config  # local file import from experimental.shoshin


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:

  config = FLAGS.config
  base_config.check_flags(config)
  ckpt_dir = os.path.join(config.output_dir,
                          generate_bias_table_lib.CHECKPOINT_SUBDIR)
  model_params = models.ModelTrainingParameters(
      model_name=config.model.name,
      train_bias=config.train_bias,
      num_classes=config.data.num_classes,
      num_subgroups=0,
      num_epochs=config.training.num_epochs,
      learning_rate=config.optimizer.learning_rate,
      hidden_sizes=config.model.hidden_sizes,
  )

  dataset_builder = data.get_dataset(config.data.name)
  if config.generate_individual_table:
    if config.round_idx == 0:
      dataloader = dataset_builder(config.data.num_splits,
                                   config.data.initial_sample_proportion,
                                   config.data.subgroup_ids,
                                   config.data.subgroup_proportions,)
    else:
      dataloader = dataset_builder(config.data.num_splits, 1,
                                   config.data.subgroup_ids,
                                   config.data.subgroup_proportions,)
       # Filter each split to only have examples from example_ids_table
      dataloader.train_splits = [
          dataloader.train_ds.filter(
              generate_bias_table_lib.filter_ids_fn(ids_tab)) for
          ids_tab in sampling_policies.convert_ids_to_table(config.ids_dir)]
    dataloader = data.apply_batch(dataloader, config.data.batch_size)
    model_params.num_subgroups = dataloader.num_subgroups
    model_checkpoints = generate_bias_table_lib.load_model_checkpoints(
        ckpt_dir, model_params, config.signal.checkpoint_list,
        config.signal.checkpoint_selection, config.signal.checkpoint_number,
        config.signal.checkpoint_name)

    logging.info('%s checkpoints loaded', len(model_checkpoints))
    if config.signal.checkpoint_selection == 'name':
      table_name = config.signal.checkpoint_name
    else:
      table_name = config.signal.checkpoint_selection
    _ = generate_bias_table_lib.get_example_id_to_tracin_value_table(
        dataloader=dataloader,
        model_checkpoints=model_checkpoints,
        included_layers=config.signal.included_layers,
        save_dir=config.save_dir,
        save_table=True,
        table_name=table_name)
  else:
    # TODO(martinstrobel): Combine individual tracinvalues to a mean value
    raise NotImplementedError('Not implemented yet')


if __name__ == '__main__':
  app.run(main)
