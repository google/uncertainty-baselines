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

r"""Binary executable for generating bias label table.

This file serves as a binary to calculate bias values and create a lookup table
that maps from example ID to bias label.


# pylint: enable=line-too-long

Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

import os

from absl import app
from absl import flags
from ml_collections import config_flags
import data  # local file import from experimental.shoshin
import generate_bias_table_lib  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import sampling_policies  # local file import from experimental.shoshin
import train_tf_lib  # local file import from experimental.shoshin
from configs import base_config  # local file import from experimental.shoshin


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')


def main(_) -> None:

  config = FLAGS.config
  base_config.check_flags(config)
  combos_dir = os.path.join(config.output_dir,
                            generate_bias_table_lib.COMBOS_SUBDIR)
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
  if config.generate_bias_table:
    # Loads data.
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

    # Selects training epochs to compute introspection signals from.
    ckpt_epochs = config.eval.signal_ckpt_epochs
    if not ckpt_epochs:
      # If `signal_ckpt_epochs` is not provided via eval config, compute the
      # list of epochs number to load checkpoint from based on
      # `config.eval.num_signal_ckpts`. If `num_signal_ckpts=0`, then only the
      # latest epoch will be loaded.
      ckpt_epochs = generate_bias_table_lib.compute_signal_epochs(
          config.eval.num_signal_ckpts,
          num_total_epochs=config.training.num_epochs)

    # Computes introspection signal for every checkpoint epoch.
    for ckpt_epoch in ckpt_epochs:
      # Loads model.
      trained_models = train_tf_lib.load_trained_models(
          combos_dir, model_params, ckpt_epoch=ckpt_epoch)

      # Generates table.
      _ = generate_bias_table_lib.get_example_id_to_bias_label_table(
          dataloader=dataloader,
          combos_dir=combos_dir,
          trained_models=trained_models,
          num_splits=config.data.num_splits,
          bias_percentile_threshold=config.bias_percentile_threshold,
          tracin_percentile_threshold=config.tracin_percentile_threshold,
          bias_value_threshold=config.bias_value_threshold,
          tracin_value_threshold=config.tracin_value_threshold,
          save_dir=config.output_dir,
          ckpt_epoch=ckpt_epoch,
          save_table=True)
  else:
    # Generates prediction table for all splits.
    dataloader = dataset_builder(
        config.data.num_splits, 1, config.data.subgroup_ids,
        config.data.subgroup_proportions)
    dataloader = data.apply_batch(dataloader, config.data.batch_size)
    model_params.num_subgroups = dataloader.num_subgroups

    # Loads model. Here we use the best checkpoint for prediction table by
    # setting `ckpt_epoch=-1`.
    trained_models = train_tf_lib.load_trained_models(
        combos_dir, model_params, ckpt_epoch=-1)

    # Generates table.
    _ = generate_bias_table_lib.get_example_id_to_predictions_table(
        dataloader=dataloader,
        trained_models=trained_models,
        has_bias=config.train_bias,
        split='train',
        save_dir=config.save_dir,
        save_table=True)
    for split_name in config.eval_splits:
      _ = generate_bias_table_lib.get_example_id_to_predictions_table(
          dataloader=dataloader,
          trained_models=trained_models,
          has_bias=config.train_bias,
          split=split_name,
          save_dir=config.save_dir,
          save_table=True)

if __name__ == '__main__':
  app.run(main)
