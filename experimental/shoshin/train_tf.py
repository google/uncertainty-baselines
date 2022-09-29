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

r"""Binary to run training on a single model once.

You can run just this file to train locally or use xm_launch.py to launch on
XManager.

Usage:
# pylint: disable=line-too-long

Note: config.output_dir can be a CNS path.

To train MLP on Cardiotoxicity Fingerprint dataset locally with two output
heads, one for the main task and one for bias, and even as an ensemble:
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/train_tf.py \
  --adhoc_import_modules=uncertainty_baselines \
    -- \
    --xm_runlocal \
    --alsologtostderr \
    --config=third_party/py/uncertainty_baselines/experimental/shoshin/configs/cardiotoxicity_mlp_config.py \
    --config.output_dir=/tmp/cardiotox/ \
    --config.train_bias=True/False
    --config.path_to_existing_bias_table=...
    --config.round_idx=(which round of active sampling (0 is treated specially)
    --config.ids_dir=(directory containing the example ids for the various splits)

# pylint: enable=line-too-long
"""

import logging as native_logging
import os

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_flags
import tensorflow as tf
import data  # local file import from experimental.shoshin
import generate_bias_table_lib  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import sampling_policies  # local file import from experimental.shoshin
import train_tf_lib  # local file import from experimental.shoshin
from configs import base_config  # local file import from experimental.shoshin


# Subdirectory for checkpoints in FLAGS.output_dir.
CHECKPOINTS_SUBDIR = 'checkpoints'


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('config')
flags.DEFINE_bool('keep_logs', True, 'If True, creates a log file in output '
                  'directory. If False, only logs to console.')
flags.DEFINE_string('ensemble_dir', '', 'If specified, loads the models at '
                    'this directory to consider the ensemble.')


def main(_) -> None:
  config = FLAGS.config
  base_config.check_flags(config)

  dataset_builder = data.get_dataset(config.data.name)
  if FLAGS.keep_logs:
    tf.io.gfile.makedirs(config.output_dir)
    stream = tf.io.gfile.GFile(os.path.join(config.output_dir, 'log'), mode='w')
    stream_handler = native_logging.StreamHandler(stream)
    logging.get_absl_logger().addHandler(stream_handler)

  model_params = models.ModelTrainingParameters(
      model_name=config.model.name,
      train_bias=config.train_bias,
      num_classes=config.data.num_classes,
      num_epochs=config.training.num_epochs,
      optimizer=config.optimizer.type,
      learning_rate=config.optimizer.learning_rate,
      hidden_sizes=config.model.hidden_sizes,
      do_reweighting=config.reweighting.do_reweighting,
      reweighting_lambda=config.reweighting.lambda_value,
      reweighting_signal=config.reweighting.signal
  )
  model_params.train_bias = config.train_bias
  output_dir = config.output_dir

  logging.info('Running Round %d of Training.', config.round_idx)
  if config.round_idx == 0:
    # If initial round of sampling, sample randomly initial_sample_proportion
    dataloader = dataset_builder(config.data.num_splits,
                                 config.data.initial_sample_proportion,
                                 config.data.subgroup_ids,
                                 config.data.subgroup_proportions)
  else:
    # If latter round, keep track of split generated in last round of active
    #   sampling
    dataloader = dataset_builder(config.data.num_splits,
                                 1,
                                 (), ())
    # Filter each split to only have examples from example_ids_table
    dataloader.train_splits = [
        dataloader.train_ds.filter(
            generate_bias_table_lib.filter_ids_fn(ids_tab)) for
        ids_tab in sampling_policies.convert_ids_to_table(config.ids_dir)]

  # Apply batching (must apply batching only after filtering)
  dataloader = data.apply_batch(dataloader, config.data.batch_size)
  tf.io.gfile.makedirs(output_dir)
  example_id_to_bias_table = None

  if config.train_bias or config.reweighting.do_reweighting:
    # Bias head will be trained as well, so gets bias labels.
    if config.path_to_existing_bias_table:
      example_id_to_bias_table = generate_bias_table_lib.load_existing_bias_table(
          config.path_to_existing_bias_table)
    else:
      logging.info(
          'Error: Bias table not found')
      return
  # Training a single model on a combination of data splits.
  included_splits_idx = [int(i) for i in config.data.included_splits_idx]
  train_ds = data.gather_data_splits(included_splits_idx,
                                     dataloader.train_splits)
  val_ds = data.gather_data_splits(included_splits_idx,
                                   dataloader.val_splits)
  dataloader.train_ds = train_ds
  dataloader.eval_ds['val'] = val_ds
  experiment_name = 'stage_2' if config.train_bias else 'stage_1'

  _ = train_tf_lib.train_and_evaluate(
      train_as_ensemble=config.train_stage_2_as_ensemble,
      dataloader=dataloader,
      model_params=model_params,
      num_splits=config.data.num_splits,
      ood_ratio=config.data.ood_ratio,
      checkpoint_dir=os.path.join(output_dir, CHECKPOINTS_SUBDIR),
      experiment_name=experiment_name,
      save_model_checkpoints=config.training.save_model_checkpoints,
      early_stopping=config.training.early_stopping,
      ensemble_dir=FLAGS.ensemble_dir,
      example_id_to_bias_table=example_id_to_bias_table)


if __name__ == '__main__':
  app.run(main)
