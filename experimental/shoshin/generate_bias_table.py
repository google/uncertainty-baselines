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

Usage:
# pylint: disable=line-too-long

  ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/generate_bias_table.py \
      --adhoc_import_modules=uncertainty_baselines \
      -- \
      --xm_runlocal \
      --logtostderr \
      --dataset_name=cardiotoxicity \
      --model_name=mlp \
      --output_dir='/tmp/cardiotox/round_0' \
      --bias_percentile_threshold=0.1

# pylint: enable=line-too-long

Note: In output_dir, models trained on different splits of data must already
exist and be present in directory.
"""

import os

from absl import app
from absl import flags
import data  # local file import from experimental.shoshin
import generate_bias_table_lib  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin


FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_name', '', 'Name of registered TF dataset to use.')
flags.DEFINE_string('model_name', '', 'Name of registered model to use.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes for main task.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
flags.DEFINE_integer(
    'num_splits', 5, 'Number of shards into which train and '
    'val will be split to train models used in bias label '
    'generation. Use a number that can divide 100 easily since we use '
    'TFDS functionality to split the dataset by percentage.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_list('hidden_sizes', '1024,512,128',
                  'Number and sizes of hidden layers for MLP model.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('bias_percentile_threshold', 0.2, 'Threshold to generate '
                   'bias labels, using the top percentile of bias values. '
                   'Uses percentile by default or --bias_value_threshold '
                   'if it is specified.', lower_bound=0., upper_bound=1.)
flags.DEFINE_float('bias_value_threshold', None, 'Threshold to generate bias '
                   'labels, using the calculated bias value. If value is above '
                   'the threshold, the bias label will be 1. Else, the bias '
                   'label will be 0. Uses --bias_threshold_percentile if '
                   'this flag is not specified.', lower_bound=0.,
                   upper_bound=1.)


def main(_) -> None:

  model_params = models.ModelTrainingParameters(
      model_name=FLAGS.model_name,
      train_bias=False,
      num_classes=FLAGS.num_classes,
      num_epochs=FLAGS.num_epochs,
      learning_rate=FLAGS.learning_rate,
      hidden_sizes=[int(size) for size in FLAGS.hidden_sizes]
    )

  dataset_builder = data.get_dataset(FLAGS.dataset_name)

  dataloader = dataset_builder(FLAGS.num_splits, FLAGS.batch_size)
  combos_dir = os.path.join(FLAGS.output_dir,
                            generate_bias_table_lib.COMBOS_SUBDIR)
  trained_models = generate_bias_table_lib.load_trained_models(
      combos_dir, model_params)
  _ = generate_bias_table_lib.get_example_id_to_bias_label_table(
      dataloader=dataloader,
      combos_dir=combos_dir,
      trained_models=trained_models,
      num_splits=FLAGS.num_splits,
      bias_percentile_threshold=FLAGS.bias_percentile_threshold,
      bias_value_threshold=FLAGS.bias_value_threshold,
      save_dir=FLAGS.output_dir,
      save_table=True)


if __name__ == '__main__':
  app.run(main)
