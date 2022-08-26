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

r"""Launch binary to run training.

Usage:
# pylint: disable=line-too-long

To train MLP on Cardiotoxicity Fingerprint dataset locally on only the main
classification task (no bias):
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/launch_tf.py \
  --adhoc_import_modules=uncertainty_baselines \
    -- \
    --xm_runlocal \
    --alsologtostderr \
    --dataset_name=cardiotoxicity \
    --model_name=mlp \
    --num_epochs=10 \
    --output_dir=/tmp/cardiotox/ \  # can be a CNS path
    --train_main_only=True

To train MLP on Cardiotoxicity Fingerprint dataset locally with two output
heads, one for the main task and one for bias:
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/launch_tf.py \
  --adhoc_import_modules=uncertainty_baselines \
    -- \
    --xm_runlocal \
    --alsologtostderr \
    --dataset_name=cardiotoxicity \
    --model_name=mlp \
    --output_dir=/tmp/cardiotox/ \  # can be a CNS path
    --num_epochs=10

# pylint: enable=line-too-long
"""

import itertools
import logging as native_logging
import os

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import data  # local file import from experimental.shoshin
import generate_bias_table_lib  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import train_tf  # local file import from experimental.shoshin


# Subdirectory for checkpoints in FLAGS.output_dir.
CHECKPOINTS_SUBDIR = 'checkpoints'


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', '', 'Name of registered TF dataset to use.')
flags.DEFINE_string('model_name', '', 'Name of registered model to use.')
# TODO(jihyeonlee): Use num_classes flag across files.
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_bool('keep_logs', True, 'If True, creates a log file in output '
                  'directory. If False, only logs to console.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes for main task.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
# Model parameter flags.
flags.DEFINE_integer(
    'num_splits', 5, 'Number of shards into which train and '
    'val will be split to train models used in bias label '
    'generation. Use a number that can divide 100 easily since we use '
    'TFDS functionality to split the dataset by percentage.')
flags.register_validator(
    'num_splits',
    lambda value: 100 % value == 0,
    message='100 must be divisible by --num_splits.')
flags.DEFINE_list(
    'included_splits_idx',
    '0,1,2,3,4',
    'Indices of the data splits to include in training. '
    'Uses all by default.')
flags.DEFINE_integer(
    'num_rounds', 3, 'Number of rounds of active sampling '
    'to conduct. Bias values are calculated for all examples '
    'at the end of every round.')
flags.DEFINE_float(
    'ood_ratio', 0.4, 'Ratio of splits that will be considered '
    'out-of-distribution from each combination. For example, '
    'when ood_ratio == 0.4 and num_splits == 5, 2 out of 5 '
    'slices of data will be excluded in training (for every '
    'combination used to train a model).')
flags.DEFINE_list('hidden_sizes', '1024,512,128',
                  'Number and sizes of hidden layers for MLP model.')
flags.DEFINE_boolean('train_main_only', False,
                     'If True, trains only main task head, not bias head.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('dropout_rate', 0.2, 'Dropout rate.')
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
flags.DEFINE_boolean('save_bias_table', True,
                     'If True, saves table mapping example ID to bias label.')
flags.DEFINE_boolean('save_model_checkpoints', True,
                     'If True, saves checkpoints with best validation AUC '
                     'for the main task during training.')
flags.DEFINE_boolean('early_stopping', True,
                     'If True, stops training when validation AUC does not '
                     'improve any further after 3 epochs.')
flags.DEFINE_boolean('train_stage_2_as_ensemble', False,
                     'If True, trains stage 2 model (stage 1 is calculating '
                     'bias table; stage 2 is training on main task with a '
                     'bias head) as an ensemble of models. If False, trains a '
                     'single model.')


def main(_) -> None:
  dataset_builder = data.get_dataset(FLAGS.dataset_name)
  if FLAGS.keep_logs:
    tf.io.gfile.makedirs(FLAGS.output_dir)
    stream = tf.io.gfile.GFile(os.path.join(FLAGS.output_dir, 'log'), mode='w')
    stream_handler = native_logging.StreamHandler(stream)
    root_logger = logging.get_absl_logger()
    root_logger.addHandler(stream_handler)

  callbacks = []
  if FLAGS.save_model_checkpoints:
    save_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            FLAGS.output_dir, CHECKPOINTS_SUBDIR,
            'epoch-{epoch:02d}-val_auc-{val_main_auc:.2f}.ckpt'),
        monitor='val_main_auc',
        mode='max',
        save_weights_only=True,
        save_best_only=True)
    callbacks.append(save_checkpoint_callback)

  if FLAGS.early_stopping:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_main_auc',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

  model_params = models.ModelTrainingParameters(
      model_name=FLAGS.model_name,
      train_bias=False,
      num_classes=FLAGS.num_classes,
      num_epochs=FLAGS.num_epochs,
      learning_rate=FLAGS.learning_rate,
      hidden_sizes=[int(size) for size in FLAGS.hidden_sizes]
  )

  if FLAGS.train_main_only:
    # Trains only the main classification task with no bias output head.
    train_splits, val_splits, _, eval_ds = dataset_builder(
        FLAGS.num_splits, FLAGS.batch_size)
    included_splits_idx = [int(i) for i in FLAGS.included_splits_idx]
    train_data = data.gather_data_splits(included_splits_idx,
                                         train_splits)
    val_data = data.gather_data_splits(included_splits_idx, val_splits)
    two_head_model = train_tf.run_train(
        train_data,
        val_data,
        model_params=model_params,
        experiment_name='main_only',
        callbacks=callbacks)
    train_tf.evaluate_model(two_head_model,
                            os.path.join(FLAGS.output_dir, CHECKPOINTS_SUBDIR),
                            eval_ds)

  # TODO(jihyeonlee): Parallelize training models on different combinations
  # using XM v2 pipelines.
  else:
    num_ood_splits = int(FLAGS.num_splits * FLAGS.ood_ratio)
    num_id_splits = FLAGS.num_splits - num_ood_splits
    train_combos = [
        list(c) for c in list(
            itertools.combinations(range(FLAGS.num_splits), num_id_splits))
    ]
    for round_idx in range(FLAGS.num_rounds):
      logging.info('Running Round %d of Active Learning.', round_idx)
      train_splits, val_splits, _, eval_ds = dataset_builder(
          FLAGS.num_splits, FLAGS.batch_size)
      model_dir = os.path.join(FLAGS.output_dir, f'round_{round_idx}')
      tf.io.gfile.makedirs(model_dir)

      logging.info(
          'Training models on different splits of data to calculate bias...')
      model_params.train_bias = False
      trained_models = train_tf.run_ensemble(train_idx_combos=train_combos,
                                             train_splits=train_splits,
                                             val_splits=val_splits,
                                             model_params=model_params)

      example_id_to_bias_table = generate_bias_table_lib.get_example_id_to_bias_label_table(
          train_splits=train_splits,
          val_splits=val_splits,
          combos=train_combos,
          trained_models=trained_models,
          num_splits=FLAGS.num_splits,
          bias_value_threshold=FLAGS.bias_value_threshold,
          bias_percentile_threshold=FLAGS.bias_percentile_threshold,
          save_dir=model_dir,
          save_table=FLAGS.save_bias_table)

      _, _, train_ds, eval_ds = dataset_builder(FLAGS.num_splits,
                                                FLAGS.batch_size)
      model_params.train_bias = True
      if FLAGS.train_stage_2_as_ensemble:
        trained_models = train_tf.run_ensemble(
            train_idx_combos=train_combos,
            train_splits=train_splits,
            val_splits=val_splits,
            model_params=model_params,
            callbacks=callbacks,
            example_id_to_bias_table=example_id_to_bias_table,
            eval_ds=eval_ds)
      else:
        two_head_model = train_tf.run_train(
            train_ds,
            eval_ds['val'],
            model_params=model_params,
            experiment_name=f'round_{round_idx}',
            callbacks=callbacks,
            example_id_to_bias_table=example_id_to_bias_table)
        train_tf.evaluate_model(
            two_head_model, os.path.join(FLAGS.output_dir,
                                         CHECKPOINTS_SUBDIR), eval_ds)

  # TODO(jihyeonlee): Will add Waterbirds dataloader and ResNet model to support
  # vision modality.

  # TODO(jihyeonlee): Create dataclass to be base class for dataloaders so that
  # expected return/properties are more clear.

if __name__ == '__main__':
  app.run(main)
