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

r"""Model definition and binary for running Introspective Active Sampling.

Usage:
# pylint: disable=line-too-long

To train MLP on Cardiotoxicity Fingerprint dataset locally on only the main
classification task (no bias):
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/introspective_active_learning.py \
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
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/introspective_active_learning.py \
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
import os
from typing import Dict, List, Optional

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import utils  # local file import from experimental.shoshin


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', '', 'Name of registered TF dataset to use.')
flags.DEFINE_string('model_name', '', 'Name of registered model to use.')
# TODO(jihyeonlee): Use num_classes flag across files.
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_integer('num_classes', 2, 'Number of classes for main task.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
# Model parameter flags.
flags.DEFINE_integer(
    'num_splits', 5, 'Number of shards into which train and '
    'val will be split to train models used in bias label '
    'generation. Use a number that can divide 100 easily since we use '
    'TFDS functionality to split the dataset by percentage.')
flags.register_validator('num_splits',
                         lambda value: 100 % value == 0,
                         message='100 must be divisible by --num_splits.')
flags.DEFINE_integer('num_rounds', 3, 'Number of rounds of active sampling '
                     'to conduct. Bias values are calculated for all examples '
                     'at the end of every round.')
flags.DEFINE_float('ood_ratio', 0.4, 'Ratio of splits that will be considered '
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


# Subdirectory for checkpoints in FLAGS.output_dir.
CHECKPOINTS_SUBDIR = 'checkpoints'


class IntrospectiveActiveSampling(tf.keras.Model):
  """Defines Introspective Active Sampling method."""

  def __init__(self, model: tf.keras.Model, train_bias: bool, name: str):
    super(IntrospectiveActiveSampling, self).__init__(name=name)
    self.train_bias = train_bias
    if self.train_bias:
      self.id_to_bias_table = None

    self.model = model

  def call(self, inputs):
    return self.model(inputs)

  def update_id_to_bias_table(self, table):
    self.id_to_bias_table = table

  def train_step(self, inputs):
    features, labels, example_ids = inputs
    y_true_main = tf.one_hot(labels, depth=2)

    with tf.GradientTape() as tape:
      y_pred = self(features, training=True)

      y_true_bias = None
      if self.train_bias:
        if self.id_to_bias_table is None:
          raise ValueError('id_to_bias_table must not be None.')
        y_true_bias = self.id_to_bias_table.lookup(example_ids)
        y_true_bias = tf.one_hot(y_true_bias, depth=2)
      y_true = {
          'main': y_true_main,
          'bias': y_true_bias
      }
      total_loss = self.compiled_loss(y_true, y_pred)
      total_loss += sum(self.losses)  # Regularization loss

    gradients = tape.gradient(total_loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    return results

  def test_step(self, inputs):
    features, labels, example_ids = inputs
    y_true_main = tf.one_hot(labels, depth=2)
    y_pred = self(features, training=False)

    y_true_bias = None
    if self.train_bias:
      if self.id_to_bias_table is None:
        raise ValueError('id_to_bias_table must not be None.')
      y_true_bias = self.id_to_bias_table.lookup(example_ids)
      y_true_bias = tf.one_hot(y_true_bias, depth=2)

    y_true = {
        'main': y_true_main,
        'bias': y_true_bias
    }

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    return results


def compute_loss_main(y_true_main: tf.Tensor, y_pred_main: tf.Tensor):
  """Defines loss function for main classification task."""
  loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  return loss_func(y_true_main, y_pred_main)


def compute_loss_bias(y_true_bias: tf.Tensor, y_pred_bias: tf.Tensor):
  """Defines loss function for bias classification task."""
  loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  return loss_func(y_true_bias, y_pred_bias)


def compile_model(model: tf.keras.Model):
  """Compiles model with optimizer, custom loss functions, and metrics."""
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
      loss={
          'main': compute_loss_main,
          'bias': compute_loss_bias
      },
      loss_weights={
          'main': 1,
          'bias': 1 if model.train_bias else 0
      },
      metrics={
          'main': [
              tf.keras.metrics.CategoricalAccuracy(name='acc'),
              tf.keras.metrics.AUC(name='auc')
          ],
          'bias': [
              tf.keras.metrics.CategoricalAccuracy(name='acc'),
              tf.keras.metrics.AUC(name='auc')
          ]
      })
  return model


def evaluate_model(model: tf.keras.Model, eval_ds: Dict[str, tf.data.Dataset]):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
  """
  checkpoint_dir = os.path.join(FLAGS.output_dir, CHECKPOINTS_SUBDIR)
  best_latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  load_status = model.load_weights(best_latest_checkpoint)
  load_status.assert_consumed()
  for ds_name in eval_ds.keys():
    result = model.evaluate(eval_ds[ds_name], return_dict=True)
    logging.info('Evaluation Dataset Name: %s', ds_name)
    logging.info('Main Acc: %f', result['main_acc'])
    logging.info('Main AUC: %f', result['main_auc'])
    logging.info('Bias Acc: %f', result['bias_acc'])
    logging.info('Bias Acc: %f', result['bias_auc'])


def init_model(
    model_name: str,
    train_bias: bool,
    experiment_name: str,
    hidden_sizes: Optional[List[int]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes an IntrospectiveActiveSampling with a base model.

  Args:
    model_name: String name of model class.
    train_bias: Boolean for whether or not to train bias output head.
    experiment_name: String describing experiment to use model name.
    hidden_sizes: List of integers for sizes of hidden layers if MLP
      model chosen.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Initialized IntrospectiveActiveSampling model.
  """
  model_class = models.get_model(model_name)
  if FLAGS.model_name == 'mlp':
    hidden_sizes = [int(size) for size in hidden_sizes]
    base_model = model_class(
        train_bias=train_bias, name=model_name, hidden_sizes=hidden_sizes)
  else:
    base_model = model_class(train_bias=train_bias, name=model_name)

  introspective_model = IntrospectiveActiveSampling(model=base_model,
                                                    train_bias=train_bias,
                                                    name=experiment_name)
  if train_bias and example_id_to_bias_table:
    introspective_model.update_id_to_bias_table(example_id_to_bias_table)

  introspective_model = compile_model(introspective_model)
  return introspective_model


def run_train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    train_bias: bool,
    experiment_name: str,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes and trains model on given training and validation data.

  Args:
    train_ds: Training dataset.
    val_ds: Evaluation dataset.
    train_bias: Boolean for whether or not to train bias head.
    experiment_name: String to name model being trained.
    callbacks: Keras Callbacks, like saving checkpoints or early stopping.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Trained model.
  """
  introspective_model = init_model(
      model_name=FLAGS.model_name,
      train_bias=train_bias,
      experiment_name=experiment_name,
      hidden_sizes=FLAGS.hidden_sizes,
      example_id_to_bias_table=example_id_to_bias_table
  )
  introspective_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=FLAGS.num_epochs,
      callbacks=callbacks)
  return introspective_model


def main(_) -> None:
  dataset_builder = data.get_dataset(FLAGS.dataset_name)
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
        monitor='val_auc',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True
    )
    callbacks.append(early_stopping_callback)

  if FLAGS.train_main_only:
    # Trains only the main classification task with no bias output head.
    _, _, train_ds, eval_ds = dataset_builder(FLAGS.num_splits,
                                              FLAGS.batch_size)
    introspective_model = run_train(
        train_ds, eval_ds['val'], train_bias=False, experiment_name='main_only',
        callbacks=callbacks)
    evaluate_model(introspective_model, eval_ds)

  # TODO(jihyeonlee): Parallelize training models on different combinations
  # using XM v2 pipelines.
  else:
    num_ood_splits = int(FLAGS.num_splits * FLAGS.ood_ratio)
    num_id_splits = FLAGS.num_splits - num_ood_splits
    train_combos = list(
        itertools.combinations(range(FLAGS.num_splits), num_id_splits))
    for round_idx in range(FLAGS.num_rounds):
      logging.info('Running Round %d of Active Learning.', round_idx)
      trained_models = []
      train_splits, val_splits, _, _ = dataset_builder(
          FLAGS.num_splits, FLAGS.batch_size)
      model_dir = os.path.join(FLAGS.output_dir, f'round_{round_idx}')
      tf.io.gfile.makedirs(model_dir)

      logging.info(
          'Training models on different splits of data to calculate bias...')
      for combo in train_combos:
        combo_name = '_'.join(map(str, combo))
        combo_train = data.gather_data_splits(combo, train_splits)
        combo_val = data.gather_data_splits(combo, val_splits)
        combo_model = run_train(
            combo_train,
            combo_val,
            train_bias=False,
            experiment_name=combo_name)
        trained_models.append(combo_model)

      example_id_to_bias_table = utils.get_example_id_to_bias_label_table(
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
      introspective_model = run_train(
          train_ds,
          eval_ds['val'],
          train_bias=True,
          experiment_name=combo_name,
          callbacks=callbacks,
          example_id_to_bias_table=example_id_to_bias_table)
      evaluate_model(introspective_model, eval_ds)

  # TODO(jihyeonlee): Will add Waterbirds dataloader and ResNet model to support
  # vision modality.

  # TODO(jihyeonlee): Create dataclass to be base class for dataloaders so that
  # expected return/properties are more clear.

if __name__ == '__main__':
  app.run(main)
