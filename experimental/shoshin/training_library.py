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

r"""TF Training pipeline."""

import os
from typing import Any, Dict, List, Optional

from absl import logging
from ml_collections import config_dict
import numpy as np
import tensorflow as tf
import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin
import read_predictions  # local file import from experimental.shoshin

from google3.learning.deepmind.researchdata import datatables

# Subdirectory for checkpoints.
CHECKPOINTS_SUBDIR = 'checkpoints'


def write_predictions(predictor, data_iterator, writer, in_sample):
  """Writer predictions of predictor on data from data_iterator to writer."""
  for batch in data_iterator:
    predictions = predictor.predict(batch[0])['main'][:, 0]
    for j in range(predictions.size):
      measures = {
          'id': batch[2].numpy()[j].decode('UTF-8'),
          'in_sample': in_sample,
          'prediction': predictions[j].item()
      }
      writer.write(measures)
  return


class IntrospectiveActiveSampling(tf.keras.Model):
  """Defines Introspective Active Sampling method."""

  def __init__(self, model: tf.keras.Model, train_bias: bool, name: str):
    super(IntrospectiveActiveSampling, self).__init__(name=name)
    self.train_bias = train_bias
    if self.train_bias:
      self.id_to_bias_table = None
      self.bias_threshold = 0.

    self.model = model

  def _lookup_bias(self, example_ids):
    values = self.id_to_bias_table.lookup(example_ids)
    return tf.cast(
        tf.greater_equal(
            values,
            tf.constant(self.bias_threshold, shape=(), dtype=values.dtype)),
        tf.int32)

  def call(self, inputs):
    return self.model(inputs)

  def update_id_to_bias_table(self, table, bias_percentile):
    keys = table.index.to_numpy()
    values = np.abs(table.loc[keys, 'prediction_insample'].to_numpy() -
                    table.loc[keys, 'prediction_outsample'].to_numpy())
    self.bias_threshold = 0.
    self.id_to_bias_table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            tf.convert_to_tensor(keys), tf.convert_to_tensor(values)),
        default_value=0)

  def train_step(self, inputs):
    features, labels, example_ids = inputs
    y_true_main = tf.one_hot(labels, depth=2)

    with tf.GradientTape() as tape:
      y_pred = self(features, training=True)

      y_true_bias = None
      if self.train_bias:
        if self.id_to_bias_table is None:
          raise ValueError('id_to_bias_table must not be None.')
        y_true_bias = self._lookup_bias(example_ids)
        y_true_bias = tf.one_hot(y_true_bias, depth=2)
      y_true = {'main': y_true_main, 'bias': y_true_bias}
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
      y_true_bias = self._lookup_bias(example_ids)
      y_true_bias = tf.one_hot(y_true_bias, depth=2)

    y_true = {'main': y_true_main, 'bias': y_true_bias}

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


def compile_model(model: tf.keras.Model, learning_rate: float):
  """Compiles model with optimizer, custom loss functions, and metrics."""
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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


def evaluate_model(model: tf.keras.Model, eval_ds: Dict[str, tf.data.Dataset],
                   output_dir: str):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
    output_dir: String output directory
  """
  checkpoint_dir = os.path.join(output_dir, CHECKPOINTS_SUBDIR)
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
    learning_rate: float,
    experiment_name: str,
    hidden_sizes: Optional[List[int]] = None,
    bias_percentile: Optional[float] = .2,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes an IntrospectiveActiveSampling with a base model.

  Args:
    model_name: String name of model class.
    train_bias: Boolean for whether or not to train bias output head.
    learning_rate: learning rate for model.
    experiment_name: String describing experiment to use model name.
    hidden_sizes: List of integers for sizes of hidden layers if MLP model
      chosen.
    bias_percentile: Bias values above this percentile are treated as having
      bias label 1.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Initialized IntrospectiveActiveSampling model.
  """
  model_class = models.get_model(model_name)
  if model_name == 'mlp':
    hidden_sizes = [int(size) for size in hidden_sizes]
    model_params = models.ModelTrainingParameters(
        model_name=model_name,
        train_bias=train_bias,
        num_classes=2,
        num_epochs=1,
        learning_rate=learning_rate,
        hidden_sizes=hidden_sizes)
    base_model = model_class(model_params)

  introspective_model = IntrospectiveActiveSampling(
      model=base_model, train_bias=train_bias, name=experiment_name)
  if train_bias and (example_id_to_bias_table is not None):
    introspective_model.update_id_to_bias_table(example_id_to_bias_table,
                                                bias_percentile)

  introspective_model = compile_model(introspective_model, learning_rate)
  return introspective_model


def run_train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    experiment_name: str,
    train_config: config_dict.ConfigDict,
    model_config: config_dict.ConfigDict,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes and trains model on given training and validation data.

  Args:
    train_ds: Training dataset.
    val_ds: Evaluation dataset.
    experiment_name: String to name model being trained.
    train_config: Config parameters for training
    model_config: Config parameters for model
    callbacks: Keras Callbacks, like saving checkpoints or early stopping.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Trained model.
  """
  introspective_model = init_model(
      model_name=model_config.model_name,
      train_bias=train_config.train_bias,
      learning_rate=train_config.optimizer.learning_rate,
      experiment_name=experiment_name,
      hidden_sizes=model_config.hidden_sizes,
      example_id_to_bias_table=example_id_to_bias_table)
  introspective_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=train_config.optimizer.num_epochs,
      callbacks=callbacks)
  return introspective_model


def load_data_train_model(
    dataset_name: str,
    train_config: config_dict.ConfigDict,
    model_config: config_dict.ConfigDict,
    index: int,
    writer: datatables.Writer,
) -> Any:
  """Load data and train model.

  Args:
    dataset_name: Dataset to use
    train_config: Config parameters for training
    model_config: Config parameters for model
    index: Index of CV split to use
    writer: Datatables writer to write bias predictions to.

  Returns:
    Trained model.
  """
  dataset_builder = data.get_dataset(dataset_name)
  callbacks = []
  if train_config.save_model_checkpoints:
    save_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            train_config.output_dir, CHECKPOINTS_SUBDIR,
            'epoch-{epoch:02d}-val_auc-{val_main_auc:.2f}.ckpt'),
        monitor='val_main_auc',
        mode='max',
        save_weights_only=True,
        save_best_only=True)
    callbacks.append(save_checkpoint_callback)

  if train_config.early_stopping:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        min_delta=0.001,
        patience=3,
        verbose=1,
        mode='max',
        baseline=None,
        restore_best_weights=True)
    callbacks.append(early_stopping_callback)

  # Trains the model.
  dataloader = dataset_builder(train_config.num_splits, train_config.batch_size)
  train_ds = dataloader.train_splits
  eval_ds = dataloader.train_splits[index]
  train_sets = (
      dataloader.train_splits[:index] + dataloader.train_splits[(index + 1):])
  train_ds = train_sets[0]
  for ds in train_sets[1:]:
    train_ds = train_ds.concatenate(ds)
  if train_config.train_bias:
    df_bias = read_predictions.read_predictions(train_config.bias_id)
  else:
    df_bias = None
  introspective_model = run_train(
      train_ds,
      eval_ds,
      train_config=train_config,
      model_config=model_config,
      experiment_name='main_only',
      callbacks=callbacks,
      example_id_to_bias_table=df_bias)
  evaluate_model(introspective_model, {'val': eval_ds}, train_config.output_dir)
  if writer:
    write_predictions(introspective_model, train_ds, writer, True)
    write_predictions(introspective_model, eval_ds, writer, False)
    writer.close()

  return introspective_model
