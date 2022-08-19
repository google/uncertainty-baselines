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

r"""Training pipeline for a two-headed output model, where one head is for bias.

Includes the model definition, which implements two-headed output using
custom losses and allows for any base model. Also provides training pipeline,
starting from compiling and initializing the model, fitting on training data,
and evaluating on provided eval datasets.
"""

from typing import Dict, List, Optional

from absl import logging
import tensorflow as tf

import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin


class TwoHeadedOutputModel(tf.keras.Model):
  """Defines a two-headed output model."""

  def __init__(self, model: tf.keras.Model, train_bias: bool, name: str):
    super(TwoHeadedOutputModel, self).__init__(name=name)
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
      },
      weighted_metrics=[])
  return model


def evaluate_model(model: tf.keras.Model,
                   output_dir: str,
                   eval_ds: Dict[str, tf.data.Dataset]):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    output_dir: Directory path to write model checkpoints.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
  """
  best_latest_checkpoint = tf.train.latest_checkpoint(output_dir)
  load_status = model.load_weights(best_latest_checkpoint)
  load_status.assert_consumed()
  for ds_name in eval_ds.keys():
    result = model.evaluate(
        eval_ds[ds_name], return_dict=True)
    logging.info('Evaluation Dataset Name: %s', ds_name)
    logging.info('Main Acc: %f', result['main_acc'])
    logging.info('Main AUC: %f', result['main_auc'])
    logging.info('Bias Acc: %f', result['bias_acc'])
    logging.info('Bias Acc: %f', result['bias_auc'])


def init_model(
    model_params: models.ModelTrainingParameters,
    experiment_name: str,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes an TwoHeadedOutputModel with a base model.

  Args:
    model_params: Dataclass object containing model and training parameters.
    experiment_name: String describing experiment to use model name.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Initialized TwoHeadedOutputModel model.
  """
  model_class = models.get_model(model_params.model_name)
  base_model = model_class(model_params=model_params)

  two_head_model = TwoHeadedOutputModel(
      model=base_model,
      train_bias=model_params.train_bias,
      name=experiment_name)

  if model_params.train_bias and example_id_to_bias_table:
    two_head_model.update_id_to_bias_table(example_id_to_bias_table)

  two_head_model = compile_model(two_head_model, model_params.learning_rate)
  return two_head_model


def run_train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_params: models.ModelTrainingParameters,
    experiment_name: str,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Initializes and trains model on given training and validation data.

  Args:
    train_ds: Training dataset.
    val_ds: Evaluation dataset.
    model_params: Dataclass object containing model and training parameters.
    experiment_name: String to describe model being trained.
    callbacks: Keras Callbacks, like saving checkpoints or early stopping.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    Trained model.
  """
  two_head_model = init_model(
      model_params=model_params,
      experiment_name=experiment_name,
      example_id_to_bias_table=example_id_to_bias_table
  )
  two_head_model.fit(
      train_ds,
      validation_data=val_ds,
      epochs=model_params.num_epochs,
      callbacks=callbacks)
  return two_head_model


def run_ensemble(
    train_idx_combos: List[List[int]],
    train_splits: tf.data.Dataset,
    val_splits: tf.data.Dataset,
    model_params: models.ModelTrainingParameters,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None,
    eval_ds: Optional[Dict[str, tf.data.Dataset]] = None
) -> List[tf.keras.Model]:
  """Trains an ensemble of models and optionally gets their average predictions.

  Args:
    train_idx_combos: List of indices of data splits to include.
    train_splits: Training data splits.
    val_splits: Validation data splits.
    model_params: Dataclass object containing model and training parameters.
    callbacks: Keras Callbacks, like saving checkpoints or early stopping.
    example_id_to_bias_table: Hash table mapping example ID to bias label.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset. If
      provided, gets average predictions on each dataset using ensemble.
      If None, does nothing.

  Returns:
    List of trained models and, optionally, predictions.
  """
  ensemble = []
  for combo in train_idx_combos:
    combo_name = '_'.join(map(str, combo))
    combo_train = data.gather_data_splits(combo, train_splits)
    combo_val = data.gather_data_splits(combo, val_splits)
    combo_model = run_train(
        combo_train,
        combo_val,
        model_params=model_params,
        experiment_name=combo_name,
        callbacks=callbacks,
        example_id_to_bias_table=example_id_to_bias_table)
    ensemble.append(combo_model)

  if eval_ds and example_id_to_bias_table:
    # Calculates average predictions using ensemble and uses them in evaluation.
    for ds_name in eval_ds.keys():
      test_examples = eval_ds[ds_name]
      y_pred_main = []
      y_pred_bias = []
      for model in ensemble:
        ensemble_prob_samples = model.predict(test_examples)
        y_pred_main.append(ensemble_prob_samples['main'])
        y_pred_bias.append(ensemble_prob_samples['bias'])
      y_pred_main = tf.reduce_mean(y_pred_main, axis=0)
      y_pred_bias = tf.reduce_mean(y_pred_bias, axis=0)
      y_true_main = list(test_examples.map(
          lambda feats, label, example_id: label).as_numpy_iterator())
      y_true_main = tf.concat(y_true_main, axis=0)
      y_true_main = tf.convert_to_tensor(y_true_main, dtype=tf.int64)
      y_true_main = tf.one_hot(y_true_main, depth=2)
      example_ids = list(test_examples.map(
          lambda feats, label, example_id: example_id).as_numpy_iterator())
      example_ids = tf.concat(example_ids, axis=0)
      example_ids = tf.convert_to_tensor(example_ids, dtype=tf.string)
      y_true_bias = example_id_to_bias_table.lookup(example_ids)
      y_true_bias = tf.one_hot(y_true_bias, depth=2)
      for m in ensemble[0].metrics:
        m.reset_state()
      ensemble[0].compiled_metrics.update_state({
          'main': y_true_main,
          'bias': y_true_bias
      }, {
          'main': y_pred_main,
          'bias': y_pred_bias
      })
      result = {m.name: m.result() for m in ensemble[0].metrics}
      logging.info('Evaluation Dataset Name: %s', ds_name)
      logging.info('Main Acc: %f', result['main_acc'])
      logging.info('Main AUC: %f', result['main_auc'])
      # TODO(jihyeonlee): Bias labels are not calculated for other evaluation
      # datasets beyond validation, e.g. 'test' or 'test2' for Cardiotox.
      # Provide way to save the predictions themselves.
      logging.info('Bias Acc: %f', result['bias_acc'])
      logging.info('Bias AUC: %f', result['bias_auc'])

  return ensemble
