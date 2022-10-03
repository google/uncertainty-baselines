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

import itertools
import os
from typing import Dict, List, Optional

from absl import logging
import numpy as np
import tensorflow as tf
import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin


class TwoHeadedOutputModel(tf.keras.Model):
  """Defines a two-headed output model."""

  def __init__(self,
               model: tf.keras.Model,
               train_bias: bool,
               name: str,
               do_reweighting: Optional[bool] = False,
               reweighting_signal: Optional[str] = 'bias',
               reweighting_lambda: Optional[float] = 0.5,
               error_percentile_threshold: Optional[float] = 0.2):
    super(TwoHeadedOutputModel, self).__init__(name=name)
    self.train_bias = train_bias
    if self.train_bias or do_reweighting:
      self.id_to_bias_table = None

    self.do_reweighting = do_reweighting
    if do_reweighting:
      self.reweighting_signal = reweighting_signal
      self.reweighting_lambda = reweighting_lambda
      if self.reweighting_signal == 'error':
        self.error_percentile_threshold = error_percentile_threshold

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
      if self.train_bias or self.do_reweighting:
        if self.id_to_bias_table is None:
          raise ValueError('id_to_bias_table must not be None.')
        y_true_bias = self.id_to_bias_table.lookup(example_ids)
        y_true_bias_original = y_true_bias
        y_true_bias = tf.one_hot(y_true_bias, depth=2)
      y_true = {
          'main': y_true_main,
          'bias': y_true_bias
      }
      sample_weight = None
      if self.do_reweighting:
        if self.reweighting_signal == 'bias':
          example_labels = y_true_bias_original
        else:  # Use prediction error.
          error = tf.math.subtract(
              tf.ones_like(y_pred), tf.gather_nd(y_pred, y_true_main))
          threshold = np.percentile(error, self.error_percentile_threshold)
          example_labels = tf.math.greater(error, threshold)

        above_threshold_example_multiplex = tf.math.multiply(
            self.reweighting_lambda,
            tf.ones_like(example_labels, dtype=tf.float32))
        below_threshold_example_multiplex = tf.math.multiply(
            1. - self.reweighting_lambda,
            tf.ones_like(example_labels, dtype=tf.float32))
        sample_weight = tf.where(
            tf.math.equal(example_labels, 1),
            above_threshold_example_multiplex,
            below_threshold_example_multiplex)

      total_loss = self.compiled_loss(
          y_true, y_pred, sample_weight=sample_weight)
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


def compile_model(model: tf.keras.Model,
                  model_params: models.ModelTrainingParameters):
  """Compiles model with optimizer, custom loss functions, and metrics."""
  if model_params.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_params.learning_rate)
  else:  # sgd
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=model_params.learning_rate, momentum=0.9)
  model.compile(
      optimizer=optimizer,
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
                   checkpoint_dir: str,
                   eval_ds: Dict[str, tf.data.Dataset]):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    checkpoint_dir: Path to directory where checkpoints are stored.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
  """
  best_latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
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
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None
) -> tf.keras.Model:
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
      name=experiment_name,
      do_reweighting=model_params.do_reweighting,
      reweighting_signal=model_params.reweighting_signal,
      reweighting_lambda=model_params.reweighting_lambda,
      error_percentile_threshold=model_params
      .reweighting_error_percentile_threshold)

  if model_params.train_bias or model_params.do_reweighting:
    if example_id_to_bias_table:
      two_head_model.update_id_to_bias_table(example_id_to_bias_table)

  two_head_model = compile_model(two_head_model, model_params)
  return two_head_model


def create_callbacks(
    checkpoint_dir: str,
    save_model_checkpoints: bool = True,
    early_stopping: bool = True) -> List[tf.keras.callbacks.Callback]:
  """Creates callbacks, such as saving model checkpoints, for training.

  Args:
    checkpoint_dir: Directory where checkpoints will be saved.
    save_model_checkpoints: Boolean for whether or not to save checkpoints.
    early_stopping: Boolean for whether or not to use early stopping during
      training.

  Returns:
    List of callbacks.
  """
  callbacks = []
  if save_model_checkpoints:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            checkpoint_dir,
            'epoch-{epoch:02d}-val_auc-{val_main_auc:.2f}.ckpt'),
        monitor='val_main_auc',
        mode='max',
        save_weights_only=True,
        save_best_only=True)
    callbacks.append(checkpoint_callback)
  if early_stopping:
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
  return callbacks


def run_train(
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    model_params: models.ModelTrainingParameters,
    experiment_name: str,
    callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None
) -> tf.keras.Model:
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


def train_ensemble(
    dataloader: data.Dataloader,
    model_params: models.ModelTrainingParameters,
    num_splits: int,
    ood_ratio: float,
    output_dir: str,
    save_model_checkpoints: bool = True,
    early_stopping: bool = True,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None
) -> List[tf.keras.Model]:
  """Trains an ensemble of models, locally. See xm_launch.py for parallelized.

  Args:
    dataloader: Dataclass object containing training and validation data.
    model_params: Dataclass object containing model and training parameters.
    num_splits: Integer number for total slices of dataset.
    ood_ratio: Float for the ratio of slices that will be considered
      out-of-distribution.
    output_dir: String for directory path where checkpoints will be saved.
    save_model_checkpoints: Boolean for saving checkpoints during training.
    early_stopping: Boolean for early stopping during training.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    List of trained models and, optionally, predictions.
  """
  num_ood_splits = int(num_splits * ood_ratio)
  num_id_splits = num_splits - num_ood_splits
  train_idx_combos = [
      list(c) for c in list(
          itertools.combinations(range(num_splits), num_id_splits))
  ]
  ensemble = []
  for combo in train_idx_combos:
    combo_name = '_'.join(map(str, combo))
    combo_train = data.gather_data_splits(combo, dataloader.train_splits)
    combo_val = data.gather_data_splits(combo, dataloader.val_splits)
    combo_ckpt_dir = os.path.join(output_dir, combo_name, 'checkpoints')
    combo_callbacks = create_callbacks(combo_ckpt_dir, save_model_checkpoints,
                                       early_stopping)
    combo_model = run_train(
        combo_train,
        combo_val,
        model_params=model_params,
        experiment_name=combo_name,
        callbacks=combo_callbacks,
        example_id_to_bias_table=example_id_to_bias_table)
    ensemble.append(combo_model)
  return ensemble


def load_trained_models(combos_dir: str,
                        model_params: models.ModelTrainingParameters):
  """Loads models trained on different combinations of data splits."""
  trained_models = []
  for combo_name in tf.io.gfile.listdir(combos_dir):
    combo_model = init_model(
        model_params=model_params,
        experiment_name=combo_name)
    ckpt_dir = os.path.join(combos_dir, combo_name, 'checkpoints')
    best_latest_checkpoint = tf.train.latest_checkpoint(ckpt_dir)
    load_status = combo_model.load_weights(best_latest_checkpoint)
    # Optimizer will not be loaded (https://b.corp.google.com/issues/124099628),
    # so expect only partial load. This is not currently an issue because
    # model is only used for inference.
    load_status.expect_partial()
    load_status.assert_existing_objects_matched()
    trained_models.append(combo_model)
  return trained_models


def eval_ensemble(
    dataloader: data.Dataloader,
    ensemble: List[tf.keras.Model],
    example_id_to_bias_table: tf.lookup.StaticHashTable):
  """Calculates the average predictions of the ensemble for evaluation.

  Args:
    dataloader: Dataclass object containing training and validation data.
    ensemble: List of trained models.
    example_id_to_bias_table: Hash table mapping example ID to bias label.
  """
  for ds_name in dataloader.eval_ds.keys():
    test_examples = dataloader.eval_ds[ds_name]
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


def run_ensemble(
    dataloader: data.Dataloader,
    model_params: models.ModelTrainingParameters,
    num_splits: int,
    ood_ratio: float,
    output_dir: str,
    save_model_checkpoints: bool = True,
    early_stopping: bool = True,
    ensemble_dir: Optional[str] = '',
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None
) -> List[tf.keras.Model]:
  """Trains an ensemble of models and optionally gets their average predictions.

  Args:
    dataloader: Dataclass object containing training and validation data.
    model_params: Dataclass object containing model and training parameters.
    num_splits: Integer number for total slices of dataset.
    ood_ratio: Float for the ratio of slices that will be considered
      out-of-distribution.
    output_dir: String for directory path where checkpoints will be saved.
    save_model_checkpoints: Boolean for saving checkpoints during training.
    early_stopping: Boolean for early stopping during training.
    ensemble_dir: Optional string for a directory that stores trained model
      checkpoints. If specified, will load the models from directory.
    example_id_to_bias_table: Hash table mapping example ID to bias label.

  Returns:
    List of trained models and, optionally, predictions.
  """

  if ensemble_dir:
    ensemble = load_trained_models(ensemble_dir, model_params)
  else:
    ensemble = train_ensemble(dataloader, model_params, num_splits, ood_ratio,
                              output_dir, save_model_checkpoints,
                              early_stopping, example_id_to_bias_table)
  if dataloader.eval_ds and example_id_to_bias_table:
    eval_ensemble(dataloader, ensemble, example_id_to_bias_table)

  return ensemble


def train_and_evaluate(
    train_as_ensemble: bool,
    dataloader: data.Dataloader,
    model_params: models.ModelTrainingParameters,
    num_splits: int,
    ood_ratio: float,
    checkpoint_dir: str,
    experiment_name: str,
    save_model_checkpoints: bool,
    early_stopping: bool,
    ensemble_dir: Optional[str] = '',
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None):
  """Performs the operations of training, optionally ensembling, and evaluation.

  Args:
    train_as_ensemble: Boolean for whether or not to train an ensemble of
      models. Also performs evaluation for ensemble.
    dataloader: Dataclass object containing training and validation data.
    model_params: Dataclass object containing model parameters.
    num_splits: Integer for number of data splits.
    ood_ratio: Float for ratio of splits to consider as out-of-distribution.
    checkpoint_dir: Path to directory where checkpoints will be written.
    experiment_name: String describing experiment.
    save_model_checkpoints: Boolean for saving checkpoints during training.
    early_stopping: Boolean for early stopping during training.
    ensemble_dir: Optional string for a directory that stores trained model
      checkpoints. If specified, will load the models from directory.
    example_id_to_bias_table: Lookup table mapping example ID to bias label.

  Returns:
    Trained Model(s)
  """
  if train_as_ensemble:
    return run_ensemble(
        dataloader=dataloader,
        model_params=model_params,
        num_splits=num_splits,
        ood_ratio=ood_ratio,
        output_dir=checkpoint_dir,
        save_model_checkpoints=save_model_checkpoints,
        early_stopping=early_stopping,
        ensemble_dir=ensemble_dir,
        example_id_to_bias_table=example_id_to_bias_table)
  else:
    callbacks = create_callbacks(checkpoint_dir, save_model_checkpoints,
                                 early_stopping)
    two_head_model = run_train(
        dataloader.train_ds,
        dataloader.eval_ds['val'],
        model_params=model_params,
        experiment_name=experiment_name,
        callbacks=callbacks,
        example_id_to_bias_table=example_id_to_bias_table)
    evaluate_model(two_head_model, checkpoint_dir, dataloader.eval_ds)
    return two_head_model
