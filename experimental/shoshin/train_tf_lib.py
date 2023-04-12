# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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
from typing import Dict, List, Optional, Union

from absl import logging
import numpy as np
import tensorflow as tf
import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin



@tf.keras.saving.register_keras_serializable('two_headed_output_model')
class TwoHeadedOutputModel(tf.keras.Model):
  """Defines a two-headed output model."""

  def __init__(self,
               model: tf.keras.Model,
               num_subgroups: int,
               subgroup_sizes: Dict[str, int],
               train_bias: bool,
               name: str,
               worst_group_label: Optional[int] = 2,
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
    self.num_subgroups = num_subgroups
    if self.num_subgroups > 1:
      self.avg_acc = tf.keras.metrics.Mean(name='avg_acc')
      self.weighted_avg_acc = tf.keras.metrics.Sum(name='weighted_avg_acc')
      self.subgroup_sizes = subgroup_sizes
      self.worst_group_label = worst_group_label

  def get_config(self):
    config = super().get_config()
    config.update({
        'model': self.model,
        'num_subgroups': self.num_subgroups,
        'subgroup_sizes': self.subgroup_sizes,
        'train_bias': self.train_bias,
        'name': self.name,
        'worst_group_label': self.worst_group_label
    })
    return config

  def call(self, inputs):
    return self.model(inputs)

  def update_id_to_bias_table(self, table):
    self.id_to_bias_table = table

  def _compute_average_metrics(
      self, metrics: List[tf.keras.metrics.Metric]
  ) -> Dict[str, tf.keras.metrics.Metric]:
    """Computes metrics as an average or weighted average of all subgroups.

    For the weighted metric, the subgroups are weighed by their proportionality.

    Args:
      metrics: List of metrics to be parsed.

    Returns:
      Dictionary mapping metric name to result.
    """
    accs = []
    total_size = sum(self.subgroup_sizes.values())
    weighted_accs = []
    for m in metrics:
      if 'subgroup' in m.name and 'main' in m.name:
        accs.append(m.result())
        subgroup_label = m.name.split('_')[1]
        weighted_accs.append(
            m.result() * float(self.subgroup_sizes[subgroup_label]) / total_size
        )
    self.avg_acc.reset_state()
    self.avg_acc.update_state(accs)
    self.weighted_avg_acc.reset_state()
    self.weighted_avg_acc.update_state(weighted_accs)
    return {
        self.avg_acc.name: self.avg_acc.result(),
        self.weighted_avg_acc.name: self.weighted_avg_acc.result(),
    }

  def train_step(self, inputs):
    features = inputs['input_feature']
    labels = inputs['label']
    example_ids = inputs['example_id']
    subgroup_labels = inputs['subgroup_label']

    y_true_main = tf.one_hot(labels, depth=2)

    with tf.GradientTape() as tape:
      y_pred = self(features, training=True)

      y_true = {'main': y_true_main}
      if self.train_bias or (self.do_reweighting and
                             self.reweighting_signal == 'bias'):
        if self.id_to_bias_table is None:
          raise ValueError('id_to_bias_table must not be None.')
        y_true_bias = self.id_to_bias_table.lookup(example_ids)
        y_true_bias_original = y_true_bias
        y_true_bias = tf.one_hot(y_true_bias, depth=2)
        y_true['bias'] = y_true_bias

      sample_weight = None
      if self.do_reweighting:
        if self.reweighting_signal == 'bias':
          # Loads bias label from table, which has already been determined by
          # threshold.
          reweighting_labels = y_true_bias_original
        elif self.reweighting_signal == 'error':  # Use prediction error.
          error = tf.math.subtract(
              tf.ones_like(y_pred), tf.gather_nd(y_pred, y_true_main))
          threshold = np.percentile(error, self.error_percentile_threshold)
          reweighting_labels = tf.math.greater(error, threshold)
        else:  # Give weight to worst group only.
          reweighting_labels = tf.math.equal(subgroup_labels,
                                             self.worst_group_label)

        above_threshold_example_multiplex = tf.math.multiply(
            self.reweighting_lambda,
            tf.ones_like(reweighting_labels, dtype=tf.float32))
        below_threshold_example_multiplex = tf.math.multiply(
            1. - self.reweighting_lambda,
            tf.ones_like(reweighting_labels, dtype=tf.float32))
        sample_weight = tf.where(
            reweighting_labels,
            above_threshold_example_multiplex,
            below_threshold_example_multiplex)

      total_loss = self.compiled_loss(
          y_true, y_pred, sample_weight=sample_weight)
      total_loss += sum(self.losses)  # Regularization loss.

    gradients = tape.gradient(total_loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))

    for i in range(self.num_subgroups):
      subgroup_idx = tf.where(tf.math.equal(subgroup_labels, i))
      subgroup_pred = tf.gather(y_pred['main'], subgroup_idx, axis=0)

      subgroup_true = tf.gather(y_true['main'], subgroup_idx, axis=0)
      y_true['_'.join(['subgroup', str(i), 'main'])] = subgroup_true
      y_pred['_'.join(['subgroup', str(i), 'main'])] = subgroup_pred
      if self.train_bias:
        subgroup_pred = tf.gather(y_pred['bias'], subgroup_idx, axis=0)
        subgroup_true = tf.gather(y_true['bias'], subgroup_idx, axis=0)
        y_true['_'.join(['subgroup', str(i), 'bias'])] = subgroup_true
        y_pred['_'.join(['subgroup', str(i), 'bias'])] = subgroup_pred

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    if self.num_subgroups > 1:
      results.update(self._compute_average_metrics(self.metrics))

    return results

  def test_step(self, inputs):
    features = inputs['input_feature']
    labels = inputs['label']
    example_ids = inputs['example_id']
    subgroup_labels = inputs['subgroup_label']
    y_true_main = tf.one_hot(labels, depth=2)
    y_pred = self(features, training=False)
    y_true = {'main': y_true_main}
    if self.train_bias:
      if self.id_to_bias_table is None:
        raise ValueError('id_to_bias_table must not be None.')
      y_true_bias = self.id_to_bias_table.lookup(example_ids)
      y_true['bias'] = tf.one_hot(y_true_bias, depth=2)

    for i in range(self.num_subgroups):
      subgroup_idx = tf.where(tf.math.equal(subgroup_labels, i))
      subgroup_pred = tf.gather(y_pred['main'], subgroup_idx, axis=0)
      subgroup_true = tf.gather(y_true['main'], subgroup_idx, axis=0)
      y_true['_'.join(['subgroup', str(i), 'main'])] = subgroup_true
      y_pred['_'.join(['subgroup', str(i), 'main'])] = subgroup_pred
      if self.train_bias:
        subgroup_pred = tf.gather(y_pred['bias'], subgroup_idx, axis=0)
        subgroup_true = tf.gather(y_true['bias'], subgroup_idx, axis=0)
        y_true['_'.join(['subgroup', str(i), 'bias'])] = subgroup_true
        y_pred['_'.join(['subgroup', str(i), 'bias'])] = subgroup_pred

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    if self.num_subgroups > 1:
      results.update(self._compute_average_metrics(self.metrics))
    return results


def compile_model(
    model: tf.keras.Model, model_params: models.ModelTrainingParameters
):
  """Compiles model with optimizer, custom loss functions, and metrics."""
  if model_params.optimizer == 'adam':
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=model_params.learning_rate
    )
  else:  # sgd
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=model_params.learning_rate, momentum=0.9
    )
  loss = {
      'main': tf.keras.losses.CategoricalCrossentropy(
          from_logits=False, name='main'
      )
  }
  loss_weights = {'main': 1}

  main_metrics = [
      tf.keras.metrics.CategoricalAccuracy(name='acc'),
      tf.keras.metrics.AUC(name='auc'),
  ]
  for i in range(model_params.num_classes):
    main_metrics.append(
        metrics_lib.OneVsRest(
            tf.keras.metrics.AUC(name=f'auroc_{i}_vs_rest'), i
        )
    )
    main_metrics.append(
        metrics_lib.OneVsRest(
            tf.keras.metrics.AUC(name=f'aucpr_{i}_vs_rest', curve='PR'),
            i,
        )
    )
  metrics = {'main': main_metrics}
  if model_params.train_bias:
    metrics['bias'] = [
        tf.keras.metrics.CategoricalAccuracy(name='acc'),
        tf.keras.metrics.AUC(name='auc'),
    ]
    loss['bias'] = tf.keras.losses.CategoricalCrossentropy(
        from_logits=False, name='bias'
    )
    loss_weights['bias'] = 1
  for i in range(model_params.num_subgroups):
    metrics.update(
        {
            '_'.join(['subgroup', str(i), 'main']): [
                tf.keras.metrics.CategoricalAccuracy(name='acc'),
            ]
        }
    )
    if model.train_bias:
      metrics.update(
          {
              '_'.join(['subgroup', str(i), 'bias']): [
                  tf.keras.metrics.CategoricalAccuracy(name='acc'),
                  tf.keras.metrics.AUC(name='auc'),
              ]
          }
      )
  model.compile(
      optimizer=optimizer, loss=loss, loss_weights=loss_weights, metrics=metrics
  )
  return model


def evaluate_model(
    model: tf.keras.Model,
    output_dir: str,
    eval_ds: Dict[str, tf.data.Dataset],
    save_model_checkpoints: bool = False,
    save_best_model: bool = True,
):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    output_dir: Path to directory where model is saved.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
    save_model_checkpoints: Boolean for saving checkpoints during training.
    save_best_model: Boolean for saving best model during training.
  """
  checkpoint_dir = os.path.join(output_dir, 'checkpoints')
  if save_model_checkpoints and tf.io.gfile.listdir(checkpoint_dir):
    best_latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    load_status = model.load_weights(best_latest_checkpoint)
    load_status.assert_consumed()
    for ds_name in eval_ds.keys():
      results = model.evaluate(
          eval_ds[ds_name], return_dict=True)
      logging.info('Evaluation Dataset Name: %s', ds_name)
      logging.info('Main Acc: %f', results['main_acc'])
      logging.info('Main AUC: %f', results['main_auc'])
      if model.train_bias:
        logging.info('Bias Acc: %f', results['bias_acc'])
        logging.info('Bias Acc: %f', results['bias_auc'])
      if model.num_subgroups > 1:
        for i in range(model.num_subgroups):
          logging.info('Subgroup %d Acc: %f', i,
                       results[f'subgroup_{i}_main_acc'])
        logging.info('Average Acc: %f', results['avg_acc'])
        logging.info('Average Acc: %f', results['weighted_avg_acc'])
  if save_best_model:
    model_dir = os.path.join(output_dir, 'model')
    loaded_model = tf.keras.models.load_model(model_dir)
    compiled_model = compile_model(
        loaded_model, loaded_model.model.model_params
    )
    results = compiled_model.evaluate(
        eval_ds['val'],
        return_dict=True,
    )
    logging.info(results)


def init_model(
    model_params: models.ModelTrainingParameters,
    experiment_name: str,
    example_id_to_bias_table: Optional[tf.lookup.StaticHashTable] = None,
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
      num_subgroups=model_params.num_subgroups,
      subgroup_sizes=model_params.subgroup_sizes,
      worst_group_label=model_params.worst_group_label,
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
    output_dir: str,
    save_model_checkpoints: bool = False,
    save_best_model: bool = True,
    early_stopping: bool = True,
    batch_size: Optional[int] = 64,
    num_train_examples: Optional[int] = None,
) -> List[tf.keras.callbacks.Callback]:
  """Creates callbacks, such as saving model checkpoints, for training.

  Args:
    output_dir: Directory where model will be saved.
    save_model_checkpoints: Boolean for whether or not to save checkpoints.
    save_best_model: Boolean for whether or not to save best model.
    early_stopping: Boolean for whether or not to use early stopping during
      training.
    batch_size: Optional integer for batch size.
    num_train_examples: Optional integer for total number of training examples.

  Returns:
    List of callbacks.
  """
  callbacks = []
  if save_model_checkpoints:
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            os.path.join(output_dir, 'checkpoints'),
            'epoch-{epoch:02d}-val_auc-{val_main_auc:.2f}.ckpt'),
        monitor='val_main_auc',
        mode='max',
        save_weights_only=True,
        save_best_only=True)
    callbacks.append(checkpoint_callback)
  if save_best_model:
    model_dir = os.path.join(output_dir, 'model')
    # TODO(jihyeonlee,melfatih): Update to AUPRC.
    model_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(
            model_dir,
            'avg_acc-{val_avg_acc:.2f}'),
        monitor='val_avg_acc',
        mode='max',
        save_weights_only=False,
        save_best_only=True,
        save_traces=True)
    callbacks.append(model_callback)
  if early_stopping:
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_main_auc',
        min_delta=0.001,
        patience=30,
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


def find_epoch_ckpt_path(epoch: int,
                         ckpt_dir: str,
                         metric_name: str = 'val_auc',
                         mode: str = 'highest') -> Union[str, List[str]]:
  r"""Finds the checkpoints for a given epoch.

  This function extracts the checkpoints corresponding to a given epoch under
  `ckpt_dir`. It assumes the checkpoints follows the naming convention:

  `{ckpt_dir}\epoch-{epoch}-{metric_name}-{metric_val}.ckpt`

  If a checkpoint for a given epoch is not found, it will issue a warning and
  return the checkpoint for the nearest epoch instead.

  Args:
    epoch: The epoch to exact checkpoints for.
    ckpt_dir: The directory of checkpoints.
    metric_name: The name of the performance metric.
    mode: The return mode. One of ('highest', 'lowest', 'all'). Here, 'highest'
      / 'lowest' means if there are multiple checkpoint for the required epoch,
      return the checkpoint with the highest / lowest value for the metric.

  Returns:
    Strings for checkpoint directories.
  """
  if mode not in ('highest', 'lowest', 'all'):
    raise ValueError(
        f'mode `{mode}` not supported. Should be one of ("best", "all").')

  # Collects checkpoint names.
  ckpt_names = [
      f_name.split('.ckpt')[0]
      for f_name in tf.io.gfile.listdir(ckpt_dir)
      if '.ckpt.index' in f_name
  ]

  if not ckpt_names:
    raise ValueError(f'No valid checkpoint under the directory {ckpt_dir}.')

  # Extract epoch number and metric values.
  ckpt_epochs = np.array(
      [int(f_name.split('epoch-')[1].split('-')[0]) for f_name in ckpt_names])
  ckpt_metric = np.array([
      float(f_name.split(f'{metric_name}-')[1].split('-')[0])
      for f_name in ckpt_names
  ])

  if epoch not in ckpt_epochs:
    # Uses nearest available epoch in `ckpt_epochs`.
    nearest_epoch_id = np.argmin(np.abs(ckpt_epochs - epoch))
    nearest_epoch = ckpt_epochs[nearest_epoch_id]
    tf.compat.v1.logging.warn(
        'Required epoch (%s) not in list of available epochs `%s`.'
        'Use nearest epoch `%s`', epoch, np.unique(ckpt_epochs), nearest_epoch)
    epoch = nearest_epoch

  make_ckpt_path = lambda name: os.path.join(ckpt_dir, name + '.ckpt')
  if mode == 'highest':
    # Returns the checkpoint with highest metric value.
    ckpt_id = np.argmax(ckpt_metric * (ckpt_epochs == epoch))
    return make_ckpt_path(ckpt_names[ckpt_id])
  elif mode == 'lowest':
    # Returns the checkpoint with lowest metric value.
    ckpt_id = np.argmin(-ckpt_metric * (ckpt_epochs == epoch))
    return make_ckpt_path(ckpt_names[ckpt_id])
  else:
    # Returns all the checkpoints.
    ckpt_ids = np.where(ckpt_epochs == epoch)[0]
    return [make_ckpt_path(ckpt_names[ckpt_id]) for ckpt_id in ckpt_ids]


def load_trained_models(combos_dir: str,
                        model_params: models.ModelTrainingParameters,
                        ckpt_epoch: int = -1):
  """Loads models trained on different combinations of data splits.

  Args:
    combos_dir: Path to the checkpoint trained on different data splits.
    model_params: Model config.
    ckpt_epoch: The epoch to load the checkpoint from. If negative, load the
      latest checkpoint.

  Returns:
    The list of loaded models for different combinations of data splits.
  """
  trained_models = []
  for combo_name in tf.io.gfile.listdir(combos_dir):

    ckpt_dir = os.path.join(combos_dir, combo_name, 'checkpoints')
    if ckpt_epoch < 0:
      # Loads the latest checkpoint.
      checkpoint_path = tf.train.latest_checkpoint(ckpt_dir)
      tf.compat.v1.logging.info(f'Loading best model from `{checkpoint_path}`')
    else:
      # Loads the required checkpoint.
      # By default, select the checkpoint with highest validation AUC.
      checkpoint_path = find_epoch_ckpt_path(
          ckpt_epoch, ckpt_dir, metric_name='val_auc', mode='highest')
      tf.compat.v1.logging.info(
          f'Loading model for checkpoint {ckpt_epoch} from `{checkpoint_path}`')

    combo_model = load_one_checkpoint(checkpoint_path=checkpoint_path,
                                      model_params=model_params,
                                      experiment_name=combo_name)
    trained_models.append(combo_model)
  return trained_models


def load_one_checkpoint(
    checkpoint_path: str,
    model_params: models.ModelTrainingParameters,
    experiment_name: str,
) -> tf.keras.Model:
  """Loads a model checkpoint.

  Args:
    checkpoint_path: Path to checkpoint
    model_params: Model training parameters
    experiment_name: Name of experiment

  Returns:
    A model checkpoint.
  """
  if not tf.io.gfile.exists(checkpoint_path + '.index'):
    raise ValueError(
        f'Required checkpoint file `{checkpoint_path}` not exist.')

  model = init_model(
      model_params=model_params,
      experiment_name=experiment_name)
  load_status = model.load_weights(checkpoint_path)
  # Optimizer will not be loaded (https://b.corp.google.com/issues/124099628),
  # so expect only partial load. This is not currently an issue because
  # model is only used for inference.
  load_status.expect_partial()
  load_status.assert_existing_objects_matched()
  return model


# TODO(martinstrobel): Merge this function with `find_epoch_ckpt_path`.
def generate_checkpoint_list(
    checkpoint_dir: str,
    checkpoint_list: Optional[List[str]] = None,
    checkpoint_selection: Optional[str] = 'first',
    checkpoint_number: Optional[int] = 5,
    checkpoint_name: Optional[str] = '',
) -> Optional[List[str]]:
  """Creates a list of checkpoints to load.

  Args:
    checkpoint_dir: Path to the checkpoint directory.
    checkpoint_list: List of checkpoint names (only used when checkpoint
      selection is list)
    checkpoint_selection: Mode of how to select checkpoints.
    'first': Select the first x checkpoints by epoch
    'last' : Select the las x checkpoints by epoch.
    'spread': Select x checlpoints spread out evenly over all epochs
    'list':  Select the checkpoints provided in checkpoint_list.
    'name': Select the named checkpoint
    checkpoint_number: Number of chekcpoints returned (only used when checkpoint
      selection is first, last, or spread)
    checkpoint_name: Name of a single checkpoint (only used when a checkpoint
      selection is name)
  Returns:
    List of checkpoints to load
  """
  if checkpoint_selection != 'list':
    ckpts_names = tf.io.gfile.listdir(checkpoint_dir)
    ckpts_names = list(filter(lambda x: '.ckpt.index' in x, ckpts_names))
    ckpts_names = list(map(lambda x: x[:-6], ckpts_names))
    epochs = [int(ckpt_name.split('-')[1]) for ckpt_name in ckpts_names]
    sorted_ckpts_names = [ckpts_names[i] for i in np.argsort(epochs)]
    if checkpoint_selection == 'first':
      checkpoint_list = sorted_ckpts_names[:checkpoint_number]
    elif checkpoint_selection == 'last' and checkpoint_number:
      checkpoint_list = sorted_ckpts_names[int(-checkpoint_number):]
    elif checkpoint_selection == 'spread':
      checkpoint_list = [
          sorted_ckpts_names[i]
          for i in range(0, len(sorted_ckpts_names),
                         int(len(sorted_ckpts_names) / checkpoint_number))
      ]
    elif checkpoint_selection == 'all':
      checkpoint_list = sorted_ckpts_names
    elif checkpoint_selection == 'name':
      checkpoint_list = [str(checkpoint_name)]
  return checkpoint_list


# TODO(martinstrobel): Merge this function with `load_trained_models`.
def load_model_checkpoints(checkpoint_dir: str,
                           model_params: models.ModelTrainingParameters,
                           checkpoint_list: Optional[List[str]],
                           checkpoint_selection: Optional[str] = 'first',
                           checkpoint_number: Optional[int] = 5,
                           checkpoint_name: Optional[str] = '',
                           ) -> List[tf.keras.Model]:
  """Loads model checkpoints from a given checkpoint directory.

  Args:
    checkpoint_dir: Path to the checkpoint directory.
    model_params: Model training parameters
    checkpoint_list: List of checkpoint names (only used when checkpoint
      selection is list)
    checkpoint_selection: Mode of how to select checkpoints.
    'first': Select the first x checkpoints by epoch
    'last' : Select the las x checkpoints by epoch.
    'spread': Select x checlpoints spread out evenly over all epochs
    'list':  Select the checkpoints provided in checkpoint_list.
    checkpoint_number: Number of chekcpoints returned (only used when checkpoint
      selection is first, last, or spread)
    checkpoint_name: Name of a single checkpoint (only used when a checkpoint
      selection is name)

  Returns:
    A list of model checkpoints.
  """
  checkpoint_list = generate_checkpoint_list(checkpoint_dir, checkpoint_list,
                                             checkpoint_selection,
                                             checkpoint_number, checkpoint_name)
  checkpoints = []
  for checkpoint in checkpoint_list:
    ckpt_path = os.path.join(checkpoint_dir, checkpoint)
    ckpt = load_one_checkpoint(checkpoint_path=ckpt_path,
                               model_params=model_params,
                               experiment_name='')
    checkpoints.append(ckpt)
  return checkpoints


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
      ensemble_prob_samples = model.predict(
          test_examples.map(lambda x: x['input_feature']))
      y_pred_main.append(ensemble_prob_samples['main'])
      y_pred_bias.append(ensemble_prob_samples['bias'])
    y_pred_main = tf.reduce_mean(y_pred_main, axis=0)
    y_pred_bias = tf.reduce_mean(y_pred_bias, axis=0)
    y_true_main = list(test_examples.map(
        lambda x: x['label']).as_numpy_iterator())
    y_true_main = tf.concat(y_true_main, axis=0)
    y_true_main = tf.convert_to_tensor(y_true_main, dtype=tf.int64)
    y_true_main = tf.one_hot(y_true_main, depth=2)
    example_ids = list(test_examples.map(
        lambda x: x['example_id']).as_numpy_iterator())
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
    output_dir: str,
    experiment_name: str,
    save_model_checkpoints: bool,
    save_best_model: bool,
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
    output_dir: Path to directory where model will be saved.
    experiment_name: String describing experiment.
    save_model_checkpoints: Boolean for saving checkpoints during training.
    save_best_model: Boolean for saving best model during training.
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
        output_dir=output_dir,
        save_model_checkpoints=save_model_checkpoints,
        early_stopping=early_stopping,
        ensemble_dir=ensemble_dir,
        example_id_to_bias_table=example_id_to_bias_table)
  else:
    callbacks = create_callbacks(
        output_dir,
        save_model_checkpoints,
        save_best_model,
        early_stopping,
        model_params.batch_size,
        dataloader.num_train_examples)

    two_head_model = run_train(
        dataloader.train_ds,
        dataloader.eval_ds['val'],
        model_params=model_params,
        experiment_name=experiment_name,
        callbacks=callbacks,
        example_id_to_bias_table=example_id_to_bias_table)
    evaluate_model(two_head_model, output_dir, dataloader.eval_ds,
                   save_model_checkpoints, save_best_model)
    return two_head_model
