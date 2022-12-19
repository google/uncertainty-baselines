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

"""Utilities for Introspective Active Sampling.

Library of utilities for the Introspecive Active Sampling method. Includes a
function to generate a table mapping example ID to bias label, which can be
used to train the bias output head.
"""

import os
from typing import Dict, List, Optional, Tuple, Any

from absl import logging
import numpy as np
import pandas as pd
import tensorflow as tf

import data  # local file import from experimental.shoshin


EXAMPLE_ID_KEY = 'example_id'
BIAS_LABEL_KEY = 'bias_label'
TRACIN_LABEL_KEY = 'tracin_label'
TRACIN_SCORE_KEY = 'tracin_score'
PREDICTION_KEY = 'prediction'
# Subdirectory for models trained on splits in FLAGS.output_dir.
COMBOS_SUBDIR = 'combos'
CHECKPOINT_SUBDIR = 'checkpoints/'


def compute_signal_epochs(num_signal_ckpts: int, num_total_epochs: int):
  """Computes the epochs to compute introspection signals."""
  if num_signal_ckpts <= 0:
    # This will inform the `train_tf_lib.load_trained_models` to just use the
    # best checkpoint.
    return [-1]

  # Computes the epochs in log scale.
  log_epochs = np.linspace(0, np.log(num_total_epochs), num=num_signal_ckpts)
  epochs = np.ceil(np.exp(log_epochs)).astype(int)
  epochs = list(np.unique(epochs))
  # Still allows the computation of the best checkpoint.
  epochs.append(-1)
  return epochs


def load_existing_bias_table(
    path_to_table: str,
    signal: Optional[str] = BIAS_LABEL_KEY) -> tf.lookup.StaticHashTable:
  """Loads bias table from file."""
  df = pd.read_csv(path_to_table)
  key_tensor = np.array([eval(x).decode('UTF-8') for    #  pylint:disable=eval-used
                         x in df[EXAMPLE_ID_KEY].to_list()])
  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(
          key_tensor, dtype=tf.string),
      values=tf.convert_to_tensor(
          df[signal].to_numpy(), dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def get_example_id_to_bias_label_table(
    dataloader: data.Dataloader,
    combos_dir: str,
    trained_models: List[tf.keras.Model],
    num_splits: int,
    bias_percentile_threshold: int,
    tracin_percentile_threshold: int,
    bias_value_threshold: Optional[float] = None,
    tracin_value_threshold: Optional[float] = None,
    save_dir: Optional[str] = None,
    ckpt_epoch: int = -1,
    save_table: Optional[bool] = True) -> tf.lookup.StaticHashTable:
  """Generates a lookup table mapping example ID to bias label.

  Args:
    dataloader: Dataclass object containing training and validation data.
    combos_dir: Directory of model checkpoints by the combination of data splits
      used in training.
    trained_models: List of trained models.
    num_splits: Total number of slices that data was split into.
    bias_percentile_threshold: Integer between 0 and 100  representing the
      percentile of bias values to give a label of 1 (and 0 for all others).
      Given a vector V of length N, the q-th percentile of V is the value q/100
      of the way from the minimum to the maximum in a sorted copy of V.
    tracin_percentile_threshold: Integer between 0 and 100  representing the
      percentile of tracin values to give a label of 1 (and 0 for all others).
      Given a vector V of length N, the q-th percentile of V is the value q/100
      of the way from the minimum to the maximum in a sorted copy of V.
    bias_value_threshold: Float representing the bias value threshold, above
      which examples will receive a bias label of 1 (and 0 if below). Use
      percentile by default.
    tracin_value_threshold: Float representing the tracin value threshold, above
      which examples will receive a tracin label of 1 (and 0 if below). Use
      percentile by default.
    save_dir: Directory in which bias table will be saved as CSV.
    ckpt_epoch: The training epoch where the models in `trained_models` are
      loaded from. It only impacts the file name of the bias table.
    save_table: Boolean for whether or not to save table.

  Returns:
    A lookup table mapping example ID to bias label and additional meta data
    (i.e., target label, subgroup label, and introspection signals).
  """
  is_train_all = []
  example_ids_all = []
  target_labels_all = []
  groups_labels_all = []

  bias_labels_all = []
  bias_values_all = []
  vars_values_all = []
  nois_values_all = []
  gap_values_all = []
  error_values_all = []
  id_tracin_values_all = []
  ood_tracin_values_all = []
  tracin_labels_all = []
  for split_idx in range(num_splits):
    # For each split of data,
    # 1. Get the models that included this split (as in-domain training data).
    # 2. Get the models that excluded this split (as out-of-distribution data).
    # 3. Calculate the bias value and, using the threshold, bias label.
    id_predictions_all = []
    ood_predictions_all = []
    id_tracin_values_splits = []
    ood_tracin_values_splits = []

    # Collects target and place labels.
    labels = list(dataloader.train_splits[split_idx].map(
        lambda example: example['label']).as_numpy_iterator())
    labels += list(dataloader.val_splits[split_idx].map(
        lambda example: example['label']).as_numpy_iterator())
    labels = np.concatenate(labels)
    target_labels_all.append(labels)

    group_labels = list(dataloader.train_splits[split_idx].map(
        lambda example: example['subgroup_label']).as_numpy_iterator())
    group_labels += list(dataloader.val_splits[split_idx].map(
        lambda example: example['subgroup_label']).as_numpy_iterator())
    group_labels = np.concatenate(group_labels)
    groups_labels_all.append(group_labels)

    # Collects in-sample and out-of-sample predictions.
    for combo_idx, combo in enumerate(tf.io.gfile.listdir(combos_dir)):
      splits_in_combo = [int(split_idx) for split_idx in combo.split('_')]
      model = trained_models[combo_idx]
      if split_idx in splits_in_combo:
        # Identifies in-sample model and collects its predictions.
        id_predictions_train = model.predict(
            dataloader.train_splits[split_idx].map(
                lambda example: example['input_feature']))
        id_predictions_val = model.predict(
            dataloader.val_splits[split_idx].map(
                lambda example: example['input_feature']))
        id_predictions = tf.concat(
            [id_predictions_train['main'], id_predictions_val['main']], axis=0)
        id_predictions = tf.gather_nd(
            id_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        id_predictions_all.append(id_predictions)
        _, tracin_values_train, _ = calculate_tracin_values(
            dataloader.train_splits[split_idx], [model], has_bias=True)
        _, tracin_values_val, _ = calculate_tracin_values(
            dataloader.val_splits[split_idx], [model], has_bias=True)
        id_tracin_values = tf.concat([tracin_values_train, tracin_values_val],
                                     axis=0)
        id_tracin_values_splits.append(id_tracin_values)
      else:
        # Identifies out-of-sample model and collects its predictions.
        ood_predictions_train = model.predict(
            dataloader.train_splits[split_idx].map(
                lambda example: example['input_feature']))
        ood_predictions_val = model.predict(
            dataloader.val_splits[split_idx].map(
                lambda example: example['input_feature']))
        ood_predictions = tf.concat(
            [ood_predictions_train['main'], ood_predictions_val['main']],
            axis=0)
        ood_predictions = tf.gather_nd(
            ood_predictions, tf.expand_dims(labels, axis=1), batch_dims=1)
        ood_predictions_all.append(ood_predictions)
        _, tracin_values_train, _ = calculate_tracin_values(
            dataloader.train_splits[split_idx], [model], has_bias=True)
        _, tracin_values_val, _ = calculate_tracin_values(
            dataloader.val_splits[split_idx], [model], has_bias=True)
        ood_tracin_values = tf.concat([tracin_values_train, tracin_values_val],
                                      axis=0)
        ood_tracin_values_splits.append(ood_tracin_values)

    # Collects example ids and is_train indicators.
    # NB: The extracted example id are byte strings.
    example_ids_train = list(dataloader.train_splits[split_idx].map(
        lambda example: example['example_id']).as_numpy_iterator())
    example_ids_val = list(dataloader.val_splits[split_idx].map(
        lambda example: example['example_id']).as_numpy_iterator())
    example_ids = example_ids_train + example_ids_val
    example_ids = np.concatenate(example_ids)
    example_ids_all.append(example_ids)

    is_train = tf.concat([
        tf.ones(len(np.concatenate(example_ids_train)), dtype=tf.int64),
        tf.zeros(len(np.concatenate(example_ids_val)), dtype=tf.int64)
    ], axis=0)
    is_train_all.append(is_train)

    # Computes in-sample and out-of-sample predictions and bias values.
    id_predictions_avg = np.average(np.stack(id_predictions_all), axis=0)
    ood_predictions_avg = np.average(np.stack(ood_predictions_all), axis=0)
    id_tracin_values_avg = np.average(np.stack(id_tracin_values_splits), axis=0)
    ood_tracin_values_avg = np.average(
        np.stack(ood_tracin_values_splits), axis=0)
    tracin_values_avg = np.average(
        np.stack(ood_tracin_values_splits + id_tracin_values_splits), axis=0)
    bias_values = np.absolute(
        np.subtract(id_predictions_avg, ood_predictions_avg))
    vars_values = np.std(np.stack(ood_predictions_all), axis=0)
    # Since the `id_predictions_avg` is the predictive probability for the
    # target class. The `noise` is simply the distance between the predicted
    # probability and the true probability of target class (i.e., 1.).
    nois_values = np.absolute(np.subtract(1., id_predictions_avg))
    error_values = np.average(np.subtract(1., ood_predictions_all), axis=0)
    gap_values = np.average(
        np.absolute(np.subtract(id_predictions_avg[None, :],
                                np.stack(ood_predictions_all))), axis=0)

    # Calculates bias labels using value threshold by default.
    # If percentile is specified, use percentile instead.
    if bias_percentile_threshold:
      threshold = np.percentile(bias_values, bias_percentile_threshold)
    else:
      threshold = bias_value_threshold
    bias_labels = tf.math.greater(bias_values, threshold)
    # Calculates tracin labels using value threshold by default.
    # If percentile is specified, use percentile instead.
    if tracin_percentile_threshold:
      tracin_threshold = np.percentile(tracin_values_avg,
                                       tracin_percentile_threshold)
    else:
      tracin_threshold = tracin_value_threshold
    tracin_labels = tf.math.greater(tracin_values_avg, tracin_threshold)

    bias_labels_all.append(bias_labels)
    tracin_labels_all.append(tracin_labels)
    bias_values_all.append(bias_values)
    id_tracin_values_all.append(id_tracin_values_avg)
    ood_tracin_values_all.append(ood_tracin_values_avg)
    vars_values_all.append(vars_values)
    nois_values_all.append(nois_values)

    gap_values_all.append(gap_values)
    error_values_all.append(error_values)

  is_train_all = np.concatenate(is_train_all)
  example_ids_all = np.concatenate(example_ids_all)
  target_labels_all = np.concatenate(target_labels_all)
  groups_labels_all = np.concatenate(groups_labels_all)
  id_tracin_values_all = np.concatenate(id_tracin_values_all)
  ood_tracin_values_all = np.concatenate(ood_tracin_values_all)
  tracin_labels_all = np.squeeze(np.concatenate(tracin_labels_all))

  bias_labels_all = np.squeeze(np.concatenate(bias_labels_all))
  bias_values_all = np.squeeze(np.concatenate(bias_values_all))
  vars_values_all = np.squeeze(np.concatenate(vars_values_all))
  nois_values_all = np.squeeze(np.concatenate(nois_values_all))
  gap_values_all = np.squeeze(np.concatenate(gap_values_all))
  error_values_all = np.squeeze(np.concatenate(error_values_all))

  logging.info('# of examples: %s', example_ids_all.shape[0])
  logging.info('# of target_label: %s', target_labels_all.shape[0])
  logging.info('# of groups_label: %s', groups_labels_all.shape[0])
  logging.info('# of bias_label: %s', bias_labels_all.shape[0])
  logging.info('# of non-zero bias labels: %s',
               tf.math.count_nonzero(bias_labels_all).numpy())

  logging.info('# of bias: %s', bias_values_all.shape[0])
  logging.info('# of variance: %s', vars_values_all.shape[0])
  logging.info('# of noise: %s', nois_values_all.shape[0])
  logging.info('# of gap: %s', nois_values_all.shape[0])
  logging.info('# of noise: %s', gap_values_all.shape[0])
  logging.info('# of error: %s', error_values_all.shape[0])
  logging.info('# of is_train: %s', is_train_all.shape[0])
  logging.info('# of train examples: %s',
               tf.math.count_nonzero(is_train_all).numpy())

  if save_table:
    df = pd.DataFrame({
        'example_id': example_ids_all,
        'target_label': target_labels_all,
        'groups_label': groups_labels_all,
        BIAS_LABEL_KEY: bias_labels_all,
        TRACIN_LABEL_KEY: tracin_labels_all,
        'bias': bias_values_all,
        'variance': vars_values_all,
        'noise': nois_values_all,
        'gap': gap_values_all,
        'error': error_values_all,
        'tracin_id': id_tracin_values_all,
        'tracin_ood': ood_tracin_values_all,
        # Whether this example belongs to the training data.
        'is_train': is_train_all
    })

    csv_name = os.path.join(
        save_dir,
        'bias_table.csv' if ckpt_epoch < 0 else f'bias_table_{ckpt_epoch}.csv')
    df.to_csv(csv_name, index=False)

  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(example_ids_all, dtype=tf.string),
      values=tf.convert_to_tensor(bias_labels_all, dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def get_example_id_to_predictions_table(
    dataloader: data.Dataloader,
    trained_models: List[tf.keras.Model],
    has_bias: bool,
    split: Optional[str] = 'train',
    save_dir: Optional[str] = None,
    save_table: Optional[bool] = True,
    compute_tracin: Optional[bool] = False) -> pd.DataFrame:
  """Generates a lookup table mapping example ID to bias label.

  Args:
    dataloader: Dataclass object containing training and validation data.
    trained_models: List of trained models.
    has_bias: Do the trained models have a bias prediction head
    split: Which split of the dataset to use ('train'/'val'/'test')
    save_dir: Directory in which predictions table will be saved as CSV.
    save_table: Boolean for whether or not to save table.
    compute_tracin: Boolean whether or not to calculate the tracin values
      with respect to the model predictions.

  Returns:
    A pandas dataframe mapping example ID to all label and bias predictions.
  """
  table_name = 'predictions_table'
  ds = dataloader.train_ds
  if split != 'train':
    ds = dataloader.eval_ds[split]
    table_name += '_' + split
  labels = list(
      ds.map(
          lambda example: example['label']).as_numpy_iterator())
  labels = np.concatenate(labels)
  predictions_all = []
  tracin_values_all = []
  if has_bias:
    bias_predictions_all = []
  for idx, model in enumerate(trained_models):
    model = trained_models[idx]
    predictions = model.predict(
        ds.map(lambda example: example['input_feature']))
    predictions_all.append(predictions['main'][..., 1])
    if has_bias:
      bias_predictions_all.append(predictions['bias'][..., 1])
    if compute_tracin:
      _, tracin_values, _ = calculate_tracin_values(
          ds, [model], has_bias=has_bias, use_prediction_gradient=True
      )
      tracin_values_all.append(tracin_values)
  example_ids = list(ds.map(
      lambda example: example['example_id']).as_numpy_iterator())
  example_ids = np.concatenate(example_ids)
  predictions_all = np.stack(predictions_all)
  if has_bias:
    bias_predictions_all = np.stack(bias_predictions_all)
  if compute_tracin:
    tracin_values_all = np.stack(tracin_values_all)

  logging.info('# of examples in prediction table is: %s', example_ids.shape[0])

  dict_values = {'example_id': example_ids}
  for i in range(predictions_all.shape[0]):
    dict_values[f'predictions_label_{i}'] = predictions_all[i]
    if has_bias:
      dict_values[f'predictions_bias_{i}'] = bias_predictions_all[i]
    if compute_tracin:
      dict_values[f'predictions_tracin_{i}'] = tracin_values_all[i]
  df = pd.DataFrame(dict_values)
  if save_table:
    df.to_csv(os.path.join(save_dir, table_name + '.csv'), index=False)
  return df


def get_example_id_to_tracin_value_table(
    dataloader: data.Dataloader,
    model_checkpoints: List[tf.keras.Model],
    has_bias: bool,
    split: Optional[str] = 'train',
    included_layers: Optional[int] = -2,
    save_dir: Optional[str] = None,
    save_table: Optional[bool] = True,
    table_name_suffix: Optional[str] = 'first',
) -> tf.lookup.StaticHashTable:
  """Generates a lookup table mapping example ID to tracin value.

  Args:
    dataloader: Dataclass object containing training and validation data.
    model_checkpoints: List of model checkpoints.
    has_bias: Do the trained models have a bias prediction head
    split: Which split of the dataset to use ('train'/'val'/'test')
    included_layers: Layers to include in Tracin computation (all trainable
      layers from the index forward are included)
    save_dir: Directory in which bias table will be saved as CSV.
    save_table: Boolean for whether or not to save table.
    table_name_suffix: String to add to the name of the created table
  Returns:
    A lookup table mapping example ID to tracin score.
  """

  ds = dataloader.train_ds
  table_name = 'tracin_table'
  if split != 'train':
    ds = dataloader.eval_ds[split]
    table_name += '_' + split
  example_ids_all, tracin_values_all, probs_all = calculate_tracin_values(
      ds, model_checkpoints, included_layers, has_bias)
  logging.info('# of examples: %s', example_ids_all.shape[0])

  if save_table:
    df = pd.DataFrame({
        EXAMPLE_ID_KEY: example_ids_all,
        TRACIN_SCORE_KEY: tracin_values_all,
        PREDICTION_KEY: probs_all
    })
    df.to_csv(
        os.path.join(save_dir, 'tracin_table_'+ table_name_suffix + '.csv'),
        index=False)

  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(example_ids_all, dtype=tf.string),
      values=tf.convert_to_tensor(tracin_values_all, dtype=tf.float64),
      key_dtype=tf.string,
      value_dtype=tf.float64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def load_existing_tracin_table(path_to_table: str):
  """Loads tracin table from file."""
  df = pd.read_csv(path_to_table)
  key_tensor = np.array([eval(x).decode('UTF-8') for    #  pylint:disable=eval-used
                         x in df[EXAMPLE_ID_KEY].to_list()])
  init = tf.lookup.KeyValueTensorInitializer(
      keys=tf.convert_to_tensor(
          key_tensor, dtype=tf.string),
      values=tf.convert_to_tensor(
          df[TRACIN_SCORE_KEY].to_numpy(), dtype=tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def calculate_tracin_values(
    dataset: tf.data.Dataset,
    model_checkpoints: List[tf.keras.Model],
    included_layers: Optional[int] = -2,
    has_bias: Optional[bool] = False,
    use_prediction_gradient: Optional[bool] = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Calculates the tracin values for a given dataset [1].

  Reference:
  [1]: Pruthi et al. Estimating Training Data Influence by Tracing Gradient
  Descent. https://arxiv.org/abs/2002.08484

  Args:
    dataset: Dataset object containing data.
    model_checkpoints: List of model checkpoints.
    included_layers: Layers to include in Tracin computation (all trainable
      layers from the index forward are included.) Default (-2) the last two
      layers are included.
    has_bias:  Do the trained models have a bias prediction head. If yes, layers
      to predrict bias are ignored.
    use_prediction_gradient: Calculate the Tracin values for the loss with
    respect to the predicted labels (instead of true labels)
      instead of loss

  Returns:
    An array of example_ids and a corresponding arrays of tracin values and
    predictions.

  """
  example_ids_all = []
  tracin_values_all = []
  probs_all = []

  included_layers_start = included_layers
  included_layers_end = -1
  if has_bias:
    included_layers_start -= 2
    included_layers_end = -2

  @tf.function
  def run_self_influence(
      batch: Dict[str, tf.Tensor],
      checkpoints: List[tf.keras.Model],
      included_layers_start: int,
      included_layers_end: int,
      use_prediction_gradient: Optional[bool] = False
  ) -> Tuple[tf.Tensor, Any, Any]:
    example_ids = batch['example_id']
    features = batch['input_feature']
    labels = batch['label']
    self_influences = []
    probs_np = []
    for model in checkpoints:
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(
            model.trainable_weights[included_layers_start:included_layers_end])
        probs = model(features)['main']
        if use_prediction_gradient:
          y_pred = tf.math.argmax(probs, axis=1)
          loss = tf.keras.losses.sparse_categorical_crossentropy(y_pred, probs)
          grads = tape.jacobian(
              loss,
              model.trainable_weights[
                  included_layers_start:included_layers_end
              ])
        else:
          loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
          grads = tape.jacobian(
              loss,
              model.trainable_weights[
                  included_layers_start:included_layers_end
              ])
        scores = tf.add_n(
            [
                tf.math.reduce_sum(
                    grad * grad, axis=tf.range(1, tf.rank(grad), 1)
                )
                for grad in grads
            ]
        )
      self_influences.append(scores)
      probs_np.append(probs[:, 0])

    return example_ids, tf.math.reduce_sum(
        tf.stack(self_influences, axis=-1), axis=-1), tf.math.reduce_sum(
            tf.stack(probs_np, axis=-1), axis=-1)

  inputshape = dataset.element_spec['input_feature'].shape
  for model in model_checkpoints:
    model.build(inputshape)

  logging.info('Checkpoints built.')
  for batch in dataset:
    example_ids, tracin_values, probs = run_self_influence(
        batch, model_checkpoints, included_layers_start, included_layers_end,
        use_prediction_gradient)
    example_ids_all.append(example_ids)
    tracin_values_all.append(tracin_values)
    probs_all.append(probs)

  example_ids_all = np.concatenate(example_ids_all)
  tracin_values_all = np.squeeze(np.concatenate(tracin_values_all))
  probs_all = np.concatenate(probs_all)

  return example_ids_all, tracin_values_all, probs_all

# Helper functions to process hash tables


def filter_ids_fn(hash_table, value=1):
  """Filter dataset based on whether ids take a certain value in hash table."""
  def filter_fn(examples):
    return hash_table.lookup(examples['example_id']) == value
  return filter_fn



