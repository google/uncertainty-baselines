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

"""Utilities for CIFAR-10H."""

import json
import os
from typing import Callable, Tuple

from absl import logging
import numpy as np
import tensorflow as tf


def _load_cifar10h_labels(data_dir=None):
  """Load the Cifar-10H labels."""
  # copy of unzipped:
  # https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-raw.zip
  fname = None
  if fname is None:
    url = 'https://github.com/jcpeterson/cifar-10h/raw/master/data/cifar10h-raw.zip'
    fname = tf.keras.utils.get_file(origin=url, extract=True)
    fname = fname[:-4] + '.csv'
  with tf.io.gfile.GFile(fname) as f:
    csv_reader = f.readlines()[1:]

    labels = {}
    for row in csv_reader:
      split_row = row.split(',')
      chosen_label, test_idx = split_row[6], split_row[8]
      if test_idx != '-99999':
        test_id = 'test_' + '0' * (5 - len(test_idx)) + test_idx
        label = np.zeros((10,))
        label[int(chosen_label)] = 1.0
        if test_id not in labels:
          labels[test_id] = (1, label)
        else:
          labels[test_id] = (labels[test_id][0] + 1, labels[test_id][1] + label)

  string_ids, indices, counts, probs = [], [], [], []
  for idx, (string_id, (count, label_sum)) in enumerate(labels.items()):
    string_ids.append(string_id)
    indices.append(idx)
    counts.append(count)
    probs.append((label_sum / count).tolist())

  id_mappings = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(string_ids, indices),
      default_value=-1)

  return id_mappings, tf.cast(counts, tf.float32), tf.cast(probs, tf.float32)


def create_cifar10_to_cifar10h_fn(data_dir=None):
  """Creates a function that maps CIFAR-10 to CIFAR-10H."""
  idx_map, cifar10h_counts, cifar10h_probs = _load_cifar10h_labels(data_dir)

  def convert(example):
    idx = idx_map.lookup(example['id'])
    if idx == -1:
      logging.warn('Index -1 encountered in the CIFAR-10H dataset.')
      example['labels'] = tf.zeros_like(tf.gather(cifar10h_probs, 0))
      example['count'] = tf.zeros_like(tf.gather(cifar10h_counts, 0))
    else:
      example['labels'] = tf.gather(cifar10h_probs, idx)
      example['count'] = tf.gather(cifar10h_counts, idx)
    return example

  return convert


def _load_imagenet_real_labels() -> Tuple[
    tf.lookup.StaticHashTable, tf.Tensor, tf.Tensor]:
  """Load raw ratings ReaL labels are derived from."""
  # raters.npz from github.com/google-research/reassessed-imagenet
  raters_file = None
  with tf.compat.v1.io.gfile.GFile(raters_file, 'rb') as f:
    data = np.load(f)

  summed_ratings = np.sum(data['tensor'], axis=0)
  averaged_ratings = summed_ratings / np.expand_dims(
      np.sum(summed_ratings, axis=-1), -1)
  yes_prob = averaged_ratings[:, 2]

  # convert raw ratings into soft labels
  num_labels = 1000
  soft_labels = {}
  for idx, (file_name, label_id) in enumerate(data['info']):
    if file_name not in soft_labels:
      soft_labels[file_name] = np.zeros(num_labels)
    added_label = np.zeros(num_labels)
    added_label[int(label_id)] = yes_prob[idx]
    soft_labels[file_name] = soft_labels[file_name] + added_label

  # real.json from github.com/google-research/reassessed-imagenet
  real_file = None

  # load ImageNet ReaL labels
  new_real_labels = {}
  with tf.compat.v1.io.gfile.GFile(real_file, 'rb') as f:
    real_labels = json.load(f)
    for idx, label in enumerate(real_labels):
      key = 'ILSVRC2012_val_'
      key += (8 - len(str(idx + 1))) * '0' + str(idx + 1) + '.JPEG'
      if len(label) == 1:
        one_hot_label = np.zeros(num_labels)
        one_hot_label[label[0]] = 1.0
        new_real_labels[key] = (one_hot_label, 1.0)
      else:
        new_real_labels[key] = (np.zeros(num_labels), 0.0)

  # merge soft and hard labels
  for key, soft_label in soft_labels.items():
    count = np.sum(soft_label, axis=-1)
    if count > 0.0:
      # if raw ratings available replace ReaL label with soft raw label
      new_real_labels[key] = (soft_label / np.expand_dims(count, axis=-1), 1.0)

  string_ids, indices, probs, weights = [], [], [], []
  for idx, (string_id, (soft_label, weight)) in enumerate(
      new_real_labels.items()):
    string_ids.append(string_id)
    indices.append(idx)
    probs.append(soft_label)
    weights.append(weight)

  id_mappings = tf.lookup.StaticHashTable(
      tf.lookup.KeyValueTensorInitializer(string_ids, indices),
      default_value=-1)

  return id_mappings, tf.cast(weights, tf.float32), tf.cast(probs, tf.float32)


def create_imagenet_to_real_fn() -> Callable[[tf.Tensor], tf.Tensor]:
  """Creates a function that maps ImageNet labels to ReaL labels."""
  idx_map, real_weights, real_probs = _load_imagenet_real_labels()

  def convert(example: tf.Tensor) -> tf.Tensor:
    idx = idx_map.lookup(example['file_name'])
    if idx == -1:
      logging.warn('Index -1 encountered in the ImageNet real dataset.')
      example['labels'] = tf.zeros_like(tf.gather(real_probs, 0))
      example['mask'] = tf.zeros_like(tf.gather(real_weights, 0))
    else:
      example['labels'] = tf.gather(real_probs, idx)
      example['mask'] = tf.gather(real_weights, idx)
    return example

  return convert


def generalized_energy_distance(labels, predictions, num_classes):
  """Compute generalized energy distance.

  See Eq. (8) https://arxiv.org/abs/2006.06015
  where d(a, b) = (a - b)^2.

  Args:
    labels: [batch_size, num_classes] Tensor with empirical probabilities of
      each class assigned by the labellers.
    predictions: [batch_size, num_classes] Tensor of predicted probabilities.
    num_classes: Integer.

  Returns:
    Tuple of Tensors (label_diversity, sample_diversity, ged).
  """
  y = tf.expand_dims(labels, -1)
  y_hat = tf.expand_dims(predictions, -1)

  non_diag = tf.expand_dims(1.0 - tf.eye(num_classes), 0)
  distance = tf.reduce_sum(tf.reduce_sum(
      non_diag * y * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
  label_diversity = tf.reduce_sum(tf.reduce_sum(
      non_diag * y * tf.transpose(y, perm=[0, 2, 1]), -1), -1)
  sample_diversity = tf.reduce_sum(tf.reduce_sum(
      non_diag * y_hat * tf.transpose(y_hat, perm=[0, 2, 1]), -1), -1)
  ged = 2 * distance - label_diversity - sample_diversity
  return label_diversity, sample_diversity, ged
