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

import os

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
  ged = tf.reduce_mean(2 * distance - label_diversity - sample_diversity)
  return label_diversity, sample_diversity, ged
