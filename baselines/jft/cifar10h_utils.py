# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_cifar10h_labels():
  """Load the Cifar-10H labels."""
  # copy of unzipped:
  # https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-raw.zip
  with tf.compat.v1.io.gfile.GFile(fname) as f:
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


def load_ds():
  """Load CIFAR-10H TFDS.

  Returns:
    Cifar-10H TFDS.
  """
  cifar10_ds = tfds.load('cifar10', split='test')
  idx_map, cifar10h_counts, cifar10h_probs = load_cifar10h_labels()

  def cifar10h_labels(example):
    idx = idx_map.lookup(example['id'])
    example['labels'] = tf.gather(cifar10h_probs, idx)
    example['count'] = tf.gather(cifar10h_counts, idx)
    example['mask'] = 1.0
    return example

  return cifar10_ds.map(cifar10h_labels)


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
