# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""Various losses functions."""

import tensorflow.compat.v2 as tf


def compute_focal_loss(labels, logits, gamma=0.0):
  """Focal loss.

  Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of
  the IEEE international conference on computer vision. 2017.

  focal loss = - (1-p)^gamma * log(p). When gamma=0, the loss is simply the
  regular cross entropy loss.

  Args:
    labels: true labels (batch_size, ).
    logits: logits from the last layer (batch_size, num_classes).
    gamma: the hyperparameter to the focal loss function. When gamma=0, the loss
      is simply the regular cross entropy loss.

  Returns:
    Average loss over the batch.

  """
  _, n_classes = logits.shape
  probs = tf.math.softmax(logits, axis=-1)
  labels_one_hot = tf.one_hot(labels, depth=n_classes)
  probs_target_class = tf.reduce_sum(probs * labels_one_hot, axis=1)
  ll = tf.math.pow(1 - probs_target_class,
                   gamma) * tf.math.log(probs_target_class)
  return -tf.reduce_mean(ll)
