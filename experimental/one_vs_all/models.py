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

"""subclassed models from ub.models with distance-based logits."""
from typing import Any, Dict

import tensorflow as tf
import uncertainty_baselines as ub


# This layer has the same behaviour as
# https://github.com/dlmacedo/distinction-maximization-loss/blob/5393b5d0ec25de9d7f809faa10fd22623797d3d8/losses/custom.py#L10
class DisMax(tf.keras.layers.Layer):
  r"""Implements the output layer of model for Distinction Maximization Loss.

  In Distinction Maximization loss, the logits produced by the output layer of
  a neural network are defined as `logits = - ||f_{\theta}(x) - W||`/. This
  layer implements the loss as specified here - https://arxiv.org/abs/1908.05569
  """

  def __init__(self, num_classes: int = 10):
    super(DisMax, self).__init__()
    self.num_classes = num_classes

  def build(self, input_shape):
    self.w = self.add_weight("w",
                             shape=(input_shape[-1], self.num_classes),
                             initializer="zeros",
                             trainable=True)

  def call(self, inputs: tf.Tensor):
    distances = tf.norm(
        tf.expand_dims(inputs, axis=-1) - tf.expand_dims(self.w, axis=0),
        axis=1)
    # In DM Loss, the probability predictions do not have the alpha term.
    return -1.0 * distances


def create_model(
    batch_size: int,
    l2_weight: float = 0.0,
    num_classes: int = 10,
    distance_logits: bool = False,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Resnet-20 v1, takes (32, 32, 3) input and returns logits of shape (10,)."""

  resnet_model = ub.models.get("wide_resnet", batch_size=batch_size,
                               depth=28, width_multiplier=10,
                               l2_weight=l2_weight)

  if distance_logits:
    x = resnet_model.layers[-1].output
    out = DisMax(num_classes=num_classes)(x)
    return tf.keras.Model(
        inputs=resnet_model.inputs,
        outputs=out,
        name=resnet_model.name + "_distance-logits")
  else:
    return resnet_model
