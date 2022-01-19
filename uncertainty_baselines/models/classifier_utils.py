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

"""Utilities related to classifier building."""

from typing import Dict, Any, Optional

import edward2 as ed
import tensorflow as tf


def build_classifier(
    num_classes: int,
    gp_layer_kwargs: Dict[str, Any],
    use_gp_layer: bool = False,
    kernel_regularizer: Optional[tf.keras.regularizers.Regularizer] = None
) -> tf.keras.Model:
  """Builds a classifier.

  Args:
    num_classes: Number of output classes.
    gp_layer_kwargs: Dict of parameters used in Gaussian Process layer.
    use_gp_layer: Bool, if set True, GP layer is used to build classifier.
    kernel_regularizer: Regularization function for Dense classifider.

  Returns:
    A Keras Layer producing classification logits.
  """
  if use_gp_layer:
    # We use the stddev=0.05 (i.e., the tf keras default)
    # This can be adjusted in the future.
    classifier = ed.layers.RandomFeatureGaussianProcess(
        units=num_classes,
        scale_random_features=False,
        use_custom_random_features=True,
        custom_random_features_initializer=(
            tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
        **gp_layer_kwargs)
  else:
    classifier = tf.keras.layers.Dense(
        num_classes, kernel_regularizer=kernel_regularizer)

  return classifier
