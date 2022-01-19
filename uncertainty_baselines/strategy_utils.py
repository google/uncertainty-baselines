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

"""Collection of utilities for running on different device types."""

from typing import Union
import tensorflow.compat.v2 as tf


_Strategy = Union[
    tf.distribute.MirroredStrategy, tf.distribute.TPUStrategy]


def get_strategy(tpu: str, use_tpu: bool) -> _Strategy:
  """Gets a strategy to run locally on CPU or on a fleet of TPUs.

  Args:
    tpu: A string of the main TPU to run on. Ignored if use_tpu=False.
    use_tpu: Whether or not to use TPU or CPU.

  Returns:
    A TPUStrategy if using TPUs, or a MirroredStrategy if not.
  """
  if use_tpu:
    cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=tpu)
    tf.config.experimental_connect_to_cluster(cluster_resolver)
    tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
    tpu_strategy = tf.distribute.TPUStrategy(cluster_resolver)
    return tpu_strategy

  return tf.distribute.MirroredStrategy()
