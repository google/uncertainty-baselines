# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""BatchEnsemble utilities."""
import edward2 as ed
import tensorflow.compat.v2 as tf


def make_sign_initializer(random_sign_init: float) -> tf.keras.initializers:
  """Builds initializer with specified random_sign_init.

  Args:
    random_sign_init: Value used to initialize trainable deterministic
      initializers, as applicable. Values greater than zero result in
      initialization to a random sign vector, where random_sign_init is the
      probability of a 1 value. Values less than zero result in initialization
      from a Gaussian with mean 1 and standard deviation equal to
      -random_sign_init.

  Returns:
    tf.keras.initializers
  """
  if random_sign_init > 0:
    return ed.initializers.RandomSign(random_sign_init)
  else:
    return tf.keras.initializers.RandomNormal(
        mean=1.0, stddev=-random_sign_init)


def is_batch_norm(v):
  """Decide whether a variable belongs to `batch_norm`."""
  keywords = ['batchnorm', 'batch_norm', 'bn']
  return any([k in v.name.lower() for k in keywords])
