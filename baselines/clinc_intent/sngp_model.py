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

"""SNGP with BERT encoder.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
"""
from typing import Dict, Any, Callable
from edward2.experimental import sngp

import tensorflow as tf

_EinsumDense = tf.keras.layers.experimental.EinsumDense


def make_spec_norm_dense_layer(**spec_norm_kwargs: Dict[str, Any]
                               ) -> Callable[[], tf.keras.layers.Layer]:
  """Defines a spectral-normalized EinsumDense layer.

  Args:
    **spec_norm_kwargs: Keyword arguments to the sngp.SpectralNormalization
      layer wrapper.

  Returns:
    (callable) A function that defines a dense layer and wraps it with
      sngp.SpectralNormalization.
  """

  def spec_norm_dense(*dense_args, **dense_kwargs):
    base_layer = _EinsumDense(*dense_args, **dense_kwargs)
    return sngp.SpectralNormalization(base_layer, **spec_norm_kwargs)

  return spec_norm_dense
