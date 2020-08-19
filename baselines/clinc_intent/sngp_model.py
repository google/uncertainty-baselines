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

[2]: Ashish Vaswani et al. Attention Is All You Need.
     _Neural Information Processing System_, 2017.
     https://papers.nips.cc/paper/7181-attention-is-all-you-need
"""
from typing import Any, Dict, Mapping, Optional
from edward2.experimental import sngp

import tensorflow as tf

_EinsumDense = tf.keras.layers.experimental.EinsumDense


def make_spec_norm_dense_layer(**spec_norm_kwargs: Mapping[str, Any]):
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


class SpectralNormalizedFeedforwardLayer(tf.keras.layers.Layer):
  """Two-layer feed-forward network with spectral-normalized dense layers.

  This class implements a dropout-in replacement of the feedforward_block module
  within tensorflow_models.official.nlp.modeling.layers.TransformerScaffold,
  with additional options for applying spectral normalization to its hidden
  weights, and for turning off layer normalization.

  The intended use of this class is as below:

  >>> feedforward_cls = functools.partial(
        SpectralNormalizedFeedforwardLayer,
        spec_norm_hparams=spec_norm_hparams)
  >>> common_kwargs = {
        'kernel_initializer': 'glorot_uniform'
      }
  >>> feedforward_cfg = {
        'intermediate_size': 1024,
        'intermediate_activation': 'gelu',
        'dropout': 0.1,
        'name': 'feedforward',
      }
  >>> feedforward_cfg.update(common_kwargs)
  >>> feedforward_block = feedforward_cls(**feedforward_cfg)
  """

  def __init__(self,
               intermediate_size: int,
               intermediate_activation: str,
               dropout: float,
               use_layer_norm: bool = True,
               use_spec_norm: bool = False,
               spec_norm_kwargs: Optional[Mapping[str, Any]] = None,
               name: str = 'feedforward',
               **common_kwargs: Mapping[str, Any]):
    """Initializer.

    The arguments corresponds to the keyword arguments in feedforward_cls
    in the TransformerScaffold class.

    Args:
      intermediate_size: Size of the intermediate layer.
      intermediate_activation: Activation function to be used for the
        intermediate layer.
      dropout: Dropout rate.
      use_layer_norm: Whether to use layer normalization.
      use_spec_norm: Whether to use spectral normalization.
      spec_norm_kwargs: Keyword arguments to the spectral normalization layer.
      name: Layer name.
      **common_kwargs: Other common keyword arguments for the hidden dense
        layers.
    """
    super().__init__(name=name)
    self._intermediate_size = intermediate_size
    self._intermediate_activation = intermediate_activation
    self._dropout = dropout
    self._use_layer_norm = use_layer_norm
    self._use_spec_norm = use_spec_norm
    self._spec_norm_kwargs = spec_norm_kwargs
    self._common_kwargs = common_kwargs

    # Defines the EinsumDense layer.
    if self._use_spec_norm:
      self.einsum_dense_layer = make_spec_norm_dense_layer(**spec_norm_kwargs)
    else:
      self.einsum_dense_layer = _EinsumDense

  def build(self, input_shape: tf.TensorShape) -> None:
    hidden_size = input_shape.as_list()[-1]

    self._intermediate_dense = self.einsum_dense_layer(
        'abc,cd->abd',
        output_shape=(None, self._intermediate_size),
        bias_axes='d',
        name='intermediate',
        **self._common_kwargs)
    policy = tf.keras.mixed_precision.experimental.global_policy()
    if policy.name == 'mixed_bfloat16':
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._intermediate_activation, dtype=policy)
    self._output_dense = self.einsum_dense_layer(
        'abc,cd->abd',
        output_shape=(None, hidden_size),
        bias_axes='d',
        name='output',
        **self._common_kwargs)
    self._output_dropout = tf.keras.layers.Dropout(rate=self._dropout)
    # Use float32 in layernorm for numeric stability.
    self._output_layer_norm = tf.keras.layers.LayerNormalization(
        name='output_layer_norm', axis=-1, epsilon=1e-12, dtype=tf.float32)

    super().build(input_shape)

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    intermediate_output = self._intermediate_dense(inputs)
    intermediate_output = self._intermediate_activation_layer(
        intermediate_output)
    layer_output = self._output_dense(intermediate_output)
    layer_output = self._output_dropout(layer_output, training=training)
    # During mixed precision training, attention_output is from layer norm
    # and is always fp32 for now. Cast layer_output to fp32 for the subsequent
    # add.
    layer_output = tf.cast(layer_output, tf.float32)
    residual_output = layer_output + inputs

    if self._use_layer_norm:
      return self._output_layer_norm(residual_output)
    return residual_output

  def get_config(self) -> Dict[str, Any]:
    config = super().get_config()
    config.update({
        'intermediate_size': self._intermediate_size,
        'intermediate_activation': self._intermediate_activation,
        'dropout': self._dropout,
        'use_layer_norm': self._use_layer_norm,
        'use_spec_norm': self._use_spec_norm,
        'spec_norm_kwargs': self._spec_norm_kwargs
    })
    config.update(self._common_kwargs)
    return config
