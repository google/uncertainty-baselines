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
import functools

from typing import Any, Dict, Mapping, Optional

import edward2 as ed
import tensorflow as tf

from official.modeling import tf_utils
from official.nlp.bert import configs as bert_configs
from official.nlp.modeling import layers as bert_layers
from official.nlp.modeling import networks as bert_encoder

_EinsumDense = tf.keras.layers.experimental.EinsumDense

# A dict of regex patterns and their replacements. Use to update weight names
# in a classic pre-trained checkpoint to those in
# SpectralNormalizedTransformerEncoder.
CHECKPOINT_REPL_PATTERNS = {
    '/intermediate': '/feedforward/intermediate',
    '/output': '/feedforward/output'
}


def make_spec_norm_dense_layer(**spec_norm_kwargs: Mapping[str, Any]):
  """Defines a spectral-normalized EinsumDense layer.

  Args:
    **spec_norm_kwargs: Keyword arguments to the SpectralNormalization layer
      wrapper.

  Returns:
    (callable) A function that defines a dense layer and wraps it with
      SpectralNormalization.
  """

  def spec_norm_dense(*dense_args, **dense_kwargs):
    base_layer = _EinsumDense(*dense_args, **dense_kwargs)
    # Inhere base_layer name to match with those in a classic BERT checkpoint.
    return ed.layers.SpectralNormalization(
        base_layer, inhere_layer_name=True, **spec_norm_kwargs)

  return spec_norm_dense


class SpectralNormalizedFeedforwardLayer(tf.keras.layers.Layer):
  """Two-layer feed-forward network with spectral-normalized dense layers.

  This class implements a drop-in replacement of the feedforward_block module
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
        'inner_dim': 1024,
        'inner_activation': 'gelu',
        'dropout': 0.1,
        'name': 'feedforward',
      }
  >>> feedforward_cfg.update(common_kwargs)
  >>> feedforward_block = feedforward_cls(**feedforward_cfg)
  """

  def __init__(self,
               inner_dim: int,
               inner_activation: str,
               # TODO(yquan): Remove the following 2 unused fields after they
               # are removed from TransformerScaffold.py
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
      inner_dim: Size of the intermediate layer.
      inner_activation: Activation function to be used for the intermediate
        layer.
      intermediate_size (to-be-removed): Same as inner_dim.
      intermediate_activation (to-be-removed): Same as inner_activation.
      dropout: Dropout rate.
      use_layer_norm: Whether to use layer normalization.
      use_spec_norm: Whether to use spectral normalization.
      spec_norm_kwargs: Keyword arguments to the spectral normalization layer.
      name: Layer name.
      **common_kwargs: Other common keyword arguments for the hidden dense
        layers.
    """
    super().__init__(name=name)
    self._inner_dim = inner_dim
    self._inner_activation = inner_activation
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
        output_shape=(None, self._inner_dim),
        bias_axes='d',
        name='intermediate',
        **self._common_kwargs)
    policy = tf.keras.mixed_precision.global_policy()
    if policy.name == 'mixed_bfloat16':
      # bfloat16 causes BERT with the LAMB optimizer to not converge
      # as well, so we use float32.
      policy = tf.float32
    self._intermediate_activation_layer = tf.keras.layers.Activation(
        self._inner_activation, dtype=policy)
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
        'inner_dim': self._inner_dim,
        'inner_activation': self._inner_activation,
        'intermediate_size': self._inner_dim,
        'intermediate_activation': self._inner_activation,
        'dropout': self._dropout,
        'use_layer_norm': self._use_layer_norm,
        'use_spec_norm': self._use_spec_norm,
        'spec_norm_kwargs': self._spec_norm_kwargs
    })
    config.update(self._common_kwargs)
    return config


class SpectralNormalizedMultiHeadAttention(tf.keras.layers.MultiHeadAttention):
  """Multi-head attention with spectral-normalized dense layers.

  This is an implementation of multi-headed attention layer [2] with the option
  to replace the original EinsumDense layer with its spectral-normalized
  counterparts.
  """

  def __init__(self,
               use_spec_norm: bool = False,
               spec_norm_kwargs: Optional[Dict[str, Any]] = None,
               **kwargs: Dict[str, Any]):
    super().__init__(**kwargs)
    self._use_spec_norm = use_spec_norm
    self._spec_norm_kwargs = spec_norm_kwargs
    self._spec_norm_dense_layer = make_spec_norm_dense_layer(**spec_norm_kwargs)

  def _update_einsum_dense(
      self, einsum_dense_layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
    """Updates the EinsumDense layer to its spectral-normalized counterparts."""
    if not self._use_spec_norm:
      return einsum_dense_layer

    # Overwrites EinsumDense using the same arguments.
    einsum_dense_kwargs = einsum_dense_layer.get_config()
    return self._spec_norm_dense_layer(**einsum_dense_kwargs)

  def _build_from_signature(self,
                            query: tf.Tensor,
                            value: tf.Tensor,
                            key: Optional[tf.Tensor] = None):
    """Builds layers and variables.

    This function overwrites the default _build_from_signature to build dense
    layers from self.einsum_dense_layer. Once the method is called,
    self._built_from_signature will be set to True.

    Args:
      query: query tensor or TensorShape.
      value: value tensor or TensorShape.
      key: key tensor or TensorShape.
    """
    super()._build_from_signature(query, value, key)  # pytype: disable=attribute-error  # typed-keras
    # Overwrites EinsumDense layers.
    # TODO(b/168256394): Enable spectral normalization also for key, query and
    # value layers in the self-attention module.
    self._output_dense = self._update_einsum_dense(self._output_dense)

  def get_config(self):
    config = super().get_config()
    config['use_spec_norm'] = self._use_spec_norm
    config['spec_norm_kwargs'] = self._spec_norm_kwargs
    return config


class SpectralNormalizedTransformer(bert_layers.TransformerScaffold):
  """Transformer layer with spectral-normalized dense layers."""

  def __init__(self,
               use_layer_norm_att: bool = True,
               use_layer_norm_ffn: bool = True,
               use_spec_norm_att: bool = False,
               use_spec_norm_ffn: bool = False,
               spec_norm_kwargs: Optional[Mapping[str, Any]] = None,
               **kwargs):
    """Initializer.

    Args:
      use_layer_norm_att: Whether to use layer normalization in the attention
        layer.
      use_layer_norm_ffn: Whether to use layer normalization in the feedforward
        layer.
      use_spec_norm_att: Whether to use spectral normalization in the attention
        layer.
      use_spec_norm_ffn: Whether to use spectral normalization in the
        feedforward layer.
      spec_norm_kwargs: Keyword arguments to the spectral normalization layer.
      **kwargs: Additional keyword arguments to TransformerScaffold.
    """
    self._use_layer_norm_att = use_layer_norm_att
    self._use_layer_norm_ffn = use_layer_norm_ffn
    self._use_spec_norm_att = use_spec_norm_att
    self._use_spec_norm_ffn = use_spec_norm_ffn
    self._spec_norm_kwargs = spec_norm_kwargs

    feedforward_cls = functools.partial(
        SpectralNormalizedFeedforwardLayer,
        use_layer_norm=self._use_layer_norm_ffn,
        use_spec_norm=self._use_spec_norm_ffn,
        spec_norm_kwargs=self._spec_norm_kwargs)

    attention_cls = functools.partial(
        SpectralNormalizedMultiHeadAttention,
        use_spec_norm=self._use_spec_norm_att,
        spec_norm_kwargs=self._spec_norm_kwargs)

    super().__init__(
        feedforward_cls=feedforward_cls, attention_cls=attention_cls, **kwargs)

  def call(self, inputs):
    """Overwrites default call function to allow diabling layernorm."""
    if isinstance(inputs, (list, tuple)) and len(inputs) == 2:
      input_tensor, attention_mask = inputs
    else:
      input_tensor, attention_mask = (inputs, None)

    attention_output = self._attention_layer(
        query=input_tensor, value=input_tensor, attention_mask=attention_mask)
    attention_output = self._attention_dropout(attention_output)
    attention_output = input_tensor + attention_output
    if self._use_layer_norm_att:
      attention_output = self._attention_layer_norm(attention_output)

    if self._feedforward_block is None:
      intermediate_output = self._intermediate_dense(attention_output)
      intermediate_output = self._intermediate_activation_layer(
          intermediate_output)
      layer_output = self._output_dense(intermediate_output)
      layer_output = self._output_dropout(layer_output)
      # During mixed precision training, attention_output is from layer norm
      # and is always fp32 for now. Cast layer_output to fp32 for the subsequent
      # add.
      layer_output = tf.cast(layer_output, tf.float32)
      layer_output = self._output_layer_norm(layer_output + attention_output)
    else:
      layer_output = self._feedforward_block(attention_output)

    return layer_output


class SpectralNormalizedTransformerEncoder(bert_encoder.EncoderScaffold):
  """Spectral-normalized Transformer Encoder with default embedding layer."""

  def __init__(
      self,
      use_spec_norm_att: bool = False,
      use_spec_norm_ffn: bool = False,
      use_spec_norm_plr: bool = False,
      use_layer_norm_att: bool = True,
      use_layer_norm_ffn: bool = True,
      # A dict of kwargs to pass to the Transformer class.
      hidden_cfg: Optional[Dict[str, Any]] = None,
      **kwargs: Mapping[str, Any]):
    """Initializer."""
    hidden_cls = SpectralNormalizedTransformer

    # Add layer normalization arguments to default transformer config.
    normalization_cfg = {
        'use_layer_norm_att': use_layer_norm_att,
        'use_layer_norm_ffn': use_layer_norm_ffn,
        'use_spec_norm_att': use_spec_norm_att,
        'use_spec_norm_ffn': use_spec_norm_ffn,
    }

    if hidden_cfg:
      hidden_cfg.update(normalization_cfg)
    else:
      hidden_cfg = normalization_cfg

    # Intialize default layers.
    super().__init__(hidden_cls=hidden_cls, hidden_cfg=hidden_cfg, **kwargs)

    # Rebuild BERT model graph using default layers.
    seq_length = self._embedding_cfg.get('seq_length', None)

    # Create inputs layers.
    word_ids = tf.keras.layers.Input(
        shape=(seq_length,), dtype=tf.int32, name='input_word_ids')
    mask = tf.keras.layers.Input(
        shape=(seq_length,), dtype=tf.int32, name='input_mask')
    type_ids = tf.keras.layers.Input(
        shape=(seq_length,), dtype=tf.int32, name='input_type_ids')
    inputs = [word_ids, mask, type_ids]

    # Define Input Embeddings Layers.
    word_embeddings = self._embedding_layer(word_ids)
    position_embeddings = self._position_embedding_layer(word_embeddings)
    type_embeddings = self._type_embedding_layer(type_ids)

    embeddings = tf.keras.layers.Add()(
        [word_embeddings, position_embeddings, type_embeddings])
    # TODO(jereliu): Add option to disable embedding layer normalization.
    embeddings = self._embedding_norm_layer(embeddings)
    embeddings = (
        tf.keras.layers.Dropout(
            rate=self._embedding_cfg['dropout_rate'])(embeddings))

    # Define self-attention layers. Rename to match with BERT checkpoint.
    attention_mask = bert_layers.SelfAttentionMask()([embeddings, mask])
    data = embeddings

    layer_output_data = []
    self._hidden_layers = []
    for i in range(self._num_hidden_instances):
      layer = hidden_cls(
          **self._hidden_cfg,
          name='transformer/layer_%d' % i)  # Rename to match BERT checkpoint.
      data = layer([data, attention_mask])
      layer_output_data.append(data)
      self._hidden_layers.append(layer)

    # Extract BERT encoder output (i.e., the CLS token).
    first_token_tensor = (
        tf.keras.layers.Lambda(lambda x: tf.squeeze(x[:, 0:1, :], axis=1))(
            layer_output_data[-1]))

    # Define the pooler layer (i.e., the output layer), and optionally apply
    # spectral normalization.
    self._pooler_layer = tf.keras.layers.Dense(
        units=self._pooled_output_dim,
        activation='tanh',
        kernel_initializer=self._pooler_layer_initializer,
        name='pooler_transform')
    if use_spec_norm_plr:
      self._pooler_layer = ed.layers.SpectralNormalization(
          self._pooler_layer,
          inhere_layer_name=True,
          **hidden_cfg['spec_norm_kwargs'])

    cls_output = self._pooler_layer(first_token_tensor)

    if self._return_all_layer_outputs:
      outputs = [layer_output_data, cls_output]
    else:
      outputs = [layer_output_data[-1], cls_output]

    # Compile model with updated graph.
    super(bert_encoder.EncoderScaffold, self).__init__(
        inputs=inputs, outputs=outputs, **self._kwargs)


def get_spectral_normalized_transformer_encoder(
    bert_config: bert_configs.BertConfig,
    spec_norm_kwargs: Mapping[str, Any],
    use_layer_norm_att: bool = True,
    use_layer_norm_ffn: bool = True,
    use_spec_norm_att: bool = False,
    use_spec_norm_ffn: bool = False,
    use_spec_norm_plr: bool = False) -> SpectralNormalizedTransformerEncoder:
  """Creates a SpectralNormalizedTransformerEncoder from a bert_config.

  Args:
    bert_config: A 'BertConfig' object.
    spec_norm_kwargs: Keyword arguments to the spectral normalization layer.
    use_layer_norm_att: (bool) Whether to apply layer normalization to the
      attention layer.
    use_layer_norm_ffn: (bool) Whether to apply layer normalization to the
      feedforward layer.
    use_spec_norm_att: (bool) Whether to apply spectral normalization to the
      attention layer.
    use_spec_norm_ffn: (bool) Whether to apply spectral normalization to the
      feedforward layer.
    use_spec_norm_plr: (bool) Whether to apply spectral normalization to the
      final pooler layer for CLS token.

  Returns:
    A SpectralNormalizedTransformerEncoder object.
  """
  embedding_cfg = dict(
      vocab_size=bert_config.vocab_size,
      type_vocab_size=bert_config.type_vocab_size,
      hidden_size=bert_config.hidden_size,
      max_seq_length=bert_config.max_position_embeddings,
      initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      dropout_rate=bert_config.hidden_dropout_prob,
  )
  hidden_cfg = dict(
      num_attention_heads=bert_config.num_attention_heads,
      inner_dim=bert_config.intermediate_size,
      inner_activation=tf_utils.get_activation(bert_config.hidden_act),
      dropout_rate=bert_config.hidden_dropout_prob,
      attention_dropout_rate=bert_config.attention_probs_dropout_prob,
      kernel_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range),
      spec_norm_kwargs=spec_norm_kwargs,
  )
  kwargs = dict(
      embedding_cfg=embedding_cfg,
      num_hidden_instances=bert_config.num_hidden_layers,
      pooled_output_dim=bert_config.hidden_size,
      pooler_layer_initializer=tf.keras.initializers.TruncatedNormal(
          stddev=bert_config.initializer_range))

  return SpectralNormalizedTransformerEncoder(
      use_layer_norm_att=use_layer_norm_att,
      use_layer_norm_ffn=use_layer_norm_ffn,
      use_spec_norm_att=use_spec_norm_att,
      use_spec_norm_ffn=use_spec_norm_ffn,
      use_spec_norm_plr=use_spec_norm_plr,
      hidden_cfg=hidden_cfg,
      **kwargs)


class BertGaussianProcessClassifier(tf.keras.Model):
  """Classifier model based on a Gaussian process with BERT encoder."""

  def __init__(self,
               network: tf.keras.Model,
               num_classes: int,
               num_heads: int,
               gp_layer_kwargs: Dict[str, Any],
               initializer: Optional[tf.keras.initializers.Initializer] = None,
               dropout_rate: float = 0.1,
               use_gp_layer: bool = True,
               **kwargs: Mapping[str, Any]):
    """Initializer.

    Args:
      network: A transformer network. This network should output a sequence
        output and a classification output. Furthermore, it should expose its
        embedding table via a "get_embedding_table" method.
      num_classes: Number of classes to predict from the classification network.
      num_heads: Number of additional output heads.
      gp_layer_kwargs: Keyword arguments to Gaussian process layer.
      initializer: The initializer (if any) to use in the classification
        networks. Defaults to a Glorot uniform initializer.
      dropout_rate: The dropout probability of the cls head.
      use_gp_layer: Whether to use Gaussian process output layer.
      **kwargs: Additional keyword arguments.
    """
    self._self_setattr_tracking = False
    self._network = network
    self._config = {
        'network': network,
        'num_classes': num_classes,
        'initializer': initializer,
        'dropout_rate': dropout_rate,
        'use_gp_layer': use_gp_layer,
        'gp_layer_kwargs': gp_layer_kwargs
    }

    # We want to use the inputs of the passed network as the inputs to this
    # Model. To do this, we need to keep a handle to the network inputs for use
    # when we construct the Model object at the end of init.
    inputs = network.inputs

    # Construct classifier using CLS token of the BERT encoder output.
    _, cls_output = network(inputs)
    cls_output = tf.keras.layers.Dropout(rate=dropout_rate)(cls_output)

    # Produce final logits.
    if use_gp_layer:
      # We use the stddev=0.05 (i.e., the tf keras default)
      # for the distribution of the random features instead of stddev=1.
      # (which is often suggested by the theoretical literature).
      # The reason is deep BERT model is sensitive to the scaling of the
      # initializers.
      self.classifier = ed.layers.RandomFeatureGaussianProcess(
          units=num_classes,
          scale_random_features=False,
          use_custom_random_features=True,
          kernel_initializer=initializer,
          custom_random_features_initializer=(
              tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)),
          **gp_layer_kwargs)
    else:
      self.classifier = bert_encoder.Classification(
          input_width=cls_output.shape[-1],
          num_classes=num_classes,
          initializer=initializer,
          output='logits',
          name='sentence_prediction')
    outputs = self.classifier(cls_output)

    # Build additional heads if num_heads > 1.
    if num_heads > 1:
      outputs = [outputs]
      for head_id in range(1, num_heads):
        additional_outputs = tf.keras.layers.Dense(
            num_classes,
            activation=None,
            kernel_initializer=initializer,
            name=f'predictions/transform/logits_{head_id}')(
                cls_output)

        outputs.append(additional_outputs)

    super().__init__(inputs=inputs, outputs=outputs, **kwargs)


def bert_sngp_model(num_classes,
                    bert_config,
                    gp_layer_kwargs,
                    spec_norm_kwargs,
                    num_heads=1,
                    use_gp_layer=True,
                    use_spec_norm_att=True,
                    use_spec_norm_ffn=True,
                    use_layer_norm_att=False,
                    use_layer_norm_ffn=False,
                    use_spec_norm_plr=False):
  """Creates a BERT SNGP classifier model."""
  last_layer_initializer = tf.keras.initializers.TruncatedNormal(
      stddev=bert_config.initializer_range)

  # Build encoder model.
  sngp_bert_encoder = get_spectral_normalized_transformer_encoder(
      bert_config,
      spec_norm_kwargs,
      use_layer_norm_att=use_layer_norm_att,
      use_layer_norm_ffn=use_layer_norm_ffn,
      use_spec_norm_att=use_spec_norm_att,
      use_spec_norm_ffn=use_spec_norm_ffn,
      use_spec_norm_plr=use_spec_norm_plr)

  # Build classification model.
  sngp_bert_model = BertGaussianProcessClassifier(
      sngp_bert_encoder,
      num_classes=num_classes,
      num_heads=num_heads,
      initializer=last_layer_initializer,
      dropout_rate=bert_config.hidden_dropout_prob,
      use_gp_layer=use_gp_layer,
      gp_layer_kwargs=gp_layer_kwargs)

  return sngp_bert_model, sngp_bert_encoder
