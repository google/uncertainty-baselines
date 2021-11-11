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

"""Semantic segmentation model with a Vision Transformer backbone."""

import functools
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, List

from absl import logging
import flax.linen as nn
from flax.training import common_utils
from immutabledict import immutabledict
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import scipy

from uncertainty_baselines.models import segvit_utils as model_utils

Initializer = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]

# Aliases for custom types:
Batch = Dict[str, jnp.ndarray]
MetricNormalizerFnDict = Mapping[
    str, Tuple[Callable[[jnp.ndarray, bool, Optional[jnp.ndarray]], float],
               Callable[[jnp.ndarray, bool, Optional[jnp.ndarray]], float]]]
MetricFn = Callable[[jnp.ndarray, Dict[str, jnp.ndarray]],
                    Dict[str, Tuple[float, int]]]

# ------------------------------------------------------------------------
#
# Edited from third_party/py/scenic/model_lib/base_models/base_model.py
#
# ------------------------------------------------------------------------


class BaseModel(object):
  """Defines commonalities between all models.

  A model is class with three members: get_metrics_fn, loss_fn, and a
  flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(logits, label, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines a cross_entropy_loss with weight decay, where the
  weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('fully_connected_classification')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  And this is how to call the model:
    variables = {'params': params, **model_state}
    logits, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def __init__(
      self,
      config: Optional[ml_collections.ConfigDict],
      dataset_meta_data: Dict[str, Any],
  ) -> None:
    if config is None:
      logging.warning('You are creating the model with default config.')
      config = self.default_flax_model_config()
    self.config = config
    self.dataset_meta_data = dataset_meta_data
    self.flax_model = self.build_flax_model()

  def get_metrics_fn(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].

    Returns:
      A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    raise NotImplementedError('Subclasses must implement get_metrics_fn.')

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns the loss.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    raise NotImplementedError('Subclasses must implement loss_function.')

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')


# ------------------------------------------------------------------------
#
# Edited from third_party/py/scenic/model_lib/base_models/segmentation_model.py
#
# ------------------------------------------------------------------------

GlobalMetricFn = Callable[[List[jnp.ndarray], Dict[str, Any]], Dict[str, float]]


def global_metrics_fn(all_confusion_mats: List[jnp.ndarray],
                      dataset_metadata: Dict[str, Any]) -> Dict[str, float]:
  """Returns a dict with global (whole-dataset) metrics."""
  # Compute mIoU from list of confusion matrices:
  assert isinstance(all_confusion_mats, list)  # List of eval batches.
  cm = np.sum(all_confusion_mats, axis=0)  # Sum over eval batches.
  assert cm.ndim == 3, ('Expecting confusion matrix to have shape '
                        '[batch_size, num_classes, num_classes], got '
                        f'{cm.shape}.')
  cm = np.sum(cm, axis=0)  # Sum over batch dimension.
  mean_iou, iou_per_class = model_utils.mean_iou(cm)
  metrics_dict = {'mean_iou': float(mean_iou)}
  for label, iou in enumerate(iou_per_class):
    tag = f'iou_per_class/{label:02.0f}'
    if 'class_names' in dataset_metadata:
      tag = f"{tag}_{dataset_metadata['class_names'][label]}"
    metrics_dict[tag] = float(iou)
  return metrics_dict


def num_pixels(logits: jnp.ndarray,
               one_hot_targets: jnp.ndarray,
               weights: Optional[jnp.ndarray] = None) -> float:
  """Computes number of pixels in the target to be used for normalization.

  It needs to have the same API as other defined metrics.

  Args:
    logits: Unused.
    one_hot_targets: Targets, in form of one-hot vectors.
    weights: Input weights (can be used for accounting the padding in the
      input).

  Returns:
    Number of (non-padded) pixels in the input.
  """
  del logits
  if weights is None:
    return np.prod(one_hot_targets.shape[:3])
  assert weights.ndim == 3, (
      'For segmentation task, the weights should be a pixel level mask.')
  return weights.sum()


# Standard default metrics for the semantic segmentation models.
_SEMANTIC_SEGMENTATION_METRICS = immutabledict({
    'accuracy': (model_utils.weighted_correctly_classified, num_pixels),

    # The loss is already normalized, so we set num_pixels to 1.0:
    'loss': (model_utils.weighted_softmax_cross_entropy, lambda *a, **kw: 1.0)
})


def semantic_segmentation_metrics_function(
    logits: jnp.ndarray,
    batch: Batch,
    target_is_onehot: bool = False,
    metrics: MetricNormalizerFnDict = _SEMANTIC_SEGMENTATION_METRICS,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
  """Calculates metrics for the semantic segmentation task.

  Currently we assume each metric_fn has the API:
    ```metric_fn(logits, targets, weights)```
  and returns an array of shape [batch_size]. We also assume that to compute
  the aggregate metric, one should sum across all batches, then divide by the
  total samples seen. In this way we currently only support metrics of the 1/N
  sum f(inputs, targets). Note, the caller is responsible for dividing by
  the normalizer when computing the mean of each metric.

  Args:
   logits: Output of model in shape [batch, length, num_classes].
   batch: Batch of data that has 'label' and optionally 'batch_mask'.
   target_is_onehot: If the target is a one-hot vector.
   metrics: The semantic segmentation metrics to evaluate. The key is the name
     of the metric, and the value is the metrics function.

  Returns:
    A dict of metrics, in which keys are metrics name and values are tuples of
    (metric, normalizer).
  """
  if target_is_onehot:
    one_hot_targets = batch['label']
  else:
    one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])
  weights = batch.get('batch_mask')  # batch_mask might not be defined

  # This psum is required to correctly evaluate with multihost. Only host 0
  # will report the metrics, so we must aggregate across all hosts. The psum
  # will map an array of shape [n_global_devices, batch_size] -> [batch_size]
  # by summing across the devices dimension. The outer sum then sums across the
  # batch dim. The result is then we have summed across all samples in the
  # sharded batch.
  evaluated_metrics = {}
  for key, val in metrics.items():
    evaluated_metrics[key] = model_utils.psum_metric_normalizer(
        (val[0](logits, one_hot_targets,
                weights), val[1](logits, one_hot_targets, weights)))
  return evaluated_metrics


class SegmentationModel(BaseModel):
  """Defines commonalities between all semantic segmentation models.

  A model is class with three members: get_metrics_fn, loss_fn, and a
  flax_model.

  get_metrics_fn returns a callable function, metric_fn, that calculates the
  metrics and returns a dictionary. The metric function computes f(x_i, y_i) on
  a minibatch, it has API:
    ```metric_fn(logits, label, weights).```

  The trainer will then aggregate and compute the mean across all samples
  evaluated.

  loss_fn is a function of API
    loss = loss_fn(logits, batch, model_params=None).

  This model class defines a softmax_cross_entropy_loss with weight decay,
  where the weight decay factor is determined by config.l2_decay_factor.

  flax_model is returned from the build_flax_model function. A typical
  usage pattern will be:
    ```
    model_cls = model_lib.models.get_model_cls('simple_cnn_segmentation')
    model = model_cls(config, dataset.meta_data)
    flax_model = model.build_flax_model
    dummy_input = jnp.zeros(input_shape, model_input_dtype)
    model_state, params = flax_model.init(
        rng, dummy_input, train=False).pop('params')
    ```
  And this is how to call the model:
    variables = {'params': params, **model_state}
    logits, new_model_state = flax_model.apply(variables, inputs, ...)
    ```
  """

  def get_metrics_fn(self, split: Optional[str] = None) -> MetricFn:
    """Returns a callable metric function for the model.

    Args:
      split: The split for which we calculate the metrics. It should be one of
        the ['train',  'validation', 'test'].
    Returns: A metric function with the following API: ```metrics_fn(logits,
      batch)```
    """
    del split  # For all splits, we return the same metric functions.
    return functools.partial(
        semantic_segmentation_metrics_function,
        target_is_onehot=self.dataset_meta_data.get('target_is_onehot', False),
        metrics=_SEMANTIC_SEGMENTATION_METRICS)

  def loss_function(self,
                    logits: jnp.ndarray,
                    batch: Batch,
                    model_params: Optional[jnp.ndarray] = None) -> float:
    """Returns softmax cross entropy loss with an L2 penalty on the weights.

    Args:
      logits: Output of model in shape [batch, length, num_classes].
      batch: Batch of data that has 'label' and optionally 'batch_mask'.
      model_params: Parameters of the model, for optionally applying
        regularization.

    Returns:
      Total loss.
    """
    weights = batch.get('batch_mask')

    if self.dataset_meta_data.get('target_is_onehot', False):
      one_hot_targets = batch['label']
    else:
      one_hot_targets = common_utils.onehot(batch['label'], logits.shape[-1])

    sof_ce_loss = model_utils.weighted_softmax_cross_entropy(
        logits,
        one_hot_targets,
        weights,
        label_smoothing=self.config.get('label_smoothing'),
        label_weights=self.get_label_weights())
    if self.config.get('l2_decay_factor') is None:
      total_loss = sof_ce_loss
    else:
      l2_loss = model_utils.l2_regularization(model_params)
      total_loss = sof_ce_loss + 0.5 * self.config.l2_decay_factor * l2_loss
    return total_loss

  def get_label_weights(self) -> jnp.ndarray:
    """Returns labels' weights to be used for computing weighted loss.

    This can used for weighting the loss terms based on the amount of available
    data for each class, when we have un-balances data for different classes.
    """
    if not self.config.get('class_rebalancing_factor'):
      return None
    if 'class_proportions' not in self.dataset_meta_data:
      raise ValueError(
          'When `class_rebalancing_factor` is nonzero, `class_proportions` must'
          ' be provided in `dataset_meta_data`.')
    w = self.config.get('class_rebalancing_factor')
    assert 0.0 <= w <= 1.0, '`class_rebalancing_factor` must be in [0.0, 1.0]'
    proportions = self.dataset_meta_data['class_proportions']
    proportions = np.maximum(proportions / np.sum(proportions), 1e-8)
    # Interpolate between no rebalancing (w==0.0) and full reweighting (w==1.0):
    proportions = w * proportions + (1.0 - w)
    weights = 1.0 / proportions
    weights /= np.sum(weights)  # Normalize so weights sum to 1.
    weights *= len(weights)  # Scale so weights sum to num_classes.
    return weights

  def get_global_metrics_fn(self) -> GlobalMetricFn:
    """Returns a callable metric function for global metrics.

      The return function implements metrics that require the prediction for the
      entire test/validation dataset in one place and has the following API:
        ```global_metrics_fn(all_confusion_mats, dataset_metadata)```
      If return None, no global metrics will be computed.
    """
    return global_metrics_fn

  def build_flax_model(self):
    raise NotImplementedError('Subclasses must implement build_flax_model().')

  def default_flax_model_config(self):
    """Default config for the flax model that is built in `build_flax_model`.

    This function in particular serves the testing functions and supposed to
    provide config tha are passed to the flax_model when it's build in
    `build_flax_model` function, e.g., `model_dtype_str`.
    """
    raise NotImplementedError(
        'Subclasses must implement default_flax_model_config().')


class MlpBlock(nn.Module):
  """Transformer MLP / feed-forward block."""

  mlp_dim: int
  out_dim: Optional[int] = None
  dropout_rate: float = 0.1
  use_bias: bool = True
  kernel_init: Initializer = nn.initializers.xavier_uniform()
  bias_init: Initializer = nn.initializers.normal(stddev=1e-6)
  activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
  precision: Optional[jax.lax.Precision] = None
  dtype: jnp.ndarray = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, deterministic: bool):
    """Applies Transformer MlpBlock module."""
    actual_out_dim = inputs.shape[-1] if self.out_dim is None else self.out_dim
    x = nn.Dense(
        self.mlp_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            inputs)
    x = self.activation_fn(x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=deterministic)
    output = nn.Dense(
        actual_out_dim,
        dtype=self.dtype,
        use_bias=self.use_bias,
        kernel_init=self.kernel_init,
        bias_init=self.bias_init,
        precision=self.precision)(
            x)
    output = nn.Dropout(rate=self.dropout_rate)(
        output, deterministic=deterministic)
    return output


class AddPositionEmbs(nn.Module):
  """Adds learned positional embeddings to the inputs.

  Attributes:
    posemb_init: Positional embedding initializer.

  Returns:
    Output in shape `[bs, timesteps, in_dim]`.
  """
  posemb_init: Initializer = nn.initializers.normal(stddev=0.02)  # From BERT.

  @nn.compact
  def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
    # Inputs.shape is (batch_size, seq_len, emb_dim).
    assert inputs.ndim == 3, ('Number of dimensions should be 3,'
                              ' but it is: %d' % inputs.ndim)
    pos_emb_shape = (1, inputs.shape[1], inputs.shape[2])
    pe = self.param('pos_embedding', self.posemb_init, pos_emb_shape,
                    inputs.dtype)
    return inputs + pe


class Encoder1DBlock(nn.Module):
  """Transformer encoder layer.

  Attributes:
    mlp_dim: Dimension of the mlp on top of attention block.
    num_heads: Number of self-attention heads.
    dtype: The dtype of the computation (default: float32).
    dropout_rate: Dropout rate.
    attention_dropout_rate: Dropout for attention heads.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value.

  Returns:
    output after transformer encoder block.
  """
  mlp_dim: int
  num_heads: int
  dtype: Any = jnp.float32
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0

  def get_stochastic_depth_mask(self, x: jnp.ndarray,
                                deterministic: bool) -> jnp.ndarray:
    """Generate the stochastic depth mask in order to apply layer-drop.

    Args:
      x: Input tensor.
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.

    Returns:
      Stochastic depth mask.
    """
    if not deterministic and self.stochastic_depth:
      shape = (x.shape[0],) + (1,) * (x.ndim - 1)
      return jax.random.bernoulli(
          self.make_rng('dropout'), self.stochastic_depth, shape)
    else:
      return 0.0

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
    """Applies Encoder1DBlock module.

    Args:
      inputs: Input data.
      deterministic: Deterministic or not (to apply dropout).

    Returns:
      Output after transformer encoder block.
    """
    # Attention block.
    assert inputs.ndim == 3
    x = nn.LayerNorm(dtype=self.dtype)(inputs)
    x = nn.MultiHeadDotProductAttention(
        num_heads=self.num_heads,
        dtype=self.dtype,
        kernel_init=nn.initializers.xavier_uniform(),
        broadcast_dropout=False,
        deterministic=deterministic,
        dropout_rate=self.attention_dropout_rate)(x, x)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic)
    x = x * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + inputs

    # Attention Layer MLP block.
    y = nn.LayerNorm(dtype=self.dtype)(x)
    y = MlpBlock(
        mlp_dim=self.mlp_dim,
        dtype=self.dtype,
        dropout_rate=self.dropout_rate,
        activation_fn=nn.gelu,
        kernel_init=nn.initializers.xavier_uniform(),
        bias_init=nn.initializers.normal(stddev=1e-6))(
            y, deterministic=deterministic)

    return y * (1.0 - self.get_stochastic_depth_mask(x, deterministic)) + x


class Encoder(nn.Module):
  """Transformer Encoder.

  Attributes:
    num_layers: Number of layers.
    mlp_dim: Dimension of the mlp on top of attention block.
    inputs_positions: Input subsequence positions for packed examples.
    dropout_rate: Dropout rate.
    stochastic_depth: probability of dropping a layer linearly grows from 0 to
      the provided value. Our implementation of stochastic depth follows timm
      library, which does per-example layer dropping and uses independent
      dropping patterns for each skip-connection.
    dtype: Dtype of activations.
  """
  num_layers: int
  mlp_dim: int
  num_heads: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  stochastic_depth: float = 0.0
  dtype: Any = jnp.float32

  @nn.compact
  def __call__(self, inputs: jnp.ndarray, *, train: bool = False):
    """Applies Transformer model on the inputs."""

    assert inputs.ndim == 3  # Shape is `[batch, len, emb]`.
    dtype = jax.dtypes.canonicalize_dtype(self.dtype)

    x = AddPositionEmbs(
        posemb_init=nn.initializers.normal(stddev=0.02),  # from BERT.
        name='posembed_input')(
            inputs)
    x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)

    # Input Encoder.
    for lyr in range(self.num_layers):
      x = Encoder1DBlock(
          mlp_dim=self.mlp_dim,
          num_heads=self.num_heads,
          dropout_rate=self.dropout_rate,
          attention_dropout_rate=self.attention_dropout_rate,
          stochastic_depth=(lyr / max(self.num_layers - 1, 1)) *
          self.stochastic_depth,
          name=f'encoderblock_{lyr}',
          dtype=dtype)(
              x, deterministic=not train)
    encoded = nn.LayerNorm(name='encoder_norm')(x)
    return encoded


class ViTBackbone(nn.Module):
  """Vision Transformer model backbone (everything except the head)."""

  mlp_dim: int
  num_layers: int
  num_heads: int
  patches: ml_collections.ConfigDict
  hidden_size: int
  dropout_rate: float = 0.1
  attention_dropout_rate: float = 0.1
  classifier: str = 'gap'

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool):

    n, h, w, c = x.shape
    fh, fw = self.patches.size
    x = nn.Conv(
        self.hidden_size, (fh, fw),
        strides=(fh, fw),
        padding='VALID',
        name='embedding')(
            x)

    n, h, w, c = x.shape
    x = jnp.reshape(x, [n, h * w, c])

    # If we want to add a class token, add it here.
    if self.classifier == 'token':
      cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
      cls = jnp.tile(cls, [n, 1, 1])
      x = jnp.concatenate([cls, x], axis=1)

    x = Encoder(
        mlp_dim=self.mlp_dim,
        num_layers=self.num_layers,
        num_heads=self.num_heads,
        dropout_rate=self.dropout_rate,
        attention_dropout_rate=self.attention_dropout_rate,
        name='Transformer')(
            x, train=train)
    return x


class SegVit(nn.Module):
  """Segmentation model with ViT backbone and decoder."""

  patches: ml_collections.ConfigDict
  backbone_configs: ml_collections.ConfigDict
  decoder_configs: ml_collections.ConfigDict
  num_classes: int

  @nn.compact
  def __call__(self, x: jnp.ndarray, *, train: bool, debug: bool = False):

    input_shape = x.shape
    b, h, w, _ = input_shape

    fh, fw = self.patches.size
    gh, gw = h // fh, w // fw

    if self.backbone_configs.type == 'vit':
      x = ViTBackbone(
          mlp_dim=self.backbone_configs.mlp_dim,
          num_layers=self.backbone_configs.num_layers,
          num_heads=self.backbone_configs.num_heads,
          patches=self.patches,
          hidden_size=self.backbone_configs.hidden_size,
          dropout_rate=self.backbone_configs.dropout_rate,
          attention_dropout_rate=self.backbone_configs.attention_dropout_rate,
          classifier='gap',
          name='backbone')(
              x, train=train)
    else:
      raise ValueError(f'Unknown backbone: {self.backbone_configs.type}.')

    output_projection = nn.Dense(
        self.num_classes,
        kernel_init=nn.initializers.zeros,
        name='output_projection')

    if self.decoder_configs.type == 'linear':
      # Linear head only, like Segmenter baseline:
      # https://arxiv.org/abs/2105.05633
      x = jnp.reshape(x, [b, gh, gw, -1])
      x = output_projection(x)
      # Resize bilinearly:
      x = jax.image.resize(x, [b, h, w, x.shape[-1]], 'linear')
    else:
      raise ValueError(
          f'Decoder type {self.decoder_configs.type} is not defined.')

    assert input_shape[:-1] == x.shape[:-1], (
        'Input and output shapes do not match: %d vs. %d.', input_shape[:-1],
        x.shape[:-1])

    return x


class SegVitModel(SegmentationModel):
  """ViT model for segmentation task."""

  def build_flax_model(self) -> nn.Module:
    return SegVit(
        patches=self.config.model.patches,
        backbone_configs=self.config.model.backbone,
        decoder_configs=self.config.model.decoder,
        num_classes=self.dataset_meta_data['num_classes'])

  def default_flax_model_config(self) -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()
    config.model.backbone = ml_collections.ConfigDict()
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [4, 4]
    config.model.backbone = ml_collections.ConfigDict()
    config.model.backbone.type = 'vit'
    config.model.backbone.attention_dropout_rate = 0.0
    config.model.backbone.dropout_rate = 0.0
    config.model.backbone.hidden_size = 16
    config.model.backbone.mlp_dim = 32
    config.model.backbone.num_heads = 2
    config.model.backbone.num_layers = 1
    config.model.decoder = ml_collections.ConfigDict()
    config.model.decoder.type = 'linear'
    config.model.decoder.representation_size = 16
    config.data_dtype_str = 'float32'
    return config

  def init_backbone_from_train_state(
      self, train_state: Any, restored_train_state: Any,
      restored_model_cfg: ml_collections.ConfigDict) -> Any:
    """Updates the train_state with data from restored_train_state.

    Here, we do some surgery and replace parts of the parameters/model_state
    in the train_state with some parameters/model_state from the
    pretrained_train_state.

    Note that the grid shape of our model can be different from that of the
    pretrained model (position embeddings are adapted by interpolation).

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.

    Returns:
      Updated train_state.
    """
    # Get grid sizes of target model:
    gs_segvit = [
        self.config.dataset_configs.target_size[0] //
        self.config.model.patches.size[0],
        self.config.dataset_configs.target_size[1] //
        self.config.model.patches.size[1]
    ]

    # Get grid sizes of restored model:
    if 'patches' in restored_model_cfg:
      restored_patches_cfg = restored_model_cfg.patches
    else:
      restored_patches_cfg = restored_model_cfg.stem_configs.patches
    if 'grid' in restored_patches_cfg:
      gs_vit = restored_patches_cfg.grid
    else:
      init_dset_meta = self.config.model.backbone.init_from.dataset_meta_data
      gs_vit = [
          init_dset_meta['input_shape'][1] // restored_patches_cfg.size[0],
          init_dset_meta['input_shape'][2] // restored_patches_cfg.size[1],
      ]

    backbone = train_state.optimizer.target.params['backbone']
    restored_param = restored_train_state.optimizer['target']['params']
    for m_key, m_params in restored_param.items():
      # load parameters for embedding (CNN at stem)
      if m_key in ['embedding']:
        backbone[m_key] = m_params

      # load parameters for Transformer encoder
      if m_key == 'Transformer':
        for tm_key, tm_params in m_params.items():
          if tm_key == 'posembed_input':  # might need resolution change
            vit_posemb = m_params['posembed_input']['pos_embedding']
            segvit_posemb = backbone[m_key]['posembed_input']['pos_embedding']
            if vit_posemb.shape != segvit_posemb.shape:
              # rescale the grid of pos, embeddings: param shape is (1,N,768)
              segvit_ntok = segvit_posemb.shape[1]
              if restored_model_cfg.classifier == 'token':
                # the first token is the CLS token
                vit_posemb = vit_posemb[0, 1:]
              else:
                vit_posemb = vit_posemb[0]
              logging.info('Resized variant: %s to %s', vit_posemb.shape,
                           segvit_posemb.shape)
              assert np.prod(gs_vit) == vit_posemb.shape[0]
              assert np.prod(gs_segvit) == segvit_ntok
              if gs_vit != gs_segvit:  # we need resolution change
                logging.info('Grid-size from %s to %s', gs_vit, gs_segvit)
                vit_posemb_grid = vit_posemb.reshape(gs_vit + [-1])
                zoom = (gs_segvit[0] / gs_vit[0], gs_segvit[1] / gs_vit[1], 1)
                vit_posemb_grid = scipy.ndimage.zoom(
                    vit_posemb_grid, zoom, order=1)
                vit_posemb = vit_posemb_grid.reshape(1, np.prod(gs_segvit), -1)
              else:  # just the cls token was extra and we are now fine
                vit_posemb = np.expand_dims(vit_posemb, axis=0)
            backbone[m_key][tm_key]['pos_embedding'] = vit_posemb
          else:  # other parameters of the Transformer encoder
            backbone[m_key][tm_key] = tm_params

    return train_state
