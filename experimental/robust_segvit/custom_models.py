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

"""Model wrapper to allow for model inheritance."""
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Union
from absl import logging
import flax
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.train_lib_deprecated import train_utils
import scipy
from uncertainty_metrics import get_pacc_cert  # local file import from experimental.robust_segvit
from uncertainty_metrics import get_pavpu  # local file import from experimental.robust_segvit
from uncertainty_metrics import get_puncert_inacc  # local file import from experimental.robust_segvit
from uncertainty_baselines.models.segmenter import SegVit
from uncertainty_baselines.models.segmenter_be import SegVitBE
from uncertainty_baselines.models.segmenter_gp import SegVitGP
from uncertainty_baselines.models.segmenter_heteroscedastic import SegVitHet

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]
GlobalMetricFn = Callable[[List[jnp.ndarray]], Dict[str, float]]


class UBSegmentationModel(SegmentationModel):
  """Segmentation Model class in UB."""

  def default_flax_model_config(self):
    raise NotImplementedError()

  def init_backbone_from_train_state(
      self,
      train_state: train_utils.TrainState,
      restored_train_state: Mapping[str, Any],
      model_cfg: ml_collections.ConfigDict,
      restored_model_cfg: ml_collections.ConfigDict,
      ckpt_prefix_path: Optional[List[str]] = None,
      model_prefix_path: Optional[List[str]] = None,
      name_mapping: Optional[Mapping[str, str]] = None,
      skip_regex: Optional[str] = None) -> train_utils.TrainState:
    """Updates the train_state with data from pretrain_state.

    Args:
      train_state: A raw TrainState for the model.
      restored_train_state: A TrainState that is loaded with parameters/state of
        a pretrained model.
      model_cfg: Configuration of the model.
      restored_model_cfg: Configuration of the model from which the
        restored_train_state come from. Usually used for some asserts.
      ckpt_prefix_path: Prefix to restored model parameters.
      model_prefix_path: Prefix to the parameters to replace in the subtree
        model.
      name_mapping: Mapping from parameter names of checkpoint to this model.
      skip_regex: If there is a parameter whose parent keys match the regex, the
        parameter will not be replaced from pretrain_state.

    Returns:
      Updated train_state.
    """

    # Model input shape
    input_shape = self.config.model.get('input_shape',
                                        self.config.dataset_configs.target_size)

    # Get grid sizes of target model:
    gs_segvit = [
        input_shape[0] //
        self.config.model.patches.size[0],
        input_shape[1] //
        self.config.model.patches.size[1]
    ]
    # Find size of positional embeddings (grid size) if given as input
    # otherwise we will take the will use the model checkpoint to estimate thiis
    if ('patches' in restored_model_cfg) or ('stem_configs'
                                             in restored_model_cfg):
      if 'patches' in restored_model_cfg:
        restored_patches_cfg = restored_model_cfg.patches
      else:
        restored_patches_cfg = restored_model_cfg.stem_configs.patches
      gs_vit = restored_patches_cfg.grid
    else:
      gs_vit = None

    name_mapping = name_mapping or {}

    restored_params = restored_train_state['optimizer']['target']
    restored_model_state = restored_train_state.get('model_state')

    model_params = train_state.optimizer.target
    model_params = _replace_dict(model_params, restored_params,
                                 model_cfg,
                                 restored_model_cfg, gs_vit, gs_segvit,
                                 ckpt_prefix_path, model_prefix_path,
                                 name_mapping, skip_regex)

    new_optimizer = train_state.optimizer.replace(target=model_params)
    train_state = train_state.replace(  # pytype: disable=attribute-error
        optimizer=new_optimizer)

    if (restored_model_state is not None and
        train_state.model_state is not None and train_state.model_state):
      if model_prefix_path:
        # Insert model prefix after 'batch_stats'.
        model_prefix_path = ['batch_stats'] + model_prefix_path
        if 'batch_stats' in restored_model_state:
          ckpt_prefix_path = ckpt_prefix_path or []
          ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
      elif 'batch_stats' not in restored_model_state:  # Backward compatibility.
        model_prefix_path = ['batch_stats']
      if ckpt_prefix_path and ckpt_prefix_path[0] != 'batch_stats':
        ckpt_prefix_path = ['batch_stats'] + ckpt_prefix_path
      model_state = _replace_dict(train_state.model_state, restored_model_state,
                                  model_cfg,
                                  restored_model_cfg, gs_vit, gs_segvit,
                                  ckpt_prefix_path, model_prefix_path,
                                  name_mapping, skip_regex)
      train_state = train_state.replace(  # pytype: disable=attribute-error
          model_state=model_state)
    return train_state

  def get_global_unc_metrics_fn(self) -> GlobalMetricFn:
    """Returns a callable metric function for global metrics.

      The return function implements metrics that require the prediction for the
      entire test/validation dataset in one place and has the following API:
        ```global_metrics_fn(all_unc_confusion_mats, dataset_metadata)```
      If return None, no global metrics will be computed.
    """
    return global_unc_metrics_fn


class SegmenterSegmentationModel(UBSegmentationModel):
  """Segmenter model."""

  def build_flax_model(self):
    return SegVit(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.patches,
        backbone_configs=self.config.model.backbone,
        decoder_configs=self.config.model.decoder)


class SegmenterBESegmentationModel(UBSegmentationModel):
  """Batch Ensemble segmenter model."""

  def build_flax_model(self):
    return SegVitBE(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.patches,
        backbone_configs=self.config.model.backbone,
        decoder_configs=self.config.model.decoder)


class SegmenterGPSegmentationModel(UBSegmentationModel):
  """Segmenter GP model."""

  def build_flax_model(self):
    return SegVitGP(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.patches,
        backbone_configs=self.config.model.backbone,
        decoder_configs=self.config.model.decoder)


class SegmenterHetSegmentationModel(UBSegmentationModel):
  """Segmenter Het model."""

  def build_flax_model(self):
    return SegVitHetKey(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.patches,
        backbone_configs=self.config.model.backbone,
        decoder_configs=self.config.model.decoder)


class SegVitHetKey(SegVitHet):
  """Segvit Het model with custom rng initialization.

  Edited from:
  uncertainty_baselines/google/models/t5_heteroscedastic.py;l=132;rcl=435647830
  """

  def apply(self, *args, **kwargs):
    if self.decoder_configs.type != 'het':
      return super().apply(*args, **kwargs)
    rngs = kwargs.get('rngs', None)

    # For evaluation, we use a constant seed.
    if rngs is None:
      rng = jax.random.PRNGKey(0)
      keys = ['diag_noise_samples', 'standard_norm_noise_samples', 'params']
      rngs = dict(zip(keys, jax.random.split(rng, 3)))
    else:

      all_keys = [
          'dropout', 'params', 'diag_noise_samples',
          'standard_norm_noise_samples'
      ]

      missing_keys = []
      for key_ in all_keys:
        if key_ not in rngs:
          missing_keys.append(key_)

      if 'dropout' in missing_keys and 'params' in missing_keys:
        raise ValueError(
            'Missing `dropout` and  `params` rng for the network. rngs: {}'
            .format(rngs.keys()))

      if 'dropout' in rngs:
        split_rng = 'dropout'
      elif 'params' in rngs:
        split_rng = 'params'
      else:
        raise ValueError(
            'Missing `dropout` and `params` rng for the network. keys: {}'
            .format(rngs.keys()))

      rng_updates = dict(
          zip([split_rng, *missing_keys],
              jax.random.split(rngs[split_rng], len(missing_keys) + 1)))
      rngs.update(rng_updates)

    kwargs['rngs'] = rngs
    return super().apply(*args, **kwargs)


def _replace_dict(model: PyTree,
                  restored: PyTree,
                  model_cfg: ml_collections.ConfigDict,
                  restored_model_cfg: ml_collections.ConfigDict,
                  gs_vit: Optional[List[Optional[int]]] = None,
                  gs_segvit: Optional[List[Optional[int]]] = None,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint.

  Includes changes to facilitate loading of pretrained variables from an
  encoder w a token classifier.

  Args:
    model: model parameters
    restored: restored model parameters
    model_cfg: Configuration of the model.
    restored_model_cfg: Configuration of the model from which the
      restored_train_state come from. Usually used for some asserts.
    gs_vit: token size of restored model
    gs_segvit: token size of new model
    ckpt_prefix_path: Prefix to restored model parameters.
    model_prefix_path: Prefix to the parameters to replace in the subtree model.
    name_mapping: Mapping from parameter names of checkpoint to this model.
    skip_regex: If there is a parameter whose parent keys match the regex, the
      parameter will not be replaced from pretrain_state.

  Returns:
    model state with updated model parameters
  """

  model = flax.core.unfreeze(model)  # pytype: disable=wrong-arg-types
  restored = flax.core.unfreeze(restored)  # pytype: disable=wrong-arg-types

  if ckpt_prefix_path:
    for p in ckpt_prefix_path:
      restored = restored[p]

  if model_prefix_path:
    for p in reversed(model_prefix_path):
      restored = {p: restored}

  # Flatten nested parameters to a dict of str -> tensor. Keys are tuples
  # from the path in the nested dictionary to the specific tensor. E.g.,
  # {'a1': {'b1': t1, 'b2': t2}, 'a2': t3}
  # -> {('a1', 'b1'): t1, ('a1', 'b2'): t2, ('a2',): t3}.
  restored_flat = flax.traverse_util.flatten_dict(
      dict(restored), keep_empty_nodes=True)
  model_flat = flax.traverse_util.flatten_dict(
      dict(model), keep_empty_nodes=True)

  for m_key, m_params in restored_flat.items():
    # pytype: disable=attribute-error
    for name, to_replace in name_mapping.items():
      m_key = tuple(to_replace if k == name else k for k in m_key)
    # pytype: enable=attribute-error
    m_key_str = '/'.join(m_key)
    if m_key not in model_flat:
      logging.warning('%s in checkpoint doesn\'t exist in model. Skip.',
                      m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)

    # resize positional embeddings given token
    if 'posembed_input' in m_key:  # might need resolution change
      # the backbone should be pose segmenter
      # vit_posemb = m_params['posembed_input']['pos_embedding']
      vit_posemb = m_params
      # segvit_posemb = backbone[m_key]['posembed_input']['pos_embedding']
      segvit_posemb = model_flat[m_key]
      if vit_posemb.shape != segvit_posemb.shape:
        # rescale the grid of pos, embeddings: param shape is (1,N,hidden_size)
        segvit_ntok = segvit_posemb.shape[1]
        if restored_model_cfg.classifier == 'token':
          # the first token is the CLS token
          vit_cls_token = vit_posemb[0, 0]
          vit_posemb = vit_posemb[0, 1:]
        else:
          vit_posemb = vit_posemb[0]
        logging.info('Resized variant: %s to %s', vit_posemb.shape,
                     segvit_posemb.shape)
        if gs_vit is None:
          gs_vit = [
              int(np.sqrt(vit_posemb.shape[0])),
              int(np.sqrt(vit_posemb.shape[0]))
          ]
        assert np.prod(gs_vit) == vit_posemb.shape[0]

        if model_cfg.model.backbone.classifier == 'gap':
          assert np.prod(gs_segvit) == segvit_ntok
        elif model_cfg.model.backbone.classifier == 'token':
          assert np.prod(gs_segvit) == segvit_ntok - 1
        else:
          raise NotImplementedError('')

        if gs_vit != gs_segvit:  # we need resolution change
          logging.info('Grid-size from %s to %s', gs_vit, gs_segvit)
          vit_posemb_grid = vit_posemb.reshape(gs_vit + [-1])
          zoom = (gs_segvit[0] / gs_vit[0], gs_segvit[1] / gs_vit[1], 1)
          vit_posemb_grid = scipy.ndimage.zoom(vit_posemb_grid, zoom, order=1)
          vit_posemb = vit_posemb_grid.reshape(1, np.prod(gs_segvit), -1)
        else:  # just the cls token was extra and we are now fine
          vit_posemb = np.expand_dims(vit_posemb, axis=0)

        # Initialize the cls token:
        if model_cfg.model.backbone.classifier == 'token' and restored_model_cfg.classifier == 'token':
          segvit_cls_token = segvit_posemb[0, 0, :]
          if restored_model_cfg.token_init:
            segvit_cls_token = vit_cls_token
          segvit_cls_token = np.expand_dims(
              np.expand_dims(segvit_cls_token, axis=0), axis=0)
          m_params = np.hstack([segvit_cls_token, vit_posemb])
        else:
          m_params = vit_posemb

    # TODO(kellybuchanan): add option to load weights from upstream vit_be
    # to downstream vit_deterministic
    if model_flat[m_key].shape != m_params.shape:
      logging.info(
          'Size mismatch between upstream %s and downstream %s params for %s',
          m_params.shape, model_flat[m_key].shape, m_key)

      # upstream model is deterministic and downstream model is be:
      if model_cfg.model.backbone.type == 'vit_be' and restored_model_cfg.backbone_type == 'vit':
        ens_size = model_cfg.model.backbone.ens_size
        m_params = jnp.tile(m_params, (ens_size, 1))

      else:
        raise NotImplementedError(
            'Missing support to load weights from upstream model type')

    assert model_flat[m_key].shape == m_params.shape
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))


def global_unc_metrics_fn(
    all_unc_confusion_mats: List[jnp.ndarray]) -> Dict[str, float]:
  """Returns a dict with global (whole-dataset) metrics."""
  # Compute uncertainty scores from list of uncertainty confusion matrices:
  assert isinstance(all_unc_confusion_mats, list)  # List of eval batches.
  cm = np.sum(all_unc_confusion_mats, axis=0)  # Sum over eval batches.

  if cm.ndim == 2:  # [batch_size, 4]
    pass
  elif cm.ndim == 3:  # [num_devices, batch_size per device, 4]
    cm = np.sum(cm, axis=0)  # sum over devices

  assert cm.ndim == 2, ('Expecting uncertainty confusion matrix to have shape '
                        '[batch_size, 4], got '
                        f'{cm.shape}.')
  # calculate metrics
  cm = np.sum(cm, axis=0)  # Sum over batch dimension.

  pavpu = get_pavpu(cm)
  pacc_cert = get_pacc_cert(cm)
  puncert_inacc = get_puncert_inacc(cm)

  metrics_dict = {
      'pavpu': float(pavpu),
      'pacc_cert': float(pacc_cert),
      'puncert_inacc': float(puncert_inacc)
  }

  return metrics_dict
