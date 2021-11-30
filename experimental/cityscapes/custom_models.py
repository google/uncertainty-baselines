"""
Custom models which allow for model inheritance
"""
import re
from typing import Any, Mapping, Optional, Tuple, List, Union

import flax
import ml_collections
import numpy as np
import scipy
from absl import logging

from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from scenic.train_lib import train_utils
from uncertainty_baselines.models.segmenter import SegVit

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


class SegmenterSegmentationModel(SegmentationModel):
  """Segmenter model for segmentation task."""

  def build_flax_model(self):
    return SegVit(
      num_classes=self.dataset_meta_data['num_classes'],
      patches=self.config.patches,
      backbone_configs=self.config.backbone_configs,
      decoder_configs=self.config.decoder_configs)

  def default_flax_model_config(self):
    raise NotImplementedError()

  def init_backbone_from_train_state(
          self,
          train_state: train_utils.TrainState,
          restored_train_state: Mapping[str, Any],
          restored_model_cfg: ml_collections.ConfigDict,
          ckpt_prefix_path: Optional[List[str]] = None,
          model_prefix_path: Optional[List[str]] = None,
          name_mapping: Optional[Mapping[str, str]] = None,
          skip_regex: Optional[str] = None) -> train_utils.TrainState:
      """Updates the train_state with data from pretrain_state.

      Args:
        train_state: A raw TrainState for the model.
        restored_train_state: A TrainState that is loaded with parameters/state of
          a  pretrained model.
        restored_model_cfg: Configuration of the model from which the
         restored_train_state come from. Usually used for some asserts.
        ckpt_prefix_path: Prefix to restored model parameters.
        model_prefix_path: Prefix to the parameters to replace in the subtree model.
        name_mapping: Mapping from parameter names of checkpoint to this model.
        skip_regex: If there is a parameter whose parent keys match the regex,
          the parameter will not be replaced from pretrain_state.

      Returns:
        Updated train_state.
      """
      # ---------------------------------
      # Get grid sizes of target model:
      gs_segvit = [
          self.config.dataset_configs.target_size[0] //
          self.config.patches.size[0],
          self.config.dataset_configs.target_size[1] //
          self.config.patches.size[1]
      ]

      # Get grid sizes of restored model:
      if 'patches' in restored_model_cfg:
          restored_patches_cfg = restored_model_cfg.patches
      else:
          restored_patches_cfg = restored_model_cfg.stem_configs.patches
      if 'grid' in restored_patches_cfg:
          gs_vit = restored_patches_cfg.grid
      else:
          raise NotImplementedError("")

          # init_dset_meta = self.config.model.backbone.init_from.dataset_meta_data
          # gs_vit = [
          #    init_dset_meta['input_shape'][1] // restored_patches_cfg.size[0],
          #    init_dset_meta['input_shape'][2] // restored_patches_cfg.size[1],
          # ]

      # ---------------------------------
      name_mapping = name_mapping or {}

      # converts pre-linen which doesn't apply here
      # (restored_params,
      # restored_model_state) = get_params_and_model_state_dict(restored_train_state)
      restored_params = restored_train_state['optimizer']['target']
      restored_model_state = restored_train_state.get('model_state')

      model_params = train_state.optimizer.target
      model_params = _replace_dict(model_params,
                                   restored_params,
                                   restored_model_cfg,
                                   gs_vit,
                                   gs_segvit,
                                   ckpt_prefix_path,
                                   model_prefix_path,
                                   name_mapping,
                                   skip_regex)
      new_optimizer = train_state.optimizer.replace(
          target=model_params)
      train_state = train_state.replace(  # pytype: disable=attribute-error
          optimizer=new_optimizer)
      if (restored_model_state is not None and
              train_state.model_state is not None and
              train_state.model_state):
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
          model_state = _replace_dict(train_state.model_state,
                                      restored_model_state,
                                      restored_model_cfg,
                                      gs_vit,
                                      gs_segvit,
                                      ckpt_prefix_path,
                                      model_prefix_path,
                                      name_mapping,
                                      skip_regex)
          train_state = train_state.replace(  # pytype: disable=attribute-error
              model_state=model_state)
      return train_state


def _replace_dict(model: PyTree,
                  restored: PyTree,
                  restored_model_cfg: ml_collections,
                  gs_vit: Optional[Tuple] = None,
                  gs_segvit: Optional[Tuple] = None,
                  ckpt_prefix_path: Optional[List[str]] = None,
                  model_prefix_path: Optional[List[str]] = None,
                  name_mapping: Optional[Mapping[str, str]] = None,
                  skip_regex: Optional[str] = None) -> PyTree:
  """Replaces values in model dictionary with restored ones from checkpoint.

  Include changes to facilitate loading of pretrained variables
  from an encoder w a token classifier.
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
      logging.warning(
          '%s in checkpoint doesn\'t exist in model. Skip.', m_key_str)
      continue
    if skip_regex and re.findall(skip_regex, m_key_str):
      logging.info('Skip loading parameter %s.', m_key_str)
      continue
    logging.info('Loading %s from checkpoint into model', m_key_str)

    # fix if token
    if 'posembed_input' in m_key: # might need resolution change
        # the backbone should be pose segmenter
      # vit_posemb = m_params['posembed_input']['pos_embedding']
      vit_posemb = m_params
      # segvit_posemb = backbone[m_key]['posembed_input']['pos_embedding']
      segvit_posemb = model_flat[m_key]

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
          vit_posemb_grid = scipy.ndimage.zoom(vit_posemb_grid, zoom, order=1)
          vit_posemb = vit_posemb_grid.reshape(1, np.prod(gs_segvit), -1)
        else:  # just the cls token was extra and we are now fine
          vit_posemb = np.expand_dims(vit_posemb, axis=0)
        m_params = vit_posemb

    assert model_flat[m_key].shape == m_params.shape
    model_flat[m_key] = m_params

  return flax.core.freeze(flax.traverse_util.unflatten_dict(model_flat))

