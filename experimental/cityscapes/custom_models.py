"""
Custom models which allow for model inheritance
"""
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, List
from absl import logging

from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from uncertainty_baselines.models.segmenter import SegVit
import ml_collections
import numpy as np
import scipy

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
          self, train_state: Any, restored_train_state: Any,
          restored_model_cfg: ml_collections.ConfigDict) -> Any:
      """
      Edited from scenic.
      Updates the train_state with data from restored_train_state.
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
          #init_dset_meta = self.config.model.backbone.init_from.dataset_meta_data
          #gs_vit = [
          #    init_dset_meta['input_shape'][1] // restored_patches_cfg.size[0],
          #    init_dset_meta['input_shape'][2] // restored_patches_cfg.size[1],
          #]

      #TODO(kellybuchanan): check issue where FrozenDict is immutable.

      #backbone = train_state.optimizer.target.params['backbone']
      #restored_param = restored_train_state.optimizer['target']['params']

      backbone = train_state.optimizer.target['backbone']
      restored_param = restored_train_state.optimizer['target']
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
