"""
Custom models which allow for model inheritance
"""

from scenic.model_lib.base_models.segmentation_model import SegmentationModel
from uncertainty_baselines.models.segmenter import SegVit
import ml_collections

class SegmenterSegmentationModel(SegmentationModel):
  """Segmenter model for segmentation task."""

  def build_flax_model(self):
    """
    return SegVit(
        num_classes=self.dataset_meta_data['num_classes'],
        patches=self.config.model.get('patches', {}),
        backbone_configs=self.config.model.get('backbone_configs', {}),
        decoder_configs=self.config.model.get('decoder_configs', {}))
    """
    return SegVit(
        num_classes=self.num_classes,
        patches=self.config.patches,
        backbone_configs=self.config.backbone_configs,
        decoder_configs=self.config.model.decoder_configs)

  def default_flax_model_config(self):
    raise NotImplementedError()
    """
    return ml_collections.ConfigDict({
        'model':
            dict(
                num_classes=19,
                patches=patches,
                block_size=(64, 128, 256, 512),
                data_dtype_str='float32')
    })
    """

