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

"""Uncertainty baseline training models."""

from absl import logging
import tensorflow as tf

# ==============================================================================
# Add Vision Transformer, BERT, ed2.mimo, and PyTorch models to their
# corresponding try/except blocks below these main imports, otherwise you will
# break the external build.
# ==============================================================================
from uncertainty_baselines.models import efficientnet_utils
from uncertainty_baselines.models.criteo_mlp import criteo_mlp
from uncertainty_baselines.models.efficientnet import efficientnet
from uncertainty_baselines.models.efficientnet_batch_ensemble import efficientnet_batch_ensemble
from uncertainty_baselines.models.genomics_cnn import genomics_cnn
from uncertainty_baselines.models.models import get
from uncertainty_baselines.models.movielens import movielens
from uncertainty_baselines.models.mpnn import mpnn
from uncertainty_baselines.models.resnet20 import resnet20
from uncertainty_baselines.models.resnet50_batchensemble import resnet101_batchensemble
from uncertainty_baselines.models.resnet50_batchensemble import resnet50_batchensemble
from uncertainty_baselines.models.resnet50_batchensemble import resnet_batchensemble
from uncertainty_baselines.models.resnet50_deterministic import resnet50_deterministic
from uncertainty_baselines.models.resnet50_dropout import resnet50_dropout
from uncertainty_baselines.models.resnet50_het_mimo import resnet50_het_mimo
from uncertainty_baselines.models.resnet50_het_rank1 import resnet50_het_rank1
from uncertainty_baselines.models.resnet50_heteroscedastic import resnet50_heteroscedastic
from uncertainty_baselines.models.resnet50_hetsngp import resnet50_hetsngp
from uncertainty_baselines.models.resnet50_hetsngp import resnet50_hetsngp_add_last_layer
from uncertainty_baselines.models.resnet50_radial import resnet50_radial
from uncertainty_baselines.models.resnet50_rank1 import resnet50_rank1
from uncertainty_baselines.models.resnet50_sngp import resnet50_sngp
from uncertainty_baselines.models.resnet50_sngp import resnet50_sngp_add_last_layer
from uncertainty_baselines.models.resnet50_sngp_be import resnet50_sngp_be
from uncertainty_baselines.models.resnet50_variational import resnet50_variational
from uncertainty_baselines.models.textcnn import textcnn
from uncertainty_baselines.models.unet import unet
from uncertainty_baselines.models.wide_resnet import wide_resnet
from uncertainty_baselines.models.wide_resnet_batchensemble import wide_resnet_batchensemble
from uncertainty_baselines.models.wide_resnet_condconv import wide_resnet_condconv
from uncertainty_baselines.models.wide_resnet_dropout import wide_resnet_dropout
from uncertainty_baselines.models.wide_resnet_heteroscedastic import wide_resnet_heteroscedastic
from uncertainty_baselines.models.wide_resnet_hetsngp import wide_resnet_hetsngp
from uncertainty_baselines.models.wide_resnet_hyperbatchensemble import e_factory as hyperbatchensemble_e_factory
from uncertainty_baselines.models.wide_resnet_hyperbatchensemble import LambdaConfig as HyperBatchEnsembleLambdaConfig
from uncertainty_baselines.models.wide_resnet_hyperbatchensemble import wide_resnet_hyperbatchensemble
from uncertainty_baselines.models.wide_resnet_posterior_network import wide_resnet_posterior_network
from uncertainty_baselines.models.wide_resnet_rank1 import wide_resnet_rank1
from uncertainty_baselines.models.wide_resnet_sngp import wide_resnet_sngp
from uncertainty_baselines.models.wide_resnet_sngp_be import wide_resnet_sngp_be
from uncertainty_baselines.models.wide_resnet_variational import wide_resnet_variational
from uncertainty_baselines.models.resnet50_fsvi import ResNet50FSVI
# When adding a new model, also add to models.py for easier user access.

# pylint: disable=g-import-not-at-top
try:
  # Try to import ViT models.
  from uncertainty_baselines.models import vit_batchensemble
  from uncertainty_baselines.models.bit_resnet import bit_resnet
  from uncertainty_baselines.models.vit import vision_transformer
  from uncertainty_baselines.models.vit_batchensemble import PatchTransformerBE
  from uncertainty_baselines.models.vit_gp import vision_transformer_gp
  from uncertainty_baselines.models.vit_hetgp import vision_transformer_hetgp
  from uncertainty_baselines.models.vit_heteroscedastic import het_vision_transformer
except ImportError:
  logging.warning('Skipped ViT models due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning('Skipped ViT models due to NotFoundError.', exc_info=True)

# pylint: disable=g-import-not-at-top
try:
  # Try to import Segmenter models.
  from uncertainty_baselines.models.segmenter import segmenter_transformer
except ImportError:
  logging.warning('Skipped Segmenter models due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning('Skipped Segmenter models due to NotFoundError.',
                  exc_info=True)

try:
  # Try to import models depending on tensorflow_models.official.nlp.
  from uncertainty_baselines.models import bert
  from uncertainty_baselines.models.bert import bert_model
  from uncertainty_baselines.models import bert_dropout
  from uncertainty_baselines.models.bert_dropout import bert_dropout_model
  from uncertainty_baselines.models import bert_sngp
  from uncertainty_baselines.models.bert_sngp import bert_sngp_model
except ImportError:
  logging.warning('Skipped BERT models due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning('Skipped BERT models due to NotFoundError.', exc_info=True)

try:
  # Try to import models depending on edward2.experimental.mimo.
  from uncertainty_baselines.models.resnet50_mimo import resnet50_mimo
  from uncertainty_baselines.models.wide_resnet_mimo import wide_resnet_mimo
except ImportError:
  logging.warning('Skipped MIMO models due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning('Skipped MIMO models due to NotFoundError.', exc_info=True)

# This is necessary because we cannot depend on torch internally, so the torch
# model modules cannot be imported at all, so we cannot just wrap the imports in
# a try/except.
import_torch = True
if import_torch:
  try:
    from uncertainty_baselines.models.resnet50_torch import resnet50_torch
    from uncertainty_baselines.models.resnet50_dropout_torch import (
      resnet50_dropout_torch)
  except ImportError:
    logging.warning(
      'Skipped Torch models due to ImportError.', exc_info=True)
# pylint: enable=g-import-not-at-top
