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

"""Uncertainty baseline training models."""

import warnings

# When adding a new model, also add to models.py for easier user access.
from uncertainty_baselines.models.criteo_mlp import create_model as CriteoMlpBuilder
from uncertainty_baselines.models.genomics_cnn import create_model as GenomicsCNNBuilder
from uncertainty_baselines.models.models import get
from uncertainty_baselines.models.resnet20 import create_model as ResNet20Builder
from uncertainty_baselines.models.resnet50 import create_model as ResNet50Builder
from uncertainty_baselines.models.textcnn import create_model as TextCNNBuilder
from uncertainty_baselines.models.wide_resnet import create_model as WideResNetBuilder

try:
  from uncertainty_baselines.models.bert import create_model as BERTBuilder  # pylint: disable=g-import-not-at-top
except ImportError as e:
  warnings.warn(f'Skipped due to ImportError: {e}')
