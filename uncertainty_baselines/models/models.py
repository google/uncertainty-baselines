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

# Lint as: python3
"""Model getter utility."""

import json
import logging
from typing import List
from absl import logging

import tensorflow as tf

from uncertainty_baselines.models import criteo_mlp
from uncertainty_baselines.models import efficientnet
from uncertainty_baselines.models import efficientnet_batch_ensemble
from uncertainty_baselines.models import genomics_cnn
from uncertainty_baselines.models import movielens
from uncertainty_baselines.models import resnet20
from uncertainty_baselines.models import resnet50
from uncertainty_baselines.models import textcnn
from uncertainty_baselines.models import wide_resnet

try:
  from uncertainty_baselines.models import bert  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning(
      'Skipped bert model import due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning(
      'Skipped bert model import due to tf.errors.NotFoundError.',
      exc_info=True)


def get_model_names() -> List[str]:
  return [
      'bert',
      'criteo_mlp',
      'genomics_cnn',
      'movielens',
      'resnet20',
      'resnet50',
      'textcnn',
      'wide_resnet',
  ]


# TODO(znado): check into using Keras' deserialization functionality, similar to
# edward2/tensorflow/initializers.py line 772.
def get(model_name: str, **kwargs) -> tf.keras.Model:
  """Gets a model builder by name.

  Args:
    model_name: Name of the model builder class.
    **kwargs: passed to the model constructor.

  Returns:
    A tf.keras.Model.

  Raises:
    ValueError: If model_name is unrecognized.
  """
  logging.info(
      'Building model %s with additional kwargs:\n%s',
      model_name,
      json.dumps(kwargs, indent=2, sort_keys=True))
  if model_name not in get_model_names():
    raise ValueError('Unrecognized model type: {!r}'.format(model_name))

  if model_name == 'criteo_mlp':
    return criteo_mlp.criteo_mlp(**kwargs)
  if model_name == 'movielens':
    return movielens.movielens(**kwargs)
  if model_name == 'resnet20':
    return resnet20.resnet20(**kwargs)
  if model_name == 'resnet50':
    return resnet50.resnet50(**kwargs)
  if model_name == 'textcnn':
    return textcnn.textcnn(**kwargs)
  if model_name == 'bert':
    return bert.bert_model(**kwargs)
  if model_name == 'wide_resnet':
    return wide_resnet.wide_resnet(**kwargs)
  if model_name == 'genomics_cnn':
    return genomics_cnn.genomics_cnn(**kwargs)

  # EfficientNet models don't take in the batch size.
  if model_name == 'efficientnet':
    return efficientnet.efficientnet(**kwargs)
  if model_name == 'efficientnet_batch_ensemble':
    return efficientnet_batch_ensemble.efficientnet_batch_ensemble(**kwargs)
