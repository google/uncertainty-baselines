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
from uncertainty_baselines.models import textcnn
from uncertainty_baselines.models import unet
from uncertainty_baselines.models import wide_resnet

try:
  from uncertainty_baselines.models.bert import bert_model  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning(
      'Skipped bert model import due to ImportError.', exc_info=True)
except tf.errors.NotFoundError:
  logging.warning(
      'Skipped bert model import due to tf.errors.NotFoundError.',
      exc_info=True)
except AttributeError:
  # TODO(dusenberrymw): Check if this is fixed upstream.
  logging.warning(
      'Skipped bert model import due to AttributeError.', exc_info=True)


# TODO(dusenberrymw): Update to the full list of models, or remove this module.
def get_model_names() -> List[str]:
  """Returns a list of valid model names."""
  model_names = [
      'bert',
      'criteo_mlp',
      'genomics_cnn',
      'movielens',
      'resnet20',
      'textcnn',
      'unet',
      'wide_resnet',
  ]
  return model_names


# TODO(znado): check into using Keras' deserialization functionality, similar to
# edward2/tensorflow/initializers.py line 772.
# TODO(dusenberrymw): Update to the full list of models, or remove this module.
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

  # pytype: disable=bad-return-type  # typed-keras
  # pytype: disable=not-callable  # PyType's type inference is incorrect here.
  if model_name == 'criteo_mlp':
    return criteo_mlp(**kwargs)
  if model_name == 'movielens':
    return movielens(**kwargs)
  if model_name == 'resnet20':
    return resnet20(**kwargs)
  if model_name == 'textcnn':
    return textcnn(**kwargs)
  if model_name == 'unet':
    return unet(**kwargs)
  if model_name == 'bert':
    return bert_model(**kwargs)
  if model_name == 'wide_resnet':
    return wide_resnet(**kwargs)
  if model_name == 'genomics_cnn':
    return genomics_cnn(**kwargs)

  # EfficientNet models don't take in the batch size.
  if model_name == 'efficientnet':
    return efficientnet(**kwargs)
  if model_name == 'efficientnet_batch_ensemble':
    return efficientnet_batch_ensemble(**kwargs)
  # pytype: enable=not-callable
  # pytype: enable=bad-return-type  # typed-keras
