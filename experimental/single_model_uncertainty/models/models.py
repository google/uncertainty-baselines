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

"""Model getter utility."""

import json
import logging
from typing import List

import tensorflow as tf
import genomics_cnn  # local file import
import wide_resnet  # local file import
from uncertainty_baselines.models import bert
from uncertainty_baselines.models import criteo_mlp
from uncertainty_baselines.models import resnet20
from uncertainty_baselines.models import resnet50_deterministic
from uncertainty_baselines.models import textcnn


def get_model_names() -> List[str]:
  return [
      'genomics_cnn',
      'bert',
      'criteo_mlp',
      'genomics_cnn',
      'resnet20',
      'resnet50',
      'textcnn',
      'wide_resnet',
  ]


# TODO(znado): check into using Keras' deserialization functionality, similar to
# edward2/tensorflow/initializers.py line 772.
def get(
    model_name: str,
    **hyperparameters) -> tf.keras.Model:
  """Gets a model builder by name.

  Args:
    model_name: Name of the model builder class.
    **hyperparameters: dict of possible kwargs to be passed to the model
      constructor.

  Returns:
    A model builder class with a method .build(split) which can be called to
    get the tf.data.Dataset, which has elements that are a dict with keys
    'features' and 'labels'.

  Raises:
    ValueError: If model_name is unrecognized.
  """
  logging.info(
      'Building model %s with additional kwargs:\n%s',
      model_name,
      json.dumps(hyperparameters, indent=2, sort_keys=True))
  if model_name not in get_model_names():
    raise ValueError('Unrecognized model type: {!r}'.format(model_name))

  # load from single_model_uncertainty directory
  if model_name == 'genomics_cnn':
    return genomics_cnn.genomics_cnn(**hyperparameters)  # pytype: disable=bad-return-type  # typed-keras
  if model_name == 'wide_resnet':
    return wide_resnet.wide_resnet(**hyperparameters)

  # load from uncertainty_baselines directory
  if model_name == 'criteo_mlp':
    return criteo_mlp.criteo_mlp(**hyperparameters)  # pytype: disable=bad-return-type  # typed-keras
  if model_name == 'resnet20':
    return resnet20.resnet20(**hyperparameters)  # pytype: disable=bad-return-type  # typed-keras
  if model_name == 'resnet50':
    return resnet50_deterministic.resnet50_deterministic(**hyperparameters)
  if model_name == 'textcnn':
    return textcnn.textcnn(**hyperparameters)  # pytype: disable=bad-return-type  # typed-keras
  if model_name == 'bert':
    return bert.bert_model(**hyperparameters)  # pytype: disable=bad-return-type  # typed-keras
