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

"""Uncertainty baseline training."""

import importlib
from absl import logging


_IMPORTS = [
    'datasets',
    'models',
    'optimizers',
    'strategy_utils',
    'utils',
]


def _lazy_import(name):
  module = importlib.import_module(__name__)
  imported = importlib.import_module('.' + name, 'uncertainty_baselines')
  setattr(module, name, imported)
  return imported


for module_name in _IMPORTS:
  try:
    _lazy_import(module_name)
  except ModuleNotFoundError as e:
    logging.warning(e)

