# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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
import sys
from absl import logging


_IMPORTS = [
    'datasets',
    'halton',
    'models',
    'optimizers',
    'plotting',
    'schedules',
    'strategy_utils',
    'utils',
    'test_utils',
]


def _lazy_import(name):
  """Load a submodule named `name`."""
  if name not in _IMPORTS:
    raise AttributeError(
        'module uncertainty_baselines has no attribute {}'.format(name))
  module = importlib.import_module(__name__)
  try:
    imported = importlib.import_module('.' + name, 'uncertainty_baselines')
  except AttributeError:
    logging.warning(
        'Submodule %s was found, but will not be loaded due to AttributeError '
        'within it while importing.', name)
    return
  setattr(module, name, imported)
  return imported


# Lazily load any top level modules when accessed. Requires Python 3.7.
if sys.version_info >= (3, 7):
  __getattr__ = _lazy_import
else:
  for module_name in _IMPORTS:
    try:
      _lazy_import(module_name)
    except ModuleNotFoundError:
      logging.error(
          'Skipped importing top level uncertainty_baselines module %s due to '
          'ModuleNotFoundError:', module_name, exc_info=True)
