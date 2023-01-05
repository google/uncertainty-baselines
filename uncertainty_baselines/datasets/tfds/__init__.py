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

"""Uncertainty baselines libraries for tensorflow_datasets."""

from absl import logging

try:
  from uncertainty_baselines.datasets.tfds.tfds_builder_from_sql_client_data import TFDSBuilderFromSQLClientData  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning(
      (
          'Skipped importing the TFDSBuilderFromSQLClientData used when loading'
          ' the CIFAR subpopulation dataset due to ImportError. Try installing'
          ' uncertainty baselines with the `datasets` extras.'
      ),
      exc_info=True,
  )
