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
"""Utilities for Kaggle's Diabetic Retinopathy Detection datasets."""

from typing import Optional

import tensorflow as tf
import uncertainty_baselines as ub


def load_diabetic_retinopathy_detection(
    split: ub.datasets.base.Split,
    batch_size: int,
    eval_batch_size: int,
    strategy: Optional[tf.distribute.Strategy] = None,
    data_dir: Optional[str] = None,
) -> tf.data.Dataset:
  """Loads Kaggle's Diabetic Retinopathy Detection dataset."""
  builder = ub.datasets.get(
      "diabetic_retinopathy_detection",
      batch_size=batch_size,
      eval_batch_size=eval_batch_size,
      data_dir=data_dir)
  dataset = ub.utils.build_dataset(
      builder,
      strategy=strategy or tf.distribute.MirroredStrategy(),
      split=split)

  return dataset
