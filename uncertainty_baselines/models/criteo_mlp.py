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

# Lint as: python3
"""TF Keras model for an MLP for Criteo, from arxiv.org/abs/1906.02530."""

from typing import Any, Dict, List, Tuple
import tensorflow as tf

from uncertainty_baselines import datasets


# Model hyperparameters.
_NUM_EMBED_DIMS = [
    3, 9, 29, 11, 17, None, 14, 4, None, 12, 19, 24, 29, None, 13, 25, None,
    8, 29, None, 22, None, None, 31, None, 29,
]
_NUM_HAS_BUCKETS = [
    1373, 2148, 4847, 9781, 396, 28, 3591, 2798, 14, 7403, 2511, 5598, 9501,
    46, 4753, 4056, 23, 3828, 5856, 12, 4226, 23, 61, 3098, 494, 5087,
]
_LAYER_SIZES = [2572, 1454, 1596]


def _make_input_layers(batch_size: int) -> Dict[str, tf.keras.layers.Input]:
  """Defines an input layer for tf.keras model with int32 and string dtypes."""
  out = {}
  for idx in range(1, datasets.criteo.NUM_TOTAL_FEATURES + 1):
    if idx <= datasets.criteo.NUM_INT_FEATURES:
      dtype = tf.int32
    else:
      dtype = tf.string
    name = datasets.criteo.feature_name(idx)
    out[name] = tf.keras.layers.Input(
        [], batch_size=batch_size, dtype=dtype, name=name)
  return out


# In order to get better typing we would need to be able to reference the
# FeatureColumn class defined here, but it is not exported:
# https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/feature_column/feature_column_v2.py#L2121.
def _make_feature_columns() -> Tuple[List[Any], List[Any]]:
  """Build feature_columns for converting features to a dense vector."""
  categorical_feature_columns = []
  categorical_indices = range(
      datasets.criteo.NUM_INT_FEATURES + 1,
      datasets.criteo.NUM_TOTAL_FEATURES + 1)
  for idx in categorical_indices:
    name = datasets.criteo.feature_name(idx)
    cat_idx = idx - datasets.criteo.NUM_INT_FEATURES - 1
    num_buckets = _NUM_HAS_BUCKETS[cat_idx]
    num_embed_dims = _NUM_EMBED_DIMS[cat_idx]

    hash_col = tf.feature_column.categorical_column_with_hash_bucket(
        name, num_buckets)
    if num_embed_dims:
      cat_col = tf.feature_column.embedding_column(hash_col, num_embed_dims)
    else:
      cat_col = tf.feature_column.indicator_column(hash_col)
    categorical_feature_columns.append(cat_col)

  integer_feature_columns = []
  for idx in range(1, datasets.criteo.NUM_INT_FEATURES + 1):
    name = datasets.criteo.feature_name(idx)
    integer_feature_columns.append(tf.feature_column.numeric_column(name))
  return integer_feature_columns, categorical_feature_columns


def create_model(
    batch_size: int,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Creates a tf.keras.Model fully connected model for Criteo."""
  integer_feature_columns, categorical_feature_columns = _make_feature_columns()
  input_layer = _make_input_layers(batch_size)
  integer_features = tf.keras.layers.DenseFeatures(
      integer_feature_columns)(input_layer)
  categorical_features = tf.keras.layers.DenseFeatures(
      categorical_feature_columns)(input_layer)
  x = tf.concat([integer_features, categorical_features], axis=-1)
  x = tf.keras.layers.BatchNormalization()(x)
  for size in _LAYER_SIZES:
    x = tf.keras.layers.Dense(size, activation='relu')(x)
  logits = tf.keras.layers.Dense(1)(x)

  return tf.keras.models.Model(
      inputs=input_layer, outputs=logits, name='criteo_mlp')
