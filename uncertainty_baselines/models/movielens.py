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

"""TF Keras model for an MLP for Criteo, from arxiv.org/abs/1906.02530."""

from typing import Any, Dict, List
import tensorflow as tf


# Model hyperparameters.
_CATEGORICAL_BUCKET_DICT = {
    'user_id': 6041,
    'movie_id': 3953,
}

_CATEGORICAL_EMBED_DIM = {
    'user_id': 8,
    'movie_id': 8,
}

_LAYER_SIZES = [50, 20, 10]


def _make_input_layers(batch_size: int) -> Dict[str, tf.keras.layers.Input]:  # pytype: disable=invalid-annotation  # typed-keras
  """Defines an input layer for tf.keras model with int32 and string dtypes."""

  # TODO(chenzhe): Add more user and movie related features as inputs. Currently
  # only used movie_id and user_id as the input features.
  out = {
      'movie_id': tf.keras.layers.Input(
          [], batch_size=batch_size, dtype=tf.string, name='movie_id'),
      'user_id': tf.keras.layers.Input(
          [], batch_size=batch_size, dtype=tf.string, name='user_id')
  }
  return out


# In order to get better typing we would need to be able to reference the
# FeatureColumn class defined here, but it is not exported:
# https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/feature_column/feature_column_v2.py#L2121.
def _make_feature_columns() -> List[Any]:
  """Build feature_columns for converting features to a dense vector."""
  categorical_feature_columns = []

  # TODO(chenzhe): Add more user and movie related features as inputs. Currently
  # only used movie_id and user_id as the input features.
  categorical_feature_arr = ['user_id', 'movie_id']
  for name in categorical_feature_arr:
    feature_col = tf.feature_column.categorical_column_with_hash_bucket(
        name, _CATEGORICAL_BUCKET_DICT[name])
    categorical_feature_columns.append(
        tf.feature_column.embedding_column(
            feature_col, _CATEGORICAL_EMBED_DIM[name]))
  return categorical_feature_columns


def movielens(
    batch_size: int,
    **unused_kwargs: Dict[str, Any]) -> tf.keras.models.Model:
  """Creates a tf.keras.Model fully connected model for MovieLens."""
  categorical_feature_columns = _make_feature_columns()
  input_layer = _make_input_layers(batch_size)
  categorical_features = tf.keras.layers.DenseFeatures(
      categorical_feature_columns)(input_layer)
  x = tf.keras.layers.BatchNormalization()(categorical_features)
  for size in _LAYER_SIZES:
    x = tf.keras.layers.Dense(size, activation='relu')(x)
  logits = tf.keras.layers.Dense(1)(x)

  return tf.keras.models.Model(
      inputs=input_layer, outputs=logits, name='movielens_mlp')
