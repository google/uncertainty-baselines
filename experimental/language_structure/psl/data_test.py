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

"""Tests for data."""

import tensorflow as tf
import tensorflow_datasets as tfds
import data  # local file import from experimental.language_structure.psl


class DataTest(tfds.testing.TestCase):

  @tfds.testing.run_in_graph_and_eager_modes
  def test_create_utterance_mask_feature(self):
    dialogs = tf.constant([[[[1, 2], [0, 0]], [[4, 5], [6, 0]], [[7, 8], [9,
                                                                          0]]],
                           [[[10, 11], [12, 13]], [[14, 0], [15, 0]],
                            [[0, 0], [0, 0]]],
                           [[[16, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0],
                                                                  [0, 0]]]])
    feature = data.create_utterance_mask_feature(
        dialogs,
        pad_utterance_mask_value=1,
        utterance_mask_value=2,
        last_utterance_mask_value=3)

    self.assertAllEqual(feature, tf.constant([[2, 2, 3], [2, 3, 1], [3, 1, 1]]))

  @tfds.testing.run_in_graph_and_eager_modes
  def test_create_keyword_feature(self):
    dialogs = tf.constant([[[[1, 2], [0, 0]], [[4, 5], [6, 0]], [[7, 8], [9,
                                                                          0]]],
                           [[[10, 11], [12, 13]], [[14, 0], [15, 0]],
                            [[0, 0], [0, 0]]],
                           [[[0, 0], [0, 0]], [[0, 0], [0, 0]], [[0, 0], [0,
                                                                          0]]]])
    feature = data.create_keyword_feature(
        dialogs,
        keyword_ids=[1, 7, 14, 15],
        include_keyword_value=1,
        exclude_keyword_value=2)

    self.assertAllEqual(feature, tf.constant([[1, 2, 1], [2, 1, 2], [2, 2, 2]]))

  def test_create_feature_shape(self):
    dialog_length = 3
    utterance_per_turn = 2
    sequence_length = 4
    dialogs = tf.keras.Input(
        shape=(
            dialog_length,
            utterance_per_turn,
            sequence_length,
        ),
        dtype=tf.int32)
    keyword_ids_per_class = [[1, 2], [3]]
    feature = data.create_features(
        dialogs,
        keyword_ids_per_class,
        pad_utterance_mask_value=1,
        utterance_mask_value=2,
        last_utterance_mask_value=3,
        include_keyword_value=4,
        exclude_keyword_value=5)

    self.assertEqual(feature.shape.as_list(),
                     [None, dialog_length, 1 + len(keyword_ids_per_class)])


if __name__ == '__main__':
  tfds.testing.test_main()
