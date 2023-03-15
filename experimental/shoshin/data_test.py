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

"""Tests for data sets."""

import os
import tempfile
from typing import List
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

import data  # local file import from experimental.shoshin


def _make_temp_dir() -> str:
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _make_serialized_image(size: int) -> bytes:
  image = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
  return tf.io.encode_png(image).numpy()


def _make_example(
    example_id: str,
    longitude: float,
    latitude: float,
    encoded_coordinates: str,
    label: float,
    string_label: float,
    patch_size: int,
    large_patch_size: int,
) -> tf.train.Example:
  example = tf.train.Example()
  example.features.feature['example_id'].bytes_list.value.append(
      example_id.encode()
  )
  example.features.feature['coordinates'].float_list.value.extend(
      (longitude, latitude)
  )
  example.features.feature['encoded_coordinates'].bytes_list.value.append(
      encoded_coordinates.encode()
  )
  example.features.feature['label'].float_list.value.append(label)
  example.features.feature['string_label'].bytes_list.value.append(
      string_label.encode()
  )
  example.features.feature['pre_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size)
  )
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size)
  )
  example.features.feature['pre_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size)
  )
  example.features.feature['post_image_png_large'].bytes_list.value.append(
      _make_serialized_image(large_patch_size)
  )
  return example


def _write_tfrecord(examples: List[tf.train.Example], path: str) -> None:
  with tf.io.TFRecordWriter(path) as file_writer:
    for example in examples:
      file_writer.write(example.SerializeToString())


def _create_test_data():
  examples_dir = _make_temp_dir()
  labeled_train_path = os.path.join(
      examples_dir, 'train_labeled_examples.tfrecord')
  labeled_test_path = os.path.join(
      examples_dir, 'test_labeled_examples.tfrecord')
  unlabeled_path = os.path.join(
      examples_dir, 'unlabeled_examples.tfrecord')

  _write_tfrecord([
      _make_example('1st', 0, 0, 'A0', 0, 'no_damage', 64, 256),
      _make_example('2nd', 0, 1, 'A1', 0, 'no_damage', 64, 256),
      _make_example('3rd', 0, 2, 'A2', 1, 'major_damage', 64, 256),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example('4th', 1, 0, 'B0', 0, 'no_damage', 64, 256),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example('5th', 2, 0, 'C0', -1, 'bad_example', 64, 256),
      _make_example('6th', 2, 1, 'C1', -1, 'bad_example', 64, 256),
      _make_example('7th', 2, 2, 'C2', -1, 'bad_example', 64, 256),
      _make_example('8th', 2, 3, 'C3', -1, 'bad_example', 64, 256),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path


class SkaiDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for Skai dataset."""

  DATASET_CLASS = data.SkaiDataset
  SPLITS = {
      'labeled_train': 3,
      'labeled_test': 1,
      'unlabeled': 4
  }
  EXAMPLE_DIR = _make_temp_dir()
  BUILDER_CONFIG_NAMES_TO_TEST = ['test_config']
  SKIP_TF1_GRAPH_MODE = True

  @classmethod
  def setUpClass(cls):
    super().setUpClass()

    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data()

    cls.DATASET_CLASS.BUILDER_CONFIGS = [
        data.SkaiDatasetConfig(
            name='test_config',
            labeled_train_pattern=labeled_train_path,
            labeled_test_pattern=labeled_test_path,
            unlabeled_pattern=unlabeled_path)
    ]


if __name__ == '__main__':
  tfds.testing.test_main()
