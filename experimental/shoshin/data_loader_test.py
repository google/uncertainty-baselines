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

"""Tests for data loaders."""

import os
import tempfile
from typing import List
import numpy as np
import tensorflow as tf

import data  # local file import from experimental.shoshin
from google3.testing.pybase import googletest


def _make_temp_dir() -> str:
  return tempfile.mkdtemp(dir=os.environ.get('TEST_TMPDIR'))


def _make_serialized_image(size: int, pixel_value: int) -> bytes:
  image = np.ones((size, size, 3), dtype=np.uint8) * pixel_value
  return tf.io.encode_png(image).numpy()


def _make_example(
    longitude: float,
    latitude: float,
    encoded_coordinates: str,
    label: float,
    patch_size: int,
    before_pixel_value: int,
    after_pixel_value: int) -> tf.train.Example:
  example = tf.train.Example()
  example.features.feature['coordinates'].float_list.value.extend(
      (longitude, latitude))
  example.features.feature['encoded_coordinates'].bytes_list.value.append(
      encoded_coordinates.encode())
  example.features.feature['label'].float_list.value.append(label)
  example.features.feature['pre_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, before_pixel_value))
  example.features.feature['post_image_png'].bytes_list.value.append(
      _make_serialized_image(patch_size, after_pixel_value))
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
      _make_example(0, 0, 'A0', 0, 64, 0, 255),
      _make_example(0, 1, 'A1', 0, 64, 0, 255),
      _make_example(0, 2, 'A2', 1, 64, 0, 255),
  ], labeled_train_path)

  _write_tfrecord([
      _make_example(1, 0, 'B0', 0, 64, 0, 255),
  ], labeled_test_path)

  _write_tfrecord([
      _make_example(2, 0, 'C0', -1, 64, 0, 255),
      _make_example(2, 1, 'C1', -1, 64, 0, 255),
      _make_example(2, 2, 'C2', -1, 64, 0, 255),
      _make_example(2, 3, 'C3', -1, 64, 0, 255),
  ], unlabeled_path)

  return labeled_train_path, labeled_test_path, unlabeled_path


class DataLoaderTest(googletest.TestCase):
  def setUp(self):
    super().setUp()

    labeled_train_path, labeled_test_path, unlabeled_path = _create_test_data()
    self.labeled_train_path = labeled_train_path
    self.labeled_test_path = labeled_test_path
    self.unlabeled_path = unlabeled_path

  def test_get_skai_dataset_post_only(self):
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': True,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs)
    ds = dataloader.train_ds
    features = next(ds.as_numpy_iterator())
    self.assertIn('input_feature', features)
    input_feature = features['input_feature']
    self.assertEqual(input_feature.shape, (224, 224, 3))
    self.assertEqual(input_feature.dtype, np.float32)
    np.testing.assert_equal(input_feature, 1.0)

  def test_get_skai_dataset_pre_post(self):
    dataset_builder = data.get_dataset('skai')

    kwargs = {
        'labeled_train_pattern': self.labeled_train_path,
        'unlabeled_train_pattern': self.unlabeled_path,
        'validation_pattern': self.labeled_test_path,
        'use_post_disaster_only': False,
        'data_dir': _make_temp_dir(),
    }

    dataloader = dataset_builder(
        1,
        initial_sample_proportion=1,
        subgroup_ids=(),
        subgroup_proportions=(),
        **kwargs)
    ds = dataloader.train_ds
    features = next(ds.as_numpy_iterator())
    self.assertIn('input_feature', features)
    input_feature = features['input_feature']
    self.assertEqual(input_feature.shape, (224, 224, 6))
    self.assertEqual(input_feature.dtype, np.float32)
    np.testing.assert_equal(input_feature[:, :, :3], 0.0)
    np.testing.assert_equal(input_feature[:, :, 3:], 1.0)


if __name__ == '__main__':
  googletest.main()
