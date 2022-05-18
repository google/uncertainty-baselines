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

"""Tests for ImageNet."""

import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub


# TODO(dusenberrymw): Use TFDS mocking.
class ImageNetDatasetTest(ub.datasets.DatasetTest):

  def test_imagenet_dataset_size(self):
    super()._testDatasetSize(
        ub.datasets.ImageNetDataset, (224, 224, 3), validation_percent=0.1)

  def test_imagenet_corrupted_dataset_size(self):
    super()._testDatasetSize(
        ub.datasets.ImageNetCorruptedDataset,
        (224, 224, 3),
        splits=[tfds.Split.VALIDATION],
        corruption_type='gaussian_blur',
        severity=3,
    )

  def test_imagenet_expected_features(self):
    builder = ub.datasets.ImageNetDataset('train')
    dataset = builder.load(batch_size=3)
    self.assertEqual(list(dataset.element_spec.keys()), ['features', 'labels'])
    self.assertEqual(dataset.element_spec['features'],
                     tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))
    self.assertEqual(dataset.element_spec['labels'],
                     tf.TensorSpec(shape=(None,), dtype=tf.float32))

  def test_imagenet_expected_features_with_filename(self):
    builder = ub.datasets.ImageNetDataset('train', include_file_name=True)
    dataset = builder.load(batch_size=3)
    self.assertEqual(
        list(dataset.element_spec.keys()), ['features', 'labels', 'file_name'])
    self.assertEqual(dataset.element_spec['features'],
                     tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))
    self.assertEqual(dataset.element_spec['labels'],
                     tf.TensorSpec(shape=(None,), dtype=tf.float32))

  def test_imagenet_corrupted_expected_features(self):
    builder = ub.datasets.ImageNetCorruptedDataset(
        corruption_type='gaussian_blur', severity=3)
    batch_size = 3
    dataset = builder.load(batch_size=batch_size)
    self.assertEqual(list(dataset.element_spec.keys()), ['features', 'labels'])
    self.assertEqual(dataset.element_spec['features'],
                     tf.TensorSpec(shape=(None, 224, 224, 3), dtype=tf.float32))
    self.assertEqual(dataset.element_spec['labels'],
                     tf.TensorSpec(shape=(None,), dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
