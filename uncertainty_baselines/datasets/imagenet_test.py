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

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub


# TODO(dusenberrymw): Use TFDS mocking.
class ImageNetDatasetTest(ub.datasets.DatasetTest, parameterized.TestCase):

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

  @parameterized.parameters(
      (False, False, ['features', 'labels'], (None, 224, 224, 3), (None,)),
      (True, False, ['mask', 'features', 'labels'], (3, 224, 224, 3), (3,)),
      (False, True, ['features', 'labels', 'file_name'], (None, 224, 224, 3),
       (None,)),
  )
  def test_expected_features(self, mask_and_pad, include_file_name,
                             expected_features, expected_feature_shape,
                             expected_label_shape):
    builder = ub.datasets.ImageNetDataset(
        'train', mask_and_pad=mask_and_pad, include_file_name=include_file_name)
    dataset = builder.load(batch_size=3)
    self.assertEqual(list(dataset.element_spec.keys()), expected_features)
    # NOTE: The batch size is not statically known when drop_remainder=False
    # (default) and mask_and_pad=False (default), but is statically known if
    # mask_and_pad=True.
    self.assertEqual(
        dataset.element_spec['features'],
        tf.TensorSpec(shape=expected_feature_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))

  @parameterized.parameters(
      (False, False, ['features', 'labels'], (None, 224, 224, 3), (None,)),
      (True, False, ['mask', 'features', 'labels'], (3, 224, 224, 3), (3,)),
      (False, True, ['features', 'labels', 'file_name'], (None, 224, 224, 3),
       (None,)),
  )
  def test_corrupted_expected_features(self, mask_and_pad, include_file_name,
                                       expected_features,
                                       expected_feature_shape,
                                       expected_label_shape):
    builder = ub.datasets.ImageNetCorruptedDataset(
        corruption_type='gaussian_blur',
        severity=3,
        mask_and_pad=mask_and_pad,
        include_file_name=include_file_name)
    dataset = builder.load(batch_size=3)
    self.assertEqual(list(dataset.element_spec.keys()), expected_features)
    # NOTE: The batch size is not statically known when drop_remainder=False
    # (default) and mask_and_pad=False (default), but is statically known if
    # mask_and_pad=True.
    self.assertEqual(
        dataset.element_spec['features'],
        tf.TensorSpec(shape=expected_feature_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
