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

"""Tests for CIFAR."""

from absl.testing import parameterized
import tensorflow as tf
import uncertainty_baselines as ub


class CifarDatasetTest(ub.datasets.DatasetTest, parameterized.TestCase):

  def testCifar10DatasetShape(self):
    super()._testDatasetSize(
        ub.datasets.Cifar10Dataset, (32, 32, 3), validation_percent=0.1)

  def testCifar100DatasetShape(self):
    super()._testDatasetSize(
        ub.datasets.Cifar100Dataset, (32, 32, 3), validation_percent=0.1)

  def testCifar10CorruptedDatasetShape(self):
    super()._testDatasetSize(
        ub.datasets.Cifar10CorruptedDataset,
        (32, 32, 3),
        splits=['test'],
        corruption_type='brightness',
        severity=1)

  @parameterized.named_parameters(('Train', 'train', 45000),
                                  ('Validation', 'validation', 5000),
                                  ('Test', 'test', 10000))
  def testDatasetSize(self, split, expected_size):
    builder = ub.datasets.Cifar10Dataset(
        split=split, shuffle_buffer_size=20, validation_percent=0.1)
    self.assertEqual(builder.num_examples, expected_size)

  @parameterized.parameters(
      (False, ['features', '_enumerate_added_per_step_id', 'labels'],
       (None, 32, 32, 3), (None,)),
      (True, ['mask', 'features', '_enumerate_added_per_step_id', 'labels'],
       (3, 32, 32, 3), (3,)),
  )
  def test_expected_features(self, mask_and_pad, expected_features,
                             expected_feature_shape, expected_label_shape):
    builder = ub.datasets.Cifar10Dataset(
        split='train', mask_and_pad=mask_and_pad)
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
