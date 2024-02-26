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

"""Tests for CIFAR."""

from unittest import mock

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
        ub.datasets.Cifar10CorruptedDataset, (32, 32, 3),
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


  @parameterized.parameters(
      (3, 'clean_labels', 0., (None,), (None, 3, 1), (None, 3, 1),
       (None, 3, 1)),
      (1, 'aggre_labels', 0.5, (3, 10), (None, 1, 1), (None, 1, 1),
       (None, 1, 10)),
  )
  @mock.patch.object(ub.datasets.Cifar10NDataset, '_setup_annotator_tables',
                     lambda _: None)
  @mock.patch.object(
      ub.datasets.Cifar10NDataset,
      'average_annotator_load',
      new_callable=mock.PropertyMock,
      return_value=1)
  @mock.patch.object(
      ub.datasets.Cifar10NDataset,
      'num_effective_annotators',
      new_callable=mock.PropertyMock,
      return_value=1)
  def test_expected_cifar10n_features(self, num_annotators_per_example,
                                      supervised_label, mixup_alpha,
                                      expected_label_shape,
                                      expected_times_shape, expected_ids_shape,
                                      expected_annotator_labels_shape,
                                      *unused_mocks):
    builder = ub.datasets.Cifar10NDataset(
        split='train',
        num_annotators_per_example=num_annotators_per_example,
        aug_params={
            'mixup_alpha': mixup_alpha,
            'same_mix_weight_per_batch': True
        },
        supervised_label=supervised_label)
    dataset = builder.load(batch_size=3)
    if mixup_alpha > 0:
      self.assertEqual(
          list(dataset.element_spec.keys()), [
              'labels', 'features', 'aggre_labels', 'worse_labels',
              '_enumerate_added_per_step_id', 'clean_labels', 'pi_features',
              'mixup_weights', 'mixup_index'
          ])
    else:
      self.assertEqual(
          list(dataset.element_spec.keys()), [
              'labels', 'features', 'aggre_labels', 'worse_labels',
              '_enumerate_added_per_step_id', 'clean_labels', 'pi_features'
          ])
    self.assertEqual(
        list(dataset.element_spec['pi_features'].keys()),
        ['annotator_labels', 'annotator_ids', 'annotator_times'])
    self.assertEqual(
        dataset.element_spec['labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))
    self.assertEqual(dataset.element_spec['pi_features']['annotator_ids'],
                     tf.TensorSpec(shape=expected_ids_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_times'],
        tf.TensorSpec(shape=expected_times_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_labels'],
        tf.TensorSpec(shape=expected_annotator_labels_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['clean_labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))

  @parameterized.parameters(
      (0., (None,), (None, 1, 1), (None, 1, 1), (None, 1, 1)),
      (0.5, (None, 100), (None, 1, 1), (None, 1, 1), (None, 1, 100)),
  )
  @mock.patch.object(ub.datasets.Cifar100NDataset, '_setup_annotator_tables',
                     lambda _: None)
  @mock.patch.object(
      ub.datasets.Cifar100NDataset,
      'average_annotator_load',
      new_callable=mock.PropertyMock,
      return_value=1)
  @mock.patch.object(
      ub.datasets.Cifar100NDataset,
      'num_effective_annotators',
      new_callable=mock.PropertyMock,
      return_value=1)
  def test_expected_cifar100n_features(self, label_smoothing,
                                       expected_label_shape,
                                       expected_times_shape, expected_ids_shape,
                                       expected_annotator_labels_shape,
                                       *unused_mocks):
    builder = ub.datasets.Cifar100NDataset(
        split='train', aug_params={'label_smoothing': label_smoothing})
    dataset = builder.load(batch_size=3)
    self.assertEqual(
        list(dataset.element_spec.keys()), [
            'labels', 'features', '_enumerate_added_per_step_id',
            'clean_labels', 'pi_features'
        ])
    self.assertEqual(
        list(dataset.element_spec['pi_features'].keys()),
        ['annotator_labels', 'annotator_ids', 'annotator_times'])
    self.assertEqual(
        dataset.element_spec['labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))
    self.assertEqual(dataset.element_spec['pi_features']['annotator_ids'],
                     tf.TensorSpec(shape=expected_ids_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_times'],
        tf.TensorSpec(shape=expected_times_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_labels'],
        tf.TensorSpec(shape=expected_annotator_labels_shape, dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['clean_labels'],
        tf.TensorSpec(shape=expected_label_shape, dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
