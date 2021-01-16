# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Tests for uncertainty_baselines.datasets.mnli."""

from absl.testing import parameterized

import tensorflow as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import mnli


class MnliTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('ind,train', 'matched', tfds.Split.TRAIN),
      ('ind,valid', 'matched', tfds.Split.VALIDATION),
      ('ind,test', 'matched', tfds.Split.TEST),
      ('ood,valid', 'mismatched', tfds.Split.VALIDATION),
      ('ood,test', 'mismatched', tfds.Split.TEST))
  def testDatasetSize(self, mode, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    dataset_builder = mnli.MnliDataset(
        split=split,
        mode=mode,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))
    self.assertEqual(element['text_a'].shape[0], batch_size)
    self.assertEqual(element['text_b'].shape[0], batch_size)
    self.assertEqual(element['labels'].shape[0], batch_size)

  def testTrainSplitError(self):
    """Tests for ValueError when calling train split for mismatched data."""
    with self.assertRaises(Exception):
      mnli.MnliDataset(
          mode='mismatched',
          split=tfds.Split.TRAIN,
          shuffle_buffer_size=20)


if __name__ == '__main__':
  tf.test.main()
