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

"""Tests for DrugCardiotoxicityDataset."""

from absl import flags
from absl.testing import parameterized
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub


class DrugCardiotoxicityDatasetTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Train', tfds.Split.TRAIN, True, 6523),
      ('Validation', tfds.Split.VALIDATION, False, 1631),
      ('Test', tfds.Split.TEST, False, 839),
      ('Test2', tfds.Split('test2'), False, 177))
  def testDatasetSize(self, split, is_training, expected_size):
    # The files expected are in TFRecord format and the new default format in
    # TFDS is set to ArrayRecord.
    # Disabling the default for cardiotoxicity builder tests for the time being.
    flags.FLAGS.array_record_default = False
    dataset_builder = ub.datasets.DrugCardiotoxicityDataset(
        split=split,
        is_training=is_training,
        shuffle_buffer_size=20)
    self.assertEqual(dataset_builder.num_examples, expected_size)

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN, True),
                                  ('Validation', tfds.Split.VALIDATION, False),
                                  ('Test', tfds.Split.TEST, False),
                                  ('Test2', tfds.Split('test2'), False))
  def testDatasetShape(self, split, is_training):
    flags.FLAGS.array_record_default = False
    batch_size = 128
    dataset_builder = ub.datasets.DrugCardiotoxicityDataset(
        split=split,
        is_training=is_training,
        shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))
    atoms = element['features']['atoms']
    pairs = element['features']['pairs']
    atom_mask = element['features']['atom_mask']
    pair_mask = element['features']['pair_mask']
    dist2topk_nbs = element['features']['dist2topk_nbs']
    molecule_id = element['features']['molecule_id']
    labels = element['labels']

    self.assertEqual(atoms.shape, (batch_size, 60, 27))
    self.assertEqual(pairs.shape, (batch_size, 60, 60, 12))
    self.assertEqual(atom_mask.shape, (batch_size, 60))
    self.assertEqual(pair_mask.shape, (batch_size, 60, 60))
    self.assertEqual(dist2topk_nbs.shape, (batch_size, 1))
    self.assertEqual(molecule_id.shape, (batch_size,))
    self.assertEqual(labels.shape, (batch_size, 2))


if __name__ == '__main__':
  tf.test.main()
