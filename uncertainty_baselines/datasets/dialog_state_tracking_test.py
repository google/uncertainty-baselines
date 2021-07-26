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

"""Tests for dialog_state_tracking."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from uncertainty_baselines.datasets import dialog_state_tracking

max_dial_len = dialog_state_tracking.MAX_DIALOG_LEN
max_utt_len = dialog_state_tracking.MAX_UTT_LEN


class DialogStateTrackingTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(
      ('Train', tfds.Split.TRAIN, dialog_state_tracking.NUM_TRAIN),
      ('Test', tfds.Split.TEST, dialog_state_tracking.NUM_TEST))
  def testDatasetSize(self, split, expected_size):
    dataset_builder = ub.datasets.SimDialDataset(
        split=split, shuffle_buffer_size=20)
    self.assertEqual(dataset_builder.num_examples, expected_size)

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Test', tfds.Split.TEST))
  def testDatasetShape(self, split):
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    dataset_builder = ub.datasets.SimDialDataset(
        split=split, shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    features_usr = element['usr_utt']
    features_sys = element['sys_utt']
    labels = element['label']
    dialog_len = element['dialog_len']

    features_usr_shape = features_usr.shape
    features_sys_shape = features_sys.shape
    labels_shape = labels.shape
    dialog_len_shape = dialog_len.shape

    self.assertEqual(features_usr_shape,
                     (batch_size, max_dial_len, max_utt_len))
    self.assertEqual(features_sys_shape,
                     (batch_size, max_dial_len, max_utt_len))
    self.assertEqual(labels_shape, (batch_size, max_dial_len))
    self.assertEqual(dialog_len_shape, (batch_size,))

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Test', tfds.Split.TEST))
  def testDialogLength(self, split):
    """Checks dialog length matches with that in dialog_len."""
    batch_size = 9 if split == tfds.Split.TRAIN else 5
    dataset_builder = ub.datasets.SimDialDataset(
        split=split, shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size).take(1)
    element = next(iter(dataset))

    features_usr = element['usr_utt'].numpy()
    features_sys = element['sys_utt'].numpy()
    labels = element['label'].numpy()
    dialog_len = element['dialog_len'].numpy()

    # Compute dialog length based on user utterances.
    utter_len_usr = np.sum(features_usr > 0, axis=-1)
    dialog_len_usr = np.sum(utter_len_usr > 0, axis=-1)

    # Compute dialog length based on system utterances.
    utter_len_sys = np.sum(features_sys > 0, axis=-1)
    dialog_len_sys = np.sum(utter_len_sys > 0, axis=-1)

    # Compute dialog length based on state labels.
    dialog_len_label = np.sum(labels > 0, axis=-1)

    np.testing.assert_array_equal(dialog_len_usr, dialog_len)
    np.testing.assert_array_equal(dialog_len_sys, dialog_len)
    np.testing.assert_array_equal(dialog_len_label, dialog_len)

  def testVocab(self):
    """Tests if vocab is loaded correctly."""
    dataset_builder = ub.datasets.SimDialDataset(
        split=tfds.Split.TRAIN, shuffle_buffer_size=20)

    vocab_dict_utter = dataset_builder.vocab_utter
    vocab_dict_label = dataset_builder.vocab_label

    self.assertLen(vocab_dict_utter, dialog_state_tracking.VOCAB_SIZE_UTT)
    self.assertLen(vocab_dict_label, dialog_state_tracking.VOCAB_SIZE_LABEL)

  @parameterized.named_parameters(('Train', tfds.Split.TRAIN),
                                  ('Test', tfds.Split.TEST))
  def testDatasetSpec(self, split):
    """Tests if dataset specification returns valid tensor shapes."""
    batch_size = 9
    dataset_builder = ub.datasets.SimDialDataset(
        split=split, shuffle_buffer_size=20)
    dataset = dataset_builder.load(batch_size=batch_size)
    dataset_spec = tf.data.DatasetSpec.from_value(dataset).element_spec

    # Specify expected shape.
    utt_spec = tf.TensorSpec((batch_size, max_dial_len, max_utt_len),
                             dtype=tf.int32)
    label_spec = tf.TensorSpec((batch_size, max_dial_len), dtype=tf.int32)

    self.assertEqual(dataset_spec['sys_utt'], utt_spec)
    self.assertEqual(dataset_spec['usr_utt'], utt_spec)
    self.assertEqual(dataset_spec['label'], label_spec)


if __name__ == '__main__':
  tf.test.main()
