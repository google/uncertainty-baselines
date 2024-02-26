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

"""Tests for pi_utils."""

from absl.testing import parameterized
import tensorflow as tf
import pi_utils  # local file import from baselines.privileged_information


class PiUtilsTest(parameterized.TestCase, tf.test.TestCase):
  num_dataset_annotators = 5
  num_annotators_per_example = 3
  annotator_feature_length = 2
  num_classes = 10
  batch_size = 4

  @property
  def big_example_with_all_fields(self):
    return {
        'pi_features': {
            'annotator_ids':
                tf.random.uniform(
                    [self.batch_size, self.num_annotators_per_example],
                    minval=0,
                    maxval=self.num_dataset_annotators,
                    dtype=tf.int32),
            'annotator_features':
                tf.random.normal([
                    self.batch_size, self.num_annotators_per_example,
                    self.annotator_feature_length
                ]),
            'annotator_confidences':
                tf.random.normal(
                    [self.batch_size, self.num_annotators_per_example]),
            'annotator_labels':
                tf.random.normal([
                    self.batch_size, self.num_annotators_per_example,
                    self.num_classes
                ]),
        },
        'clean_labels': tf.range(self.batch_size)
    }

  @property
  def example_with_incorrect_labels(self):
    return {
        'pi_features': {
            'annotator_ids': tf.reshape(tf.range(4), [2, 2]),
            'annotator_labels': tf.constant([[[0], [1]], [[0], [0]]])
        },
        'clean_labels': tf.zeros((2,), dtype=tf.int32)
    }

  @parameterized.parameters(
      (('annotator_labels', 'annotator_ids', 'annotator_features',
        'annotator_confidences'), (
            4,
            3,
            (10 + 5 + 2 + 1),
        )),
      (('annotator_labels',), (
          4,
          3,
          10,
      )),
  )
  def test_pi_generation(self, pi_subset, expected_pi_shape):

    def annotator_id_encoding_fn(example):
      return tf.one_hot(example['pi_features']['annotator_ids'],
                        self.num_dataset_annotators)

    encoding_fn_dict = {
        'annotator_ids':
            annotator_id_encoding_fn,
        'annotator_features':
            lambda e: e['pi_features']['annotator_features'],
        'annotator_confidences':
            lambda e: e['pi_features']['annotator_confidences'],
        'annotator_labels':
            lambda e: e['pi_features']['annotator_labels'],
    }

    privileged_information_fn = pi_utils.get_privileged_information_fn(
        pi_subset=pi_subset, encoding_fn_dict=encoding_fn_dict)
    privileged_information = privileged_information_fn(
        self.big_example_with_all_fields)
    self.assertEqual(privileged_information.shape, expected_pi_shape)

  def test_feature_repetition(self):
    num_annotators_per_example = 2
    labels = tf.constant([0, 1, 2])

    repeated_labels = pi_utils.repeat_across_annotators(
        labels, num_annotators_per_example)
    self.assertAllEqual(repeated_labels,
                        tf.constant([[[0], [0]], [[1], [1]], [[2], [2]]]))

    labels_one_hot = tf.constant([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    repeated_labels = pi_utils.repeat_across_annotators(
        labels_one_hot, num_annotators_per_example)

    self.assertAllEqual(
        repeated_labels,
        tf.constant([[[1, 0, 0], [1, 0, 0]], [[0, 1, 0], [0, 1, 0]],
                     [[0, 0, 1], [0, 0, 1]]]))

  def test_flatten_annotator_axis(self):
    annotator_labels = tf.constant([[[1, 0, 0], [1, 0, 0]],
                                    [[0, 1, 0], [0, 1, 0]],
                                    [[0, 0, 1], [0, 0, 1]]])
    flattened_labels = pi_utils.flatten_annotator_axis(annotator_labels)
    self.assertAllEqual(
        flattened_labels,
        tf.constant([[1, 0, 0], [1, 0, 0], [0, 1, 0], [0, 1, 0], [0, 0, 1],
                     [0, 0, 1]]))

  @parameterized.parameters(True, False)
  def test_find_noisy_annotators(self, flatten_annotators):
    example = {
        'pi_features': {
            'annotator_ids': tf.reshape(tf.range(4), [2, 2]),
            'annotator_labels': tf.constant([[[0], [1]], [[0], [0]]])
        },
        'clean_labels': tf.zeros((2,), dtype=tf.int32)
    }

    is_correct_mask = pi_utils.find_noisy_annotators(example,
                                                     flatten_annotators)

    # The second annotator of the first example is wrong
    ground_truth = tf.constant([[0, 1], [0, 0]])
    if flatten_annotators:
      ground_truth = tf.reshape(ground_truth, [-1])

    self.assertAllEqual(is_correct_mask, ground_truth)

  def test_annotator_label_if_incorrect(self):
    annotator_label_if_incorrect = (
        pi_utils.annotator_label_if_incorrect_encoding_fn(
            self.example_with_incorrect_labels, label_encoding_fn=None
        )
    )
    self.assertAllClose(
        tf.constant([[[-1], [1]], [[-1], [-1]]]), annotator_label_if_incorrect)

  def test_annotator_ids_encoding(self):
    annotator_ids = pi_utils.annotator_ids_encoding_fn(
        self.big_example_with_all_fields,
        num_dataset_annotators=self.num_dataset_annotators)

    self.assertAllEqual(
        tf.shape(annotator_ids),
        tf.constant([
            self.batch_size, self.num_annotators_per_example,
            self.num_dataset_annotators
        ]))

  def test_clean_labels_encoding_fn(self):

    def label_encoding_fn(labels):
      return tf.one_hot(
          tf.cast(labels, dtype=tf.int32), self.num_classes, dtype=tf.float32)

    clean_labels = pi_utils.clean_labels_encoding_fn(
        self.big_example_with_all_fields,
        num_annotators_per_example=self.num_annotators_per_example,
        label_encoding_fn=label_encoding_fn)

    self.assertAllEqual(
        tf.shape(clean_labels),
        tf.constant([
            self.batch_size, self.num_annotators_per_example, self.num_classes
        ]))


if __name__ == '__main__':
  tf.test.main()
