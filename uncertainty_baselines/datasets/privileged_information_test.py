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

"""Tests for privileged_information."""

from typing import Dict, Iterable
from unittest import mock

from absl.testing import parameterized
import numpy as np
from robustness_metrics.common import ops
import tensorflow as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import privileged_information
from uncertainty_baselines.datasets.test_utils import DatasetTest


class _DummyDataset(base.BaseDataset):
  """Dummy dataset builder class."""
  num_classes = 2

  def __init__(self,
               split: str,
               image_shape: Iterable[int] = (32, 32, 3),
               num_examples: int = 5,
               one_hot: bool = True):
    """Create a minimal mock of a base.BaseDataset.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      image_shape: the image shape for random images to be generated. By
        default, images are generated in the shape (32, 32, 3).
      num_examples: number of examples in split.
      one_hot: whether or not to use one-hot encoding for the labels.
    """
    self._split = split
    self._num_examples = num_examples
    self._image_shape = image_shape

    self._seed = [42, 0]
    self._is_training = split in ['train', tfds.Split.TRAIN]
    self._one_hot = one_hot
    features = tf.random.stateless_normal(
        (self._num_examples,) + self._image_shape,
        seed=self._seed,
        dtype=tf.float32)
    label = tf.zeros([self._num_examples], tf.int32)
    if self._one_hot:
      label = tf.one_hot(label, 2)
    self._dataset = tf.data.Dataset.from_tensor_slices({
        'features': features,
        'labels': label,
        'ids': tf.range(self._num_examples)
    })

    self._enumerate_id_key = '_enumerate_added_per_step_id'
    self._fingerprint_key = '_enumerate_added_example_id'
    self._drop_remainder = False

  def load(self, batch_size: int = -1, **unused_kwargs) -> tf.data.Dataset:
    self._seed, self._shuffle_seed = tf.random.experimental.stateless_split(
        self._seed, num=2)

    dataset = self._dataset

    dataset = dataset.enumerate()
    add_fingerprint_key_fn = self._add_enumerate_id(self._fingerprint_key)
    dataset = dataset.map(add_fingerprint_key_fn)

    if self._is_training:
      dataset = dataset.shuffle(
          self._num_examples,
          seed=tf.cast(self._shuffle_seed[0], tf.int64),
          reshuffle_each_iteration=True)
      dataset = dataset.repeat()

    dataset = dataset.enumerate()
    add_per_step_id_key_fn = self._add_enumerate_id(self._enumerate_id_key)

    preprocess_fn = self._create_process_example_fn()
    preprocess_fn = ops.compose(add_per_step_id_key_fn, self._create_element_id,
                                preprocess_fn)
    dataset = dataset.map(preprocess_fn)
    dataset = dataset.batch(batch_size)

    return dataset

  @property
  def num_examples(self):
    return self._num_examples

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example) -> Dict[str, tf.Tensor]:
      parsed_example = {}
      parsed_example['features'] = example['features']
      parsed_example['labels'] = example['labels']
      parsed_example['ids'] = example['ids']
      return parsed_example

    return _example_parser


class _DummyPIDataset(privileged_information.AnnotatorPIMixin, _DummyDataset):
  annotator_feature_length = 7
  available_annotators_per_example = 5

  def __init__(
      self,
      num_annotators_per_example=5,
      num_annotators_per_example_and_step=None,
      reliability_interval=(0., 1.),
      artificial_id_increase_factor=1,
      one_hot=False,
      **kwargs,
  ):
    self._dataset_builder = mock.create_autospec(tfds.core.DatasetBuilder)
    self._dataset_builder.info.num_classes = mock.MagicMock(return_value=2)

    super().__init__(  # pylint: disable=unexpected-keyword-arg
        num_annotators_per_example=num_annotators_per_example,
        num_annotators_per_example_and_step=num_annotators_per_example_and_step,
        reliability_interval=reliability_interval,
        artificial_id_increase_factor=artificial_id_increase_factor,
        one_hot=one_hot,
        **kwargs)

  @property
  def pi_feature_length(self):
    feature_length_dict = {
        'annotator_labels':
            1 if not self._one_hot else  # type: ignore
            self.num_classes,
        'annotator_ids':
            1,
        'annotator_features':
            self.annotator_feature_length
    }
    if self._random_pi_length and self._random_pi_length > 0:
      feature_length_dict.update({'random_pi': self._random_pi_length})
    return feature_length_dict

  @property
  def num_dataset_annotators(self):
    return self.available_annotators_per_example * 2

  def _process_pi_features_and_labels(self, example, unprocessed_example=None):
    # Create two groups of 5 annotators. The first group annotates the first
    # half of the dataset, and the second group the second half.
    annotator_ids = tf.range(
        0, self.available_annotators_per_example, dtype=tf.float32)
    idx_shift = self.available_annotators_per_example * (example['ids'] % 2)
    annotator_ids = annotator_ids + tf.cast(idx_shift, tf.float32)
    annotator_labels = tf.constant([0, 1, 0, 0, 0])
    if self._one_hot:  # type: ignore
      annotator_labels = tf.one_hot(
          annotator_labels, self.num_classes, dtype=tf.float32)
    else:
      annotator_labels = tf.expand_dims(
          tf.cast(annotator_labels, tf.float32), axis=1)
    pi_features = {
        'annotator_labels':
            annotator_labels,
        'annotator_ids':
            tf.expand_dims(annotator_ids, axis=1),
        'annotator_features':
            tf.random.normal((self.available_annotators_per_example,
                              self.annotator_feature_length),
                             dtype=tf.float32),
    }
    example['pi_features'] = pi_features
    example['clean_labels'] = example['labels']
    return example

  def _set_adversarial_pi_features(self, example, per_example_seed):
    adversarial_features = -3 * tf.ones(
        (self._num_adversarial_annotators_per_example,
         self.annotator_feature_length))
    adversarial_labels = tf.random.stateless_categorical(
        tf.ones(
            (self._num_adversarial_annotators_per_example, self.num_classes)),
        num_samples=1,
        seed=per_example_seed)

    if self._one_hot:
      adversarial_labels = tf.one_hot(
          adversarial_labels, self.num_classes, dtype=tf.float32)
      # Remove extra dummy label dimension:
      # adversarial_labels: (num_annotators, 1, 2) -> (num_annotators, 2)
      adversarial_labels = adversarial_labels[:, 0, :]
    else:
      adversarial_labels = tf.cast(adversarial_labels, tf.float32)

    return {
        'annotator_features': adversarial_features,
        'annotator_labels': adversarial_labels,
    }

  @property
  def _max_annotators_per_example(self):
    return 5


class PrivilegedInformationTest(DatasetTest, parameterized.TestCase):

  def test_pi_loading(self):
    num_annotators_per_example = 5
    builder = _DummyPIDataset(
        split='test', image_shape=(32, 32, 3), num_examples=5)
    dataset = builder.load(batch_size=3)
    self.assertEqual(
        list(dataset.element_spec.keys()),
        ['features', 'labels', 'ids', 'pi_features', 'clean_labels'])
    self.assertEqual(
        list(dataset.element_spec['pi_features'].keys()),
        ['annotator_labels', 'annotator_ids', 'annotator_features'])

    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_labels'],
        tf.TensorSpec(
            shape=(None, num_annotators_per_example, 1), dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_ids'],
        tf.TensorSpec(
            shape=(None, num_annotators_per_example, 1), dtype=tf.float32))
    self.assertEqual(
        dataset.element_spec['pi_features']['annotator_features'],
        tf.TensorSpec(
            shape=(None, num_annotators_per_example,
                   builder.annotator_feature_length),
            dtype=tf.float32))

  @parameterized.parameters(False, True)
  def test_reliability_estimation(self, one_hot):
    builder = _DummyPIDataset(
        split='train',
        image_shape=(32, 32, 3),
        num_examples=5,
        one_hot=one_hot,
        reliability_estimation_batch_size=2)
    self.assertEqual(builder.mean_reliability, 4 / 5)

    annotator_ids = np.arange(builder.num_dataset_annotators).tolist()
    annotators_reliability = {
        annotator_id: 1.0 for annotator_id in annotator_ids
    }
    # Annotator 1 and 6 are always incorrect in _DummyPIDataset.
    annotators_reliability[annotator_ids[1]] = 0.0
    annotators_reliability[annotator_ids[6]] = 0.0
    self.assertLen(builder.annotators_reliability.keys(),
                   builder.num_dataset_annotators)

    self.assertEqual(builder.annotators_reliability, annotators_reliability)

  @parameterized.parameters(
      ('uniform', True),
      ('best', True),
      ('worst', True),
      ('uniform', False),
      ('best', False),
      ('worst', False),
  )
  def test_annotator_per_example_sampling(self, strategy, one_hot):
    num_annotators_per_example = 4
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=5,
        one_hot=one_hot,
        annotator_sampling_strategy=strategy,
        num_annotators_per_example=num_annotators_per_example)

    dataset_iterator = iter(builder.load(batch_size=1).repeat())
    for n in range(builder.num_examples + 1):
      example = next(dataset_iterator)
      if n == 0:
        example_1 = example
      elif n == builder.num_examples:
        example_n = example

    self.assertListEqual(
        example_1['pi_features']['annotator_ids'].get_shape().as_list(),
        [1, num_annotators_per_example, 1])

    self.assertAllClose(example_1['pi_features']['annotator_ids'],
                        example_n['pi_features']['annotator_ids'])

    if strategy == 'best':
      # Before sampling: num_reliable_annotators=8 num_unreliable_annotators=2,
      # After sampling: num_reliable_annotators=8 num_unreliable_annotators=0.
      self.assertEqual(builder.mean_reliability, 1.0)
    elif strategy == 'worst':
      # Before sampling: num_reliable_annotators=8 num_unreliable_annotators=2.
      # After sampling: num_reliable_annotators=6 num_unreliable_annotators=2.
      self.assertEqual(builder.mean_reliability, 0.75)

  def test_annotator_per_example_and_step_sampling(self):
    num_annotators_per_example_and_step = 2
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=5,
        num_annotators_per_example=4,
        num_annotators_per_example_and_step=num_annotators_per_example_and_step)

    example = builder.load(batch_size=1).take(1).get_single_element()
    self.assertListEqual(
        example['pi_features']['annotator_ids'].get_shape().as_list(),
        [1, num_annotators_per_example_and_step, 1])

  def test_filter_reliability(self):
    builder_without_reliability_filter = _DummyPIDataset(
        split='test', image_shape=(32, 32, 3), num_examples=5)

    self.assertEqual(builder_without_reliability_filter.mean_reliability, 0.8)
    self.assertEqual(
        builder_without_reliability_filter.num_effective_annotators, 10)

    builder_with_reliability_filter = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=5,
        reliability_interval=(0.5, 1.))

    self.assertEqual(builder_with_reliability_filter.mean_reliability, 1.0)
    self.assertEqual(builder_with_reliability_filter.num_effective_annotators,
                     8)

  @parameterized.parameters(True, False)
  def test_annotator_padding(self, one_hot):
    num_annotators_per_example_and_step = 10
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=5,
        one_hot=one_hot,
        num_annotators_per_example=4,
        num_annotators_per_example_and_step=num_annotators_per_example_and_step)

    example = builder.load(batch_size=1).take(1).get_single_element()
    self.assertListEqual(
        example['pi_features']['annotator_ids'].get_shape().as_list(),
        [1, num_annotators_per_example_and_step, 1])

  def test_private_padding_function(self):
    num_annotators = 4
    num_desired_annotators = 20
    example = {
        'features': tf.random.normal([32, 32, 3]),
        'labels': tf.constant(1.),
        'pi_features': {
            'annotator_features': tf.random.normal([num_annotators, 7]),
            'annotator_ids': tf.ones([num_annotators, 1]),
            'annotator_labels': tf.zeros([num_annotators, 1]),
        }
    }
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=5,
        num_annotators_per_example=num_desired_annotators,
    )
    padded_example = builder._pad_annotators(example)
    padded_annotator_ids = padded_example['pi_features']['annotator_ids'][
        num_annotators:]
    self.assertListEqual(
        padded_annotator_ids.numpy().tolist(),
        [[-1.0] for _ in range(num_desired_annotators - num_annotators)])

    self.assertEqual(
        int(tf.shape(padded_example['pi_features']['annotator_features'])[0]),
        num_desired_annotators)

  def test_artificial_id_multiplication(self):
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=100,
        artificial_id_increase_factor=3)

    self.assertEqual(builder.num_effective_annotators, 30)

    example = builder.load(batch_size=30).take(1).get_single_element()
    annotator_ids = example['pi_features']['annotator_ids']
    self.assertEqual(int(tf.reduce_max(annotator_ids)), 29)
    self.assertEqual(int(tf.reduce_min(annotator_ids)), 0)

  def test_estimate_correct_per_annotator(self):
    num_unique_ids = 3

    # annotator_ids: [0, 0, 1, 1, 2, 2].
    annotator_ids = tf.repeat(tf.range(num_unique_ids), repeats=2)

    # Only one incorrect example in the batch.
    is_correct = tf.constant([1, 1, 0, 1, 1, 1])
    unique_ids, annotators_correct_count, annotators_count = (
        privileged_information._estimate_correct_per_annotator(
            annotator_ids, is_correct
        )
    )

    self.assertAllClose(tf.range(num_unique_ids), unique_ids)
    self.assertAllClose(tf.constant([2, 2, 2]), annotators_count)
    self.assertAllClose(tf.constant([2, 1, 2]), annotators_correct_count)

  def test_adversarial_annotator_addition(self):
    num_adversarial_annotators = 3
    num_annotators_per_example = 8
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=100,
        num_annotators_per_example=8,
        num_adversarial_annotators_per_example=num_adversarial_annotators)

    example = builder.load(batch_size=5).take(1).get_single_element()
    annotator_ids = example['pi_features']['annotator_ids']
    self.assertEqual(int(tf.reduce_max(annotator_ids)), 15)
    unreliable_annotator_idx = tf.where(annotator_ids == 15)
    unreliable_annotator_features = tf.gather_nd(
        example['pi_features']['annotator_features'], unreliable_annotator_idx)
    self.assertEqual(int(unreliable_annotator_features[0]), -3)
    self.assertEqual(
        example['pi_features']['annotator_features'].shape,
        (5, num_annotators_per_example, builder.annotator_feature_length))

  def test_random_pi_generation(self):
    random_pi_length = 7
    num_annotators_per_example = 3
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=1,
        num_annotators_per_example=num_annotators_per_example,
        num_adversarial_annotators_per_example=1,
        random_pi_length=7)

    example = builder.load(batch_size=1).take(1).get_single_element()
    random_pi = example['pi_features']['random_pi']
    self.assertEqual(random_pi.shape,
                     (1, num_annotators_per_example, random_pi_length))

  def test_disable_reliability_estimation(self):
    num_annotators_per_example = 3
    num_annotators_per_example_and_step = 2

    # We test that disabling the reliability estimation does not raise any
    # errors when used in combination with the allowed dataset features.
    builder = _DummyPIDataset(
        split='test',
        image_shape=(32, 32, 3),
        num_examples=1,
        num_annotators_per_example=num_annotators_per_example,
        num_annotators_per_example_and_step=num_annotators_per_example_and_step,
        annotator_sampling_strategy='worst',
        disable_reliability_estimation=True)

    _ = builder.load(batch_size=1).take(1).get_single_element()

    with self.assertRaises(ValueError):
      builder = _DummyPIDataset(
          split='test',
          image_shape=(32, 32, 3),
          num_examples=1,
          num_adversarial_annotators_per_example=1,
          num_annotators_per_example=num_annotators_per_example,
          num_annotators_per_example_and_step=num_annotators_per_example_and_step,
          annotator_sampling_strategy='worst',
          disable_reliability_estimation=True)

    with self.assertRaises(ValueError):
      builder = _DummyPIDataset(
          split='test',
          image_shape=(32, 32, 3),
          num_examples=1,
          reliability_interval=(0.5, 0.7),
          num_annotators_per_example=num_annotators_per_example,
          num_annotators_per_example_and_step=num_annotators_per_example_and_step,
          annotator_sampling_strategy='worst',
          disable_reliability_estimation=True)


if __name__ == '__main__':
  tf.test.main()
