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

"""CIFAR{10,100} dataset builders."""

from typing import Any, Dict, Optional, Union

from absl import logging
import numpy as np
from robustness_metrics.common import types
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augment_utils
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets.privileged_information import AnnotatorPIMixin

# We use the convention of using mean = np.mean(train_images, axis=(0,1,2))
# and std = np.std(train_images, axis=(0,1,2)).
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
# Previously we used std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
# which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype), however
# we change convention to use the std over the entire training set instead.

# Useful constants related to the CIFAR-PI datasets. They include the number of
# annotators in each dataset, the mean and std of the pi_features, as well as
# values assigned to the pi_features of the adversarial annotators in
# CIFAR10/100-N assuming they are very low performant. In general, the
# adversarial value of a feature is set to 3 times the 90-percentile of that PI
# feature across the dataset.
CIFAR10N_NUM_ANNOTATORS = 746
CIFAR10N_TIMES_MEAN = 46.6797
CIFAR10N_TIMES_STD = 21.1822
CIFAR10N_ADV_TIME = 231

CIFAR100N_NUM_ANNOTATORS = 518
CIFAR100N_TIMES_MEAN = 113.6694
CIFAR100N_TIMES_STD = 78.1860
CIFAR100N_ADV_TIME = 708

CIFAR10H_TIMES_MEAN = 1917.9472
CIFAR10H_TIMES_STD = 11348.22427
CIFAR10H_TRIALS_MEAN = 104.4998
CIFAR10H_ADV_TIME = 8535
CIFAR10H_TRIALS_STD = 60.6211
CIFAR10H_ADV_TRIAL = 0  # Smallest possible value of trial_idx.


def _tuple_dict_fn_converter(fn, *args):
  """transform output tuple from process_batch_fn into a dictionary."""

  def dict_fn(batch_dict):
    if 'pi_features' in batch_dict.keys():
      images, labels, mixup_weights, mixup_index = fn(
          *args,
          batch_dict['features'],
          batch_dict['labels'],
          return_weights=True)
      batch_dict.update({
          'features': images,
          'labels': labels,
          'clean_labels': labels,
          'mixup_weights': mixup_weights,
          'mixup_index': mixup_index
      })
      return batch_dict
    else:
      images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
      batch_dict.update({
          'features': images,
          'labels': labels,
      })
      return batch_dict

  return dict_fn


def _store_in_hash_table(dictionary, values_length, key_dtype, value_dtype):
  """Stores the pairs of (key,values) in dictionary into a DenseHashTable.

  Args:
    dictionary: Dictionary with keys and values to store in the table.
    values_length: The default length of the values.
    key_dtype: The key dtype.
    value_dtype: The value dtype.

  Returns:
    table: A DenseHashTable that stores keys and associated values.
  """
  table = tf.lookup.experimental.DenseHashTable(
      key_dtype=key_dtype,
      value_dtype=value_dtype,
      default_value=-tf.ones(values_length, value_dtype)
      if values_length is not None else -1,
      empty_key='',
      deleted_key='$')

  for k, v in dictionary.items():
    if values_length:
      padding_length = max(0, values_length - len(v))
      insert_value = v + [-1] * padding_length
    else:
      insert_value = v
    table.insert(k, insert_value)
  return table


def _unpad_annotations(x):
  """Removes padding from stored annotation."""
  x_ragged = tf.RaggedTensor.from_tensor([x], padding=-1)
  return x_ragged.to_tensor()[0]


def _stack_worker_annotations(example, prefix, feature_length, suffix=''):
  stacked_annotations = tf.expand_dims(
      tf.stack([example[prefix + str(n) + suffix] for n in range(1, 4)],
               axis=0),
      axis=1)
  return tf.reshape(stacked_annotations, [3, feature_length])


def _normalize_pi_feature(pi_value, mean, std):
  """Normalizes the values in pi_value using mean and std."""
  return (pi_value - mean) / std


def _is_derivative_of_split(split: Union[str, tfds.Split,
                                         tfds.core.ReadInstruction],
                            origin_split: str):
  """Checks if `split` is the same or a derived version from `origin_split`."""
  if isinstance(split, str):
    return split == origin_split
  elif isinstance(split, tfds.Split):
    split_equivalences = {
        tfds.Split.TRAIN: 'train',
        tfds.Split.TEST: 'test',
        tfds.Split.VALIDATION: 'validation'
    }
    return split_equivalences[split] == origin_split
  elif isinstance(split, tfds.core.ReadInstruction):
    return split.split_name == origin_split
  else:
    raise ValueError(
        'split must be of type string or tfds.core.ReadInstruction.')


class _CifarDataset(base.BaseDataset):
  """CIFAR dataset builder abstract class."""

  def __init__(self,
               name: str,
               fingerprint_key: str,
               split: str,
               seed: Optional[Union[int, tf.Tensor]] = None,
               validation_percent: float = 0.0,
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               drop_remainder: bool = False,
               mask_and_pad: bool = False,
               normalize: bool = True,
               try_gcs: bool = False,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               use_bfloat16: bool = False,
               aug_params: Optional[Dict[str, Any]] = None,
               is_training: Optional[bool] = None):
    """Create a CIFAR10 or CIFAR100 tf.data.Dataset builder.

    Args:
      name: the name of this dataset, either 'cifar10', 'cifar100', 'cifar10_n'
        or 'cifar100_n'.
      fingerprint_key: The name of the feature holding a string that will be
        used to create an element id using a fingerprinting function. If None,
        then `ds.enumerate()` is added before the `ds.map(preprocessing_fn)` is
        called and an `id` field is added to the example Dict.
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      seed: the seed used as a source of randomness.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      drop_remainder: Whether or not to drop the last batch of data if the
        number of points is not exactly equal to the batch size.
      mask_and_pad: Whether or not to mask and pad batches such that when
        drop_remainder == False, partial batches are padded to a full batch and
        an additional `mask` feature is added to indicate which examples are
        padding.
      normalize: whether or not to normalize each image by the CIFAR dataset
        mean and stddev.
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the tfds
        dataset_builder as a data_dir parameter.
      use_bfloat16: Whether or not to load the data in bfloat16 or float32.
      aug_params: hyperparameters for the data augmentation pre-processing.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    self._normalize = normalize
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(dataset_builder,
                                                  validation_percent, split)
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        seed=seed,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        mask_and_pad=mask_and_pad,
        fingerprint_key=fingerprint_key,
        download_data=download_data,
        cache=True)

    self._use_bfloat16 = use_bfloat16
    if aug_params is None:
      aug_params = {}
    self._adaptive_mixup = aug_params.get('adaptive_mixup', False)
    ensemble_size = aug_params.get('ensemble_size', 1)
    if self._adaptive_mixup and 'mixup_coeff' not in aug_params:
      # Hard target in the first epoch!
      aug_params['mixup_coeff'] = tf.ones([ensemble_size, 10])
    self._aug_params = aug_params

    mixup_alpha = self._aug_params.get('mixup_alpha', 0)
    label_smoothing = self._aug_params.get('label_smoothing', 0.)
    self._should_onehot = mixup_alpha > 0 or label_smoothing > 0

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: types.Features) -> types.Features:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image_dtype = tf.bfloat16 if self._use_bfloat16 else tf.float32
      use_augmix = self._aug_params.get('augmix', False)
      if self._is_training:
        image_shape = tf.shape(image)
        # Expand the image by 2 pixels, then crop back down to 32x32.
        image = tf.image.resize_with_crop_or_pad(image, image_shape[0] + 4,
                                                 image_shape[1] + 4)
        # Note that self._seed will already be shape (2,), as is required for
        # stateless random ops, and so will per_example_step_seed.
        per_example_step_seed = tf.random.experimental.stateless_fold_in(
            self._seed, example[self._enumerate_id_key])
        # per_example_step_seeds will be of size (num, 3).
        # First for random_crop, second for flip, third optionally for
        # RandAugment, and foruth optionally for Augmix.
        per_example_step_seeds = tf.random.experimental.stateless_split(
            per_example_step_seed, num=4)
        image = tf.image.stateless_random_crop(
            image, (image_shape[0], image_shape[0], 3),
            seed=per_example_step_seeds[0])
        image = tf.image.stateless_random_flip_left_right(
            image, seed=per_example_step_seeds[1])

        # Only random augment for now.
        if self._aug_params.get('random_augment', False):
          count = self._aug_params['aug_count']
          augment_seeds = tf.random.experimental.stateless_split(
              per_example_step_seeds[2], num=count)
          augmenter = augment_utils.RandAugment()
          augmented = [
              augmenter.distort(image, seed=augment_seeds[c])
              for c in range(count)
          ]
          image = tf.stack(augmented)

        if use_augmix:
          augmenter = augment_utils.RandAugment()
          image = augmix.do_augmix(
              image,
              self._aug_params,
              augmenter,
              image_dtype,
              mean=CIFAR10_MEAN,
              std=CIFAR10_STD,
              seed=per_example_step_seeds[3])

      # The image has values in the range [0, 1].
      # Optionally normalize by the dataset statistics.
      if not use_augmix:
        if self._normalize:
          image = augmix.normalize_convert_image(
              image, image_dtype, mean=CIFAR10_MEAN, std=CIFAR10_STD)
        else:
          image = tf.image.convert_image_dtype(image, image_dtype)
      parsed_example = {'features': image}
      parsed_example[self._enumerate_id_key] = example[self._enumerate_id_key]
      if self._add_fingerprint_key:
        parsed_example[self._fingerprint_key] = example[self._fingerprint_key]

      # Note that labels are always float32, even when images are bfloat16.
      labels = example['label']

      if self._should_onehot:
        num_classes = 100 if self.name in ['cifar100', 'cifar100_n'] else 10
        parsed_example['labels'] = tf.one_hot(
            labels, num_classes, dtype=tf.float32)
      else:
        parsed_example['labels'] = tf.cast(labels, tf.float32)

      if self.name == 'cifar10_n':
        parsed_example = self._prepare_parsed_example_cifar10n(
            example, parsed_example)
      elif self.name == 'cifar100_n':
        parsed_example = self._prepare_parsed_example_cifar100n(
            example, parsed_example)

      return parsed_example

    return _example_parser

  def _prepare_parsed_example_cifar10n(self, example, parsed_example):

    if self._should_onehot:
      parse_example_fn = lambda e: tf.one_hot(e, 10, dtype=tf.float32)
    else:
      parse_example_fn = lambda e: tf.cast(e, tf.float32)

    parsed_example['worse_labels'] = parse_example_fn(example['worse_label'])
    parsed_example['aggre_labels'] = parse_example_fn(example['aggre_label'])
    for key in ['random_label1', 'random_label2', 'random_label3']:
      parsed_example[key] = parse_example_fn(example[key])

    for key in [
        'worker1_id', 'worker2_id', 'worker3_id', 'worker1_time',
        'worker2_time', 'worker3_time'
    ]:
      parsed_example[key] = tf.cast(example[key], dtype=tf.float32)

    return parsed_example

  def _prepare_parsed_example_cifar100n(self, example, parsed_example):
    if self._should_onehot:
      parsed_example['noise_labels'] = tf.one_hot(
          example['noise_label'], 100, dtype=tf.float32)
    else:
      parsed_example['noise_labels'] = tf.cast(example['noise_label'],
                                               tf.float32)
    parsed_example['worker_ids'] = tf.cast(example['worker_id'], tf.float32)
    parsed_example['worker_times'] = tf.cast(example['worker_time'], tf.float32)
    return parsed_example

  def _create_process_batch_fn(self,
                               batch_size: int) -> Optional[base.PreProcessFn]:
    if self._is_training and self._aug_params.get('mixup_alpha', 0) > 0:
      if self._adaptive_mixup:
        return _tuple_dict_fn_converter(augmix.adaptive_mixup, batch_size,
                                        self._aug_params)
      else:
        return _tuple_dict_fn_converter(augmix.mixup, batch_size,
                                        self._aug_params)
    return None


class Cifar10Dataset(_CifarDataset):
  """CIFAR10 dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='cifar10', fingerprint_key='id', **kwargs)


class Cifar100Dataset(_CifarDataset):
  """CIFAR100 dataset builder class."""

  def __init__(self, **kwargs):
    super().__init__(name='cifar100', fingerprint_key='id', **kwargs)


class Cifar10CorruptedDataset(_CifarDataset):
  """CIFAR10-C dataset builder class."""

  def __init__(self, corruption_type: str, severity: int, **kwargs):
    """Create a CIFAR10-C tf.data.Dataset builder.

    Args:
      corruption_type: Corruption name.
      severity: Corruption severity, an integer between 1 and 5.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(
        name=f'cifar10_corrupted/{corruption_type}_{severity}',
        fingerprint_key=None,
        **kwargs)  # pytype: disable=wrong-arg-types  # kwargs-checking


class Cifar10HDataset(AnnotatorPIMixin, _CifarDataset):
  """CIFAR10H dataset builder class."""

  def __init__(self,
               annotations_path='/path/to/cifar10h-raw.csv',
               num_annotators_per_example=None,
               normalize_pi_features=True,
               **kwargs):
    """Create a CIFAR10-H tf.data.Dataset builder.


    Args:
      annotations_path: Path to CIFAR10-H annotations CSV file which can be
        downloaded from:
        github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-raw.zip.
      num_annotators_per_example: Number of annotators loaded per example. If
        None, it loads the maximum number of annotators available per example,
        padding with -1 if necessary.
      normalize_pi_features: Whether the annotator_features, i.e.,
        annotator_times and trial_idx, are normalized based on their global mean
        and std.
      **kwargs: Additional keyword arguments.
    """
    split = kwargs.get('split', 'test')
    if not _is_derivative_of_split(split, 'test'):
      raise ValueError('Cifar-10H is only defined on the test set.')

    self._annotations_path = annotations_path
    self._cifar10h_annotations = self._get_cifar10h_annotations()
    self._normalize_pi_features = normalize_pi_features
    super().__init__(  # pylint: disable=unexpected-keyword-arg
        name='cifar10',
        fingerprint_key='id',
        num_annotators_per_example=num_annotators_per_example,
        **kwargs)

  def _get_cifar10h_annotations(self):
    """Load the CIFAR10-H dataset."""
    with tf.io.gfile.GFile(self._annotations_path, 'r') as f:
      # Read from CIFAR10-H annotations file as described in
      # https://github.com/jcpeterson/cifar-10h/blob/master/data/cifar10h-raw.zip
      csv_reader = f.readlines()[1:]

      annotations = {
          'annotator_labels': {},
          'trial_idx': {},
          'annotator_times': {},
          'annotator_ids': {},
          'count': {},
      }
      annotators = []
      max_count = 0
      for row in csv_reader:
        split_row = row.split(',')
        annotator_id = int(split_row[0])
        trial_idx = int(split_row[1])
        chosen_label = int(split_row[6])
        test_idx = split_row[8]
        annotator_times = float(split_row[11])
        if test_idx != '-99999':
          test_id = 'test_' + '0' * (5 - len(test_idx)) + test_idx
          if test_id not in annotations['count']:
            annotations['annotator_labels'][test_id] = [chosen_label]
            annotations['trial_idx'][test_id] = [trial_idx]
            annotations['annotator_times'][test_id] = [annotator_times]
            annotations['annotator_ids'][test_id] = [annotator_id]
            annotations['count'][test_id] = 1
          else:
            annotations['annotator_labels'][test_id].append(chosen_label)
            annotations['trial_idx'][test_id].append(trial_idx)
            annotations['annotator_times'][test_id].append(annotator_times)
            annotations['annotator_ids'][test_id].append(annotator_id)
            annotations['count'][test_id] += 1

          if annotations['count'][test_id] > max_count:
            max_count = annotations['count'][test_id]

          annotators.append(annotator_id)

      self._num_dataset_annotators = len(list(set(annotators)))
      self._max_annotator_count = max_count

    annotations_tables = {}
    for key in annotations:
      if key != 'count':
        annotations_tables[key] = _store_in_hash_table(
            dictionary=annotations[key],
            values_length=self._max_annotator_count,
            key_dtype=tf.string,
            value_dtype=tf.float32)

    return annotations_tables

  def _process_pi_features_and_labels(self, example, unprocessed_example):  # pytype: disable=signature-mismatch  # overriding-parameter-count-checks
    """Loads 'annotator_ids', 'annotator_labels', 'annotator_times', and 'trial_idx', and sets 'clean_labels' and 'labels'.

    In CIFAR10-H the `labels` field is popoulated with the average label of its
    annotators.

    Args:
      example: Example to which the pi_features are appended.
      unprocessed_example: Example fields prior to before being processed by
        `_example_parser`.

    Returns:
      The example with its processed pi_features.
    """

    # Copy only the required fields.
    parsed_example = {
        'labels': example['labels'],
        'features': example['features'],
    }

    example_id = unprocessed_example['id']
    parsed_example[self._enumerate_id_key] = example[self._enumerate_id_key]
    if self._add_fingerprint_key:
      parsed_example[self._fingerprint_key] = example[self._fingerprint_key]

    # Save clean label.
    parsed_example['clean_labels'] = example['labels']

    # Relabel with average label.
    annotator_labels = _unpad_annotations(
        self._cifar10h_annotations['annotator_labels'][example_id])
    annotator_labels_one_hot = tf.one_hot(
        tf.cast(annotator_labels, dtype=tf.int32), 10, dtype=tf.float32)

    # Set `labels` to the average over the `annotator_labels`.
    parsed_example['labels'] = tf.reshape(
        tf.reduce_mean(annotator_labels_one_hot, axis=0), [10])

    if self._should_onehot:
      annotator_labels = annotator_labels_one_hot
    else:
      annotator_labels = tf.expand_dims(
          tf.cast(annotator_labels, tf.float32), axis=1)

    def reshape_pi_feature(pi_feature):
      return tf.expand_dims(
          tf.cast(_unpad_annotations(pi_feature), tf.float32), axis=1)

    annotator_times = reshape_pi_feature(
        self._cifar10h_annotations['annotator_times'][example_id])
    trial_idx = reshape_pi_feature(
        self._cifar10h_annotations['trial_idx'][example_id])
    if self._normalize_pi_features:
      annotator_times = _normalize_pi_feature(annotator_times,
                                              CIFAR10H_TIMES_MEAN,
                                              CIFAR10H_TIMES_STD)
      trial_idx = _normalize_pi_feature(trial_idx, CIFAR10H_TRIALS_MEAN,
                                        CIFAR10H_TRIALS_STD)

    parsed_example['pi_features'] = {
        'annotator_labels':
            annotator_labels,
        'annotator_ids':
            reshape_pi_feature(
                self._cifar10h_annotations['annotator_ids'][example_id]),
        'annotator_times':
            annotator_times,
        'trial_idx':
            trial_idx,
    }

    return parsed_example

  def _hash_fingerprint_int(self, fingerprint: Any) -> int:
    return tf.strings.to_hash_bucket_fast(fingerprint, self.num_examples)

  @property
  def num_dataset_annotators(self):
    return self._num_dataset_annotators

  @property
  def pi_feature_length(self):
    feature_length_dict = {
        'annotator_ids': 1,
        'annotator_labels': 10 if self._should_onehot else 1,  # type: ignore
        'annotator_times': 1,
        'trial_idx': 1,
    }
    if self._random_pi_length is not None:
      if self._random_pi_length <= 0:
        raise ValueError('random_pi_length must be greater than 0.')
      feature_length_dict.update({'random_pi': self._random_pi_length})
    return feature_length_dict

  def _set_adversarial_pi_features(self, example, per_example_seed):

    # The adversarial annotators are assummed to be extremely low perfomant. In
    # this regard, we set their annotator_time to be really high in comparison
    # to the average annotator_time of the dataset.
    adversarial_labels = tf.random.stateless_categorical(
        tf.ones((self._num_adversarial_annotators_per_example, 10)),
        num_samples=1,
        seed=per_example_seed)
    if self._should_onehot:
      adversarial_labels = tf.one_hot(
          adversarial_labels, self.info.num_classes, dtype=tf.float32)
      # Remove extra dummy label dimension:
      # adversarial_labels: (num_annotators, 1, 100) -> (num_annotators, 100)
      adversarial_labels = tf.reshape(
          adversarial_labels,
          [self._num_adversarial_annotators_per_example, self.info.num_classes])
    else:
      adversarial_labels = tf.cast(adversarial_labels, tf.float32)

    adversarial_ids = tf.reshape(
        tf.range(
            self.num_dataset_annotators,
            self.num_dataset_annotators +
            self._num_adversarial_annotators_per_example,
            dtype=tf.float32),
        [self._num_adversarial_annotators_per_example, 1])

    adversarial_times = tf.reshape(
        tf.ones((self._num_adversarial_annotators_per_example, 1),
                dtype=tf.float32) * CIFAR10H_ADV_TIME,
        [self._num_adversarial_annotators_per_example, 1])

    adversarial_trial_idx = tf.reshape(
        tf.ones((self._num_adversarial_annotators_per_example, 1),
                dtype=tf.float32) * CIFAR10H_ADV_TRIAL,
        [self._num_adversarial_annotators_per_example, 1])

    if self._normalize_pi_features:
      adversarial_times = _normalize_pi_feature(adversarial_times,
                                                CIFAR10H_TIMES_MEAN,
                                                CIFAR10H_TIMES_STD)
      adversarial_trial_idx = _normalize_pi_feature(adversarial_trial_idx,
                                                    CIFAR10H_TRIALS_MEAN,
                                                    CIFAR10H_TRIALS_STD)

    return {
        'annotator_labels': adversarial_labels,
        'annotator_ids': adversarial_ids,
        'annotator_times': adversarial_times,
        'trial_idx': adversarial_trial_idx,
    }

  @property
  def _max_annotators_per_example(self):
    return self._max_annotator_count


class _CifarNDataset(AnnotatorPIMixin, _CifarDataset):
  """CIFAR-N dataset builder class."""

  def __init__(self,
               name,
               normalize_pi_features=True,
               reliability_estimation_batch_size=4096,
               **kwargs):
    """Create a CIFAR-N tf.data.Dataset builder.

    Args:
      name: Dataset name. Either 'cifar10_n' or 'cifar100_n'.
      normalize_pi_features: Whether the annotator_features, i.e.,
        annotator_times and trial_idx, are normalized based on their global mean
        and std.
      reliability_estimation_batch_size: Number of examples that are loaded in
        parallel when estimating the reliability of the dataset.
      **kwargs: Additional keyword arguments.
    """

    split = kwargs.get('split', 'train')
    if not _is_derivative_of_split(split, 'train'):
      raise ValueError('Cifar-N is only defined on the test set.')

    self._normalize_pi_features = normalize_pi_features

    super().__init__(  # pylint: disable=unexpected-keyword-arg
        name=name,
        fingerprint_key='id',
        reliability_estimation_batch_size=reliability_estimation_batch_size,
        **kwargs)

  @property
  def pi_feature_length(self):
    feature_length_dict = {
        'annotator_ids': 1,
        'annotator_labels': self.info.num_classes
                            if self._should_onehot else 1,  # type: ignore
        'annotator_times': 1,
    }
    if self._random_pi_length is not None:
      if self._random_pi_length <= 0:
        raise ValueError('random_pi_length must be greater than 0.')
      feature_length_dict.update({'random_pi': self._random_pi_length})
    return feature_length_dict

  def _set_adversarial_pi_features(self, example, per_example_seed):
    # In Cifar-N the adversarial annotators are assumed to be low performant,
    # taking much longer to annotate the examples, than a real annotator.

    adversarial_labels = tf.random.stateless_categorical(
        tf.ones((self._num_adversarial_annotators_per_example,
                 self.info.num_classes)),
        num_samples=1,
        seed=per_example_seed)
    if self._should_onehot:  # type: ignore
      adversarial_labels = tf.one_hot(
          adversarial_labels, self.info.num_classes, dtype=tf.float32)
      # Remove extra dummy label dimension:
      # adversarial_labels: (num_annotators, 1, num_classes)
      #                  -> (num_annotators, num_classes).
      adversarial_labels = tf.squeeze(adversarial_labels, axis=1)
    else:
      adversarial_labels = tf.cast(adversarial_labels, tf.float32)

    adv_time_constant = (
        CIFAR10N_ADV_TIME if self.name == 'cifar10_n' else CIFAR100N_ADV_TIME
    )
    adversarial_times = tf.reshape(
        tf.ones((self._num_adversarial_annotators_per_example, 1),
                dtype=tf.float32) * adv_time_constant,
        [self._num_adversarial_annotators_per_example, 1])

    if self._normalize_pi_features:
      time_mean = (
          CIFAR10N_TIMES_MEAN if self.name == 'cifar10_n'
          else CIFAR100N_TIMES_MEAN
      )
      time_std = (
          CIFAR10N_TIMES_STD if self.name == 'cifar10_n'
          else CIFAR100N_TIMES_STD
      )
      adversarial_times = _normalize_pi_feature(adversarial_times, time_mean,
                                                time_std)

    return {
        'annotator_labels': adversarial_labels,
        'annotator_times': adversarial_times,
    }

  def _hash_fingerprint_int(self, fingerprint: Any) -> int:
    return tf.strings.to_hash_bucket_fast(fingerprint, self.num_examples)


class Cifar10NDataset(_CifarNDataset):
  """CIFAR10-N dataset builder class."""

  def __init__(self,
               supervised_label='aggre_labels',
               num_annotators_per_example=3,
               reliability_estimation_batch_size=4096,
               normalize_pi_features=True,
               **kwargs):
    """Create a CIFAR-10N tf.data.Dataset builder.

    Args:
      supervised_label: Key of the field to use as label for the supervision. To
        select from ('aggre_labels', 'worse_labels', 'clean_labels').
      num_annotators_per_example: Number of annotators loaded per example.
      reliability_estimation_batch_size: Number of examples that are loaded in
        parallel when estimating the reliability of the dataset.
      normalize_pi_features: Whether the annotator_features, i.e.,
        annotator_times, are normalized based on their global mean and std.
      **kwargs: Additional keyword arguments.
    """

    self._supervised_label = supervised_label
    super().__init__(  # pylint: disable=unexpected-keyword-arg
        name='cifar10_n',
        normalize_pi_features=normalize_pi_features,
        num_annotators_per_example=num_annotators_per_example,
        **kwargs)

  def _process_pi_features_and_labels(self, example, unprocessed_example=None):
    """Loads 'annotator_ids', 'annotator_labels', 'annotator_times', and sets 'clean_labels' and 'labels'."""

    # Copy only the required fields.
    parsed_example = {
        'labels': example['labels'],
        'features': example['features'],
        'aggre_labels': example['aggre_labels'],
        'worse_labels': example['worse_labels'],
    }
    parsed_example[self._enumerate_id_key] = example[self._enumerate_id_key]
    if self._add_fingerprint_key:
      parsed_example[self._fingerprint_key] = example[self._fingerprint_key]

    # Save clean label.
    parsed_example['clean_labels'] = example['labels']

    # Relabel with label_key.
    parsed_example['labels'] = parsed_example[self._supervised_label]

    annotator_times = _stack_worker_annotations(
        example,
        prefix='worker',
        suffix='_time',
        feature_length=self.pi_feature_length['annotator_times'])
    if self._normalize_pi_features:
      annotator_times = _normalize_pi_feature(annotator_times,
                                              CIFAR10N_TIMES_MEAN,
                                              CIFAR10N_TIMES_STD)

    parsed_example['pi_features'] = {
        'annotator_labels':
            _stack_worker_annotations(
                example,
                prefix='random_label',
                feature_length=self.pi_feature_length['annotator_labels']),
        'annotator_ids':
            _stack_worker_annotations(
                example,
                prefix='worker',
                suffix='_id',
                feature_length=self.pi_feature_length['annotator_ids']),
        'annotator_times':
            annotator_times,
    }

    return parsed_example

  @property
  def num_dataset_annotators(self):
    return CIFAR10N_NUM_ANNOTATORS

  @property
  def _max_annotators_per_example(self):
    return 3


class Cifar100NDataset(_CifarNDataset):
  """CIFAR100-N dataset builder class."""

  def __init__(self,
               reliability_estimation_batch_size=4096,
               normalize_pi_features=True,
               **kwargs):
    """Create a CIFAR-100N tf.data.Dataset builder.

    Args:
      reliability_estimation_batch_size: Number of examples that are loaded in
        parallel when estimating the reliability of the dataset.
      normalize_pi_features: Whether the annotator_features, i.e.,
        annotator_times, are normalized based on their global mean and std.
      **kwargs: Additional keyword arguments.
    """

    if 'num_annotators_per_example' in kwargs.keys():
      del kwargs['num_annotators_per_example']
      logging.warning(
          'Cifar-100N does only support loading one annotator per example.'
          ' Ignoring provided argument and setting'
          ' num_annotators_per_example=1.'
      )
    if 'num_annotators_per_example_and_step' in kwargs.keys():
      del kwargs['num_annotators_per_example_and_step']
      logging.warning(
          'Cifar-100N does only support loading one annotator per example.'
          ' Ignoring provided argument and setting'
          ' num_annotators_per_example_and_step=1.'
      )

    super().__init__(  # pylint: disable=unexpected-keyword-arg
        name='cifar100_n',
        normalize_pi_features=normalize_pi_features,
        num_annotators_per_example=1,
        num_annotators_per_example_and_step=1,
        **kwargs)

  def _process_pi_features_and_labels(self, example, unprocessed_example=None):
    """Loads 'annotator_ids', 'annotator_labels', 'annotator_times', and sets 'clean_labels'."""

    # Copy only the required fields.
    parsed_example = {
        'labels': example['labels'],
        'features': example['features'],
    }
    parsed_example[self._enumerate_id_key] = example[self._enumerate_id_key]
    if self._add_fingerprint_key:
      parsed_example[self._fingerprint_key] = example[self._fingerprint_key]

    # Save clean label.
    parsed_example['clean_labels'] = example['labels']

    # Add pi_features including annotator axis with dimension 1.
    # num_annotators_per_example=1 always in CIFAR100-N.
    annotator_times = tf.reshape(example['worker_times'], [1, 1])
    if self._normalize_pi_features:
      annotator_times = _normalize_pi_feature(annotator_times,
                                              CIFAR100N_TIMES_MEAN,
                                              CIFAR100N_TIMES_STD)
    parsed_example['pi_features'] = {
        'annotator_labels':
            tf.reshape(example['noise_labels'],
                       [1, self.pi_feature_length['annotator_labels']]),
        'annotator_ids':
            tf.reshape(example['worker_ids'], [1, 1]),
        'annotator_times':
            annotator_times,
    }

    return parsed_example

  @property
  def num_dataset_annotators(self):
    return CIFAR100N_NUM_ANNOTATORS

  @property
  def _max_annotators_per_example(self):
    return 1
