# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""ImageNet dataset builder.

We have an option to use a percent of the training dataset as a validation set,
and treat the original validation set as the test set. This is similar to what
is also done in the NeurIPS uncertainty benchmark paper
https://arxiv.org/abs/1906.02530 (which used (100 / 1024)% as a validation set).
"""
import csv
from typing import Any, Dict, Optional, Union

from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import inception_preprocessing
from uncertainty_baselines.datasets import resnet_preprocessing
from uncertainty_baselines.datasets.privileged_information import AnnotatorPIMixin

# ImageNet statistics. Used to normalize the input to Efficientnet. Note that
# these do NOT have `* 255.` after them.
IMAGENET_MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
IMAGENET_STDDEV = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)

# Number of parameters (normalized by mean and std over the training set)
# assigned to the adversarial annotators in ImageNet-PI. This value is much
# lower than the typical values in the training set.
ADVERSARIAL_NUM_PARAMS_NORMALIZED = -3


def _tuple_dict_fn_converter(fn, *args):

  def dict_fn(batch_dict):
    images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
    return {'features': images, 'labels': labels}

  return dict_fn


def _store_in_hash_table(keys, values, values_length, key_dtype, value_dtype):
  """Stores the pairs of (key, value) in a DenseHashTable for fast lookup.

  Args:
    keys: The set of keys to store.
    values: The set of associated values to store.
    values_length: The default length of the values.
    key_dtype: The key dtype.
    value_dtype: The value dtype.

  Returns:
    table: A DenseHashTable that stores keys and associated values.
  """
  table = tf.lookup.experimental.DenseHashTable(
      key_dtype=key_dtype,
      value_dtype=value_dtype,
      default_value=tf.zeros(values_length, value_dtype),
      empty_key='',
      deleted_key='$')

  table.insert(keys, values)
  return table


class _ImageNetDataset(base.BaseDataset):
  """ImageNet dataset builder abstract class."""

  def __init__(self,
               name: str,
               split: str,
               seed: Optional[Union[int, tf.Tensor]] = None,
               validation_percent: float = 0.0,
               shuffle_buffer_size: Optional[int] = 16384,
               num_parallel_parser_calls: int = 64,
               drop_remainder: bool = False,
               mask_and_pad: bool = False,
               try_gcs: bool = False,
               download_data: bool = False,
               data_dir: Optional[str] = None,
               is_training: Optional[bool] = None,
               preprocessing_type: str = 'resnet',
               use_bfloat16: bool = False,
               normalize_input: bool = False,
               image_size: int = 224,
               resnet_preprocessing_resize_method: Optional[str] = None,
               ensemble_size: int = 1,
               one_hot: bool = False,
               mixup_params: Optional[Dict[str, Any]] = None,
               run_mixup: bool = False,
               include_file_name: bool = False):
    """Create an ImageNet tf.data.Dataset builder.

    Args:
      name: the name of this dataset.
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
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the tfds
        dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      preprocessing_type: Which type of preprocessing to apply, either
        'inception' or 'resnet'.
      use_bfloat16: Whether or not to use bfloat16 or float32 images.
      normalize_input: Whether or not to normalize images by the ImageNet mean
        and stddev.
      image_size: The size of the image in pixels.
      resnet_preprocessing_resize_method: Optional string for the resize method
        to use for resnet preprocessing.
      ensemble_size: `int` for number of ensemble members used in Mixup.
      one_hot: whether or not to use one-hot labels.
      mixup_params: hparams of mixup.
      run_mixup: An explicit flag of whether or not to run mixup if
        `mixup_params['mixup_alpha'] > 0`. By default, mixup will only be run in
        training mode if `mixup_params['mixup_alpha'] > 0`.
      include_file_name: Whether or not to include a string file_name field in
        each example. Since this field is a string, it is not compatible with
        TPUs.
    """
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder,
        validation_percent,
        split,
        test_split=tfds.Split.VALIDATION)
    if preprocessing_type == 'inception':
      decoders = {
          'image': tfds.decode.SkipDecoding(),
      }
    else:
      decoders = None
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        drop_remainder=drop_remainder,
        mask_and_pad=mask_and_pad,
        fingerprint_key='file_name',
        download_data=download_data,
        decoders=decoders,
    )
    self._preprocessing_type = preprocessing_type
    self._use_bfloat16 = use_bfloat16
    self._normalize_input = normalize_input
    self._image_size = image_size
    self._resnet_preprocessing_resize_method = (
        resnet_preprocessing_resize_method
    )
    self._run_mixup = run_mixup

    self.ensemble_size = ensemble_size
    self._one_hot = one_hot
    if mixup_params is None:
      mixup_params = {}
    self._mixup_params = mixup_params
    self._include_file_name = include_file_name

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses ImageNet image Tensors."""
      per_example_step_seed = tf.random.experimental.stateless_fold_in(
          self._seed, example[self._enumerate_id_key])
      if self._preprocessing_type == 'inception':
        # `inception_preprocessing.preprocess_image` returns images in [-1, 1].
        image = inception_preprocessing.preprocess_image(
            example['image'],
            height=self._image_size,
            width=self._image_size,
            seed=per_example_step_seed,
            is_training=self._is_training)
        # Rescale to [0, 1].
        image = (image + 1.0) / 2.0
      elif self._preprocessing_type == 'resnet':
        # `resnet_preprocessing.preprocess_image` returns images in [0, 1].
        image = resnet_preprocessing.preprocess_image(
            image_bytes=example['image'],
            is_training=self._is_training,
            use_bfloat16=self._use_bfloat16,
            image_size=self._image_size,
            seed=per_example_step_seed,
            resize_method=self._resnet_preprocessing_resize_method)
      else:
        raise ValueError(
            'Invalid preprocessing type, must be one of "inception" or '
            '"resnet", received {}.'.format(self._preprocessing_type))

      if self._normalize_input:
        image = (tf.cast(image, tf.float32) - IMAGENET_MEAN) / IMAGENET_STDDEV
      if self._use_bfloat16:
        image = tf.cast(image, tf.bfloat16)

      # Note that labels are always float32, even when images are bfloat16.
      if self._one_hot:
        label = tf.one_hot(example['label'], 1000, dtype=tf.float32)
      else:
        label = tf.cast(example['label'], tf.float32)
      parsed_example = {
          'features': image,
          'labels': label,
      }
      if self._include_file_name and 'file_name' in example:
        parsed_example['file_name'] = example['file_name']
      return parsed_example

    return _example_parser

  def _create_process_batch_fn(self,
                               batch_size: int) -> Optional[base.PreProcessFn]:
    mixup_alpha = self._mixup_params.get('mixup_alpha', 0.0)
    if (self._is_training or self._run_mixup) and mixup_alpha > 0.0:
      same_mix_weight_per_batch = self._mixup_params.get(
          'same_mix_weight_per_batch', False)
      use_truncated_beta = self._mixup_params.get('use_truncated_beta', True)
      use_random_shuffling = self._mixup_params.get('use_random_shuffling',
                                                    False)
      if self._mixup_params.get('adaptive_mixup', False):
        if 'mixup_coeff' not in self._mixup_params:
          # Hard target in the first epoch!
          if ('ensemble_size' not in self._mixup_params or
              'num_classes' not in self._mixup_params):
            raise ValueError(
                'Missing "ensemble_size" and/or "num_classes" key from '
                'mixup_params, received {}.'.format(self._mixup_params))
          self._mixup_params['mixup_coeff'] = tf.ones(
              (self._mixup_params['ensemble_size'],
               self._mixup_params['num_classes']))
        return _tuple_dict_fn_converter(augmix.adaptive_mixup, batch_size,
                                        self._mixup_params)
      else:
        aug_params = {
            'mixup_alpha': mixup_alpha,
            'same_mix_weight_per_batch': same_mix_weight_per_batch,
            'use_truncated_beta': use_truncated_beta,
            'use_random_shuffling': use_random_shuffling,
        }
        return _tuple_dict_fn_converter(augmix.mixup, batch_size, aug_params)

    return None


class ImageNetDataset(_ImageNetDataset):
  """ImageNet dataset builder class."""

  # NOTE: Existing code passes in a split string as a positional argument, so
  # included here to preserve that behavior.
  def __init__(self, split, **kwargs):
    """Create an ImageNet tf.data.Dataset builder.

    Args:
      split: A dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      **kwargs: Additional keyword arguments.
    """
    super().__init__(name='imagenet2012', split=split, **kwargs)


class ImageNetPIDataset(AnnotatorPIMixin, _ImageNetDataset):
  """ImageNet dataset builder class with access to additional privileged information from a suite of model annotators."""

  def __init__(self,
               split: str,
               annotations_path: str,
               num_annotators_per_example: int = 16,
               reliability_estimation_batch_size: int = 1024,
               **kwargs):
    """Create an ImageNet-PI tf.data.Dataset builder.

    Args:
      split: A dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDATION, TEST] or their lowercase string
        names.
      annotations_path: Path to the directory containing the labels/{split}.csv
        (with annotator annotations for each example in {split}),
        confidences/{split}.csv (with the confidence of each annotator in its
        annotation) and annotator_features.csv (with meta-data about the
        annotators themselves).
      num_annotators_per_example: Number of annotators loaded per example.
      reliability_estimation_batch_size: Number of examples that are loaded in
        parallel when estimating the reliability of the dataset.
      **kwargs: Additional keyword arguments.
    """
    if 'include_file_name' in kwargs:
      logging.warn(
          'ImageNet-PI requires to load the file_name for every example.'
          " Ignoring the supplied 'include_file_name'"
      )
      del kwargs['include_file_name']

    self._annotations_path = annotations_path
    self._split_annotations_file = (
        'validation.csv' if split in ['test', tfds.Split.TEST] else 'train.csv'
    )

    (
        self._annotator_tables,
        self._num_dataset_annotators,
        self._annotator_feature_length,
    ) = self._get_annotator_features_tables()
    super().__init__(  # pylint: disable=unexpected-keyword-arg
        name='imagenet2012',
        split=split,
        include_file_name=True,
        num_annotators_per_example=num_annotators_per_example,
        reliability_estimation_batch_size=reliability_estimation_batch_size,
        **kwargs,
    )

  @property
  def pi_feature_length(self) -> Dict[str, int]:
    feature_length_dict = {
        'annotator_ids': 1,
        'annotator_labels': 1000 if self._one_hot else 1,
        'annotator_features': self._annotator_feature_length,
        'annotator_confidences': 1
    }
    if self._random_pi_length is not None:
      if self._random_pi_length <= 0:
        raise ValueError('random_pi_length must be greater than 0.')
      feature_length_dict.update({'random_pi': self._random_pi_length})
    return feature_length_dict

  @property
  def num_dataset_annotators(self):
    return self._num_dataset_annotators

  @property
  def _max_annotators_per_example(self):
    return self._num_dataset_annotators

  def _set_adversarial_pi_features(self, example, per_example_seed):
    # In ImageNet-PI the adversarial annotators have a confidence of 1/1000 and
    # a neglible number parameters.

    adversarial_labels = tf.random.stateless_categorical(
        tf.ones((self._num_adversarial_annotators_per_example,
                 self.info.num_classes)),
        num_samples=1,
        seed=per_example_seed)
    if self._one_hot:
      adversarial_labels = tf.one_hot(
          adversarial_labels, self.info.num_classes, dtype=tf.float32)
      # Remove extra dummy label dimension:
      # adversarial_labels: (num_annotators, 1, 1000) -> (num_annotators, 1000)
      adversarial_labels = tf.squeeze(adversarial_labels, axis=1)
    else:
      adversarial_labels = tf.cast(adversarial_labels, tf.float32)

    adversarial_ids = tf.reshape(
        tf.range(
            self.num_dataset_annotators,
            self.num_dataset_annotators +
            self._num_adversarial_annotators_per_example,
            dtype=tf.float32),
        [self._num_adversarial_annotators_per_example, 1])

    adversarial_confidences = tf.ones(
        (self._num_adversarial_annotators_per_example, 1),
        dtype=tf.float32) / self.info.num_classes

    # NOTE: The num_params inside 'annotator_features' are encoded using
    # (log(num_params) - mean(log(num_params))) / std(log(num_param)), so a
    # value of -3 represents a very small num_params.
    adversarial_num_params = tf.ones(
        (self._num_adversarial_annotators_per_example, 1),
        dtype=tf.float32) * ADVERSARIAL_NUM_PARAMS_NORMALIZED
    adversarial_reliabilities = tf.ones(
        (self._num_adversarial_annotators_per_example, 1),
        dtype=tf.float32) * (1.0 / self.info.num_classes)
    adversarial_features = tf.concat(
        [adversarial_reliabilities, adversarial_num_params], axis=1)

    return {
        'annotator_labels': adversarial_labels,
        'annotator_features': adversarial_features,
        'annotator_confidences': adversarial_confidences,
        'annotator_ids': adversarial_ids
    }

  def _process_pi_features_and_labels(self, example, unprocessed_example=None):
    """Loads 'annotator_ids', 'annotator_labels', 'annotator_features', 'annotator_confidences' and sets 'clean_labels'."""

    # Final parsed example should not include file_name.
    parsed_example = {
        'features': example['features'],
        'labels': example['labels']
    }

    file_name = example['file_name']
    # Store original label under new name.
    parsed_example['clean_labels'] = example['labels']

    annotators_meta_data = {}
    annotators_meta_data['annotator_labels'] = self._annotator_tables[
        'annotator_labels'][file_name]

    def _cast_and_reshape(pi_feature, feature_length):
      return tf.reshape(tf.cast(pi_feature, tf.float32), [-1, feature_length])

    if self._one_hot:
      annotators_meta_data['annotator_labels'] = tf.one_hot(
          annotators_meta_data['annotator_labels'],
          self.info.num_classes,
          dtype=tf.float32)
    else:
      annotators_meta_data['annotator_labels'] = _cast_and_reshape(
          annotators_meta_data['annotator_labels'], 1)

    annotators_meta_data['annotator_ids'] = _cast_and_reshape(
        tf.range(0, self.num_dataset_annotators), 1)
    annotators_meta_data['annotator_features'] = _cast_and_reshape(
        self._annotator_tables['annotator_features'],
        self._annotator_feature_length)
    annotators_meta_data['annotator_confidences'] = _cast_and_reshape(
        self._annotator_tables['annotator_confidences'][file_name], 1)

    parsed_example['pi_features'] = annotators_meta_data

    return parsed_example

  def _hash_fingerprint_int(self, fingerprint) -> int:
    return tf.strings.to_hash_bucket_fast(fingerprint, self.num_examples)

  def _get_annotator_features_tables(self):
    """Loads the annotations in memory and stores them in several hash tables indexed by filename."""

    filenames_label, filenames_confidence, annotator_features, annotator_labels, annotator_confidences = [], [], [], [], []
    label_fname = f'{self._annotations_path}/labels/{self._split_annotations_file}'
    feature_fname = f'{self._annotations_path}/annotator_features.csv'
    confidence_fname = f'{self._annotations_path}/confidences/{self._split_annotations_file}'

    with tf.io.gfile.GFile(feature_fname, 'r') as f:
      # Read from annotator_features.csv file.
      # Each row corresponds to one annotator and contains metadata about it.
      # Rows are formatted following the convention:
      # FEATURE_1, ..., FEATURE_N
      # where FEATURE_{} is given as a float.
      reader = csv.reader(f)

      annotator_features = list(csv.reader(f))
      annotator_feature_length = len(annotator_features[0])
      num_annotators = len(annotator_features)

    annotator_features = np.array(annotator_features)

    with tf.io.gfile.GFile(label_fname, 'r') as f:
      # Read from labels/{split}.csv file.
      # Each row corresponds to one example and it is formatted following:
      # FILENAME, RATER_LABEL_1, ..., RATER_LABEL_N
      # where FILENAME points to the example filename
      # and RATER_LABEL_{} is given as an integer
      reader = csv.reader(f)

      for line in reader:
        # Load only the annotators specified in annotator_idx.
        filename, label = line[0], np.array(line[1:])
        filenames_label.append(filename)
        annotator_labels.append(label)

    with tf.io.gfile.GFile(confidence_fname, 'r') as f:
      # Read from confidences/{split}.csv file.
      # Each row corresponds to one example and it is formatted following:
      # FILENAME, RATER_CONFIDENCE_1, ..., RATER_CONFIDENCE_N
      # where FILENAME points to the example filename
      # and RATER_CONFIDENCE_{} is given as a float
      reader = csv.reader(f)

      for line in reader:
        filename, confidence = line[0], np.array(line[1:])
        filenames_confidence.append(filename)
        annotator_confidences.append(confidence)

    filenames_label = tf.constant(filenames_label)
    filenames_confidence = tf.constant(filenames_confidence)
    annotator_features = tf.cast(
        tf.strings.to_number(annotator_features), tf.float32)
    annotator_labels = tf.cast(tf.strings.to_number(annotator_labels), tf.int32)
    annotator_confidences = tf.cast(
        tf.strings.to_number(annotator_confidences), tf.float32)

    annotator_tables = {}

    annotator_tables['annotator_labels'] = _store_in_hash_table(
        keys=filenames_label,
        values=annotator_labels,
        values_length=num_annotators,
        key_dtype=tf.string,
        value_dtype=tf.int32)

    annotator_tables['annotator_features'] = annotator_features

    annotator_tables['annotator_confidences'] = _store_in_hash_table(
        keys=filenames_confidence,
        values=annotator_confidences,
        values_length=num_annotators,
        key_dtype=tf.string,
        value_dtype=tf.float32)

    return annotator_tables, num_annotators, annotator_feature_length


# TODO(dusenberrymw): Create a helper function to load datasets for all
# corruption types and severity levels.
class ImageNetCorruptedDataset(_ImageNetDataset):
  """ImageNet-C dataset builder class."""

  def __init__(self, corruption_type: str, severity: int, **kwargs):
    """Create an ImageNet-C tf.data.Dataset builder.

    Args:
      corruption_type: Corruption name.
      severity: Corruption severity, an integer between 1 and 5.
      **kwargs: Additional keyword arguments.
    """
    if 'split' in kwargs:
      logging.warn("ImageNet-C only has a 'validation' split. Ignoring the"
                   'supplied split.')
      del kwargs['split']
    super().__init__(
        name=f'imagenet2012_corrupted/{corruption_type}_{severity}',
        split=tfds.Split.VALIDATION,
        preprocessing_type='resnet',
        **kwargs)
