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

"""ImageNet dataset builder.

We have an option to use a percent of the training dataset as a validation set,
and treat the original validation set as the test set. This is similar to what
is also done in the NeurIPS uncertainty benchmark paper
https://arxiv.org/abs/1906.02530 (which used (100 / 1024)% as a validation set).
"""
from typing import Any, Dict, Optional, Union

import numpy as np
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets import inception_preprocessing
from uncertainty_baselines.datasets import resnet_preprocessing


# ImageNet statistics. Used to normalize the input to Efficientnet. Note that
# these do NOT have `* 255.` after them.
IMAGENET_MEAN = np.array([[[0.485, 0.456, 0.406]]], dtype=np.float32)
IMAGENET_STDDEV = np.array([[[0.229, 0.224, 0.225]]], dtype=np.float32)


def _tuple_dict_fn_converter(fn, *args):

  def dict_fn(batch_dict):
    images, labels = fn(*args, batch_dict['features'], batch_dict['labels'])
    return {'features': images, 'labels': labels}

  return dict_fn


class ImageNetDataset(base.BaseDataset):
  """ImageNet dataset builder class."""

  def __init__(
      self,
      split: str,
      seed: Optional[Union[int, tf.Tensor]] = None,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = 16384,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      is_training: Optional[bool] = None,
      preprocessing_type: str = 'resnet',
      use_bfloat16: bool = False,
      normalize_input: bool = False,
      image_size: int = 224,
      resnet_preprocessing_resize_method: Optional[str] = None,
      ensemble_size: int = 1,
      one_hot: bool = False,
      mixup_params: Optional[Dict[str, Any]] = None,
      run_mixup: bool = False):
    """Create an ImageNet tf.data.Dataset builder.

    Args:
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
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
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
      **unused_kwargs: Ignored.
    """
    name = 'imagenet2012'
    dataset_builder = tfds.builder(name, try_gcs=try_gcs)
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
        fingerprint_key='file_name',
        download_data=download_data,
        decoders=decoders)
    self._preprocessing_type = preprocessing_type
    self._use_bfloat16 = use_bfloat16
    self._normalize_input = normalize_input
    self._image_size = image_size
    self._resnet_preprocessing_resize_method = resnet_preprocessing_resize_method
    self._run_mixup = run_mixup

    self.ensemble_size = ensemble_size
    self._one_hot = one_hot
    if mixup_params is None:
      mixup_params = {}
    self._mixup_params = mixup_params

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
      if 'file_name' in example:
        parsed_example['file_name'] = example['file_name']
      return parsed_example

    return _example_parser

  def _create_process_batch_fn(
      self,
      batch_size: int) -> Optional[base.PreProcessFn]:
    mixup_alpha = self._mixup_params.get('mixup_alpha', 0.0)
    if (self._is_training or self._run_mixup) and mixup_alpha > 0.0:
      same_mix_weight_per_batch = self._mixup_params.get(
          'same_mix_weight_per_batch', False)
      use_truncated_beta = self._mixup_params.get('use_truncated_beta', True)
      use_random_shuffling = self._mixup_params.get(
          'use_random_shuffling', False)
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
        return _tuple_dict_fn_converter(
            augmix.adaptive_mixup, batch_size, self._mixup_params)
      else:
        aug_params = {
            'mixup_alpha': mixup_alpha,
            'same_mix_weight_per_batch': same_mix_weight_per_batch,
            'use_truncated_beta': use_truncated_beta,
            'use_random_shuffling': use_random_shuffling,
        }
        return _tuple_dict_fn_converter(augmix.mixup, batch_size, aug_params)

    return None
