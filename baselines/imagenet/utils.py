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

"""Utilities for ImageNet."""

import functools
import os

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from uncertainty_baselines.datasets import augmix
from uncertainty_baselines.datasets import resnet_preprocessing
tfd = tfp.distributions

# ImageNet statistics. Used to normalize the input to Efficientnet.
IMAGENET_MEAN = np.array([[[0.485, 0.456, 0.406]]], np.float32) * 255.
IMAGENET_STDDEV = np.array([[[0.229, 0.224, 0.225]]], np.float32) * 255.


class ImageNetInput(object):
  """Generates ImageNet for training or evaluation.

  The training data is assumed to be in TFRecord format with keys as specified
  in the dataset_parser below, sharded across 1024 files, named sequentially:
      train-00000-of-01024
      train-00001-of-01024
      ...
      train-01023-of-01024

  The validation data is in the same format but sharded in 128 files.

  The format of the data required is created by the script at:
      https://github.com/tensorflow/tpu/blob/master/tools/datasets/imagenet_to_gcs.py

  Attributes:
    data_dir: `str` for the directory of the training and validation data.
    use_bfloat16: If True, use bfloat16 precision; else use float32.
    image_size: `int` for image size (both width and height).
    normalize_input: `bool` for normalizing the input. Enable in Efficientnet.
    one_hot: `bool` for using one-hot label. Enable in Efficientnet.
    resize_method: If None, use bicubic in default.
    mixup_alpha: `float` to control the strength of Mixup regularization, set to
        0.0 to disable.
    mixup_params: `Dict` to store the hparams of mixup.
    ensemble_size: `int` for number of ensemble members. Used in Mixup.
    validation: `Bool`, if True, return a training set without augmentation.
    same_mix_weight_per_batch: `Bool`, whether to use the same mixing weight
        across the batch (default False). Used in Mixup.
    use_truncated_beta: whether to sample from Beta_[0,1](alpha, alpha) or from
       the truncated distribution Beta_[1/2, 1](alpha, alpha). Used in Mixup.
    use_random_shuffling: `Bool`, whether to use random shuffling to pair points
        within the batch while applying mixup (default False). Used in Mixup.
  """

  def __init__(self,
               data_dir=None,
               image_size=224,
               normalize_input=False,
               one_hot=False,
               use_bfloat16=False,
               resize_method=None,
               mixup_params=None,
               ensemble_size=1):
    self.use_bfloat16 = use_bfloat16
    self.data_dir = data_dir
    self.image_size = image_size
    self.normalize_input = normalize_input
    self.one_hot = one_hot
    self.resize_method = resize_method
    self.ensemble_size = ensemble_size
    self.mixup_params = mixup_params

    default_mixup_alpha = 0.0
    default_same_mix_weight_per_batch = False
    default_use_truncated_beta = True
    default_use_random_shuffling = False
    if mixup_params is not None:
      self.mixup_alpha = mixup_params.get('mixup_alpha', default_mixup_alpha)
      self.same_mix_weight_per_batch = mixup_params.get(
          'same_mix_weight_per_batch', default_same_mix_weight_per_batch)
      self.use_truncated_beta = mixup_params.get(
          'use_truncated_beta', default_use_truncated_beta)
      self.use_random_shuffling = mixup_params.get(
          'use_random_shuffling', default_use_random_shuffling)
    else:
      self.mixup_alpha = default_mixup_alpha
      self.same_mix_weight_per_batch = default_same_mix_weight_per_batch
      self.use_truncated_beta = default_use_truncated_beta
      self.use_random_shuffling = default_use_random_shuffling

  def dataset_parser(self, value, split):
    """Parse an ImageNet record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.parse_single_example(value, keys_to_features)
    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])

    image = resnet_preprocessing.preprocess_image(
        image_bytes=image_bytes,
        is_training=split == tfds.Split.TRAIN,
        use_bfloat16=self.use_bfloat16,
        image_size=self.image_size,
        resize_method=self.resize_method)

    # Subtract one so that labels are in [0, 1000), and cast to float32 for
    # Keras model.
    if self.one_hot:
      # TODO(ywenxu): The number of classes is hard coded for now.
      label = tf.cast(parsed['image/class/label'], tf.int32) - 1
      label = tf.one_hot(label, 1000, dtype=tf.float32)
    else:
      label = tf.cast(parsed['image/class/label'], dtype=tf.int32) - 1
      label = tf.cast(label, tf.float32)

    if self.normalize_input:
      mean = np.reshape(IMAGENET_MEAN, [1, 1, 3])
      stddev = np.reshape(IMAGENET_STDDEV, [1, 1, 3])
      image = (tf.cast(image, tf.float32) - mean) / stddev
      if self.use_bfloat16:
        image = tf.cast(image, tf.bfloat16)
    return image, label

  def as_dataset(self,
                 split,
                 batch_size,
                 drop_remainder=True):
    """Builds a `tf.data.Dataset` object.

    Args:
      split: tfds.Split.
      batch_size: The global batch size.
      drop_remainder: `bool` for dropping the remainder when batching.

    Returns:
      tf.data.Dataset.
    """
    # Shuffle the filenames to ensure better randomization.
    # If validation split is specified, we use a partition of the training data.
    is_training = split in (tfds.Split.TRAIN, tfds.Split.VALIDATION)
    file_pattern = os.path.join(
        self.data_dir, 'train-*' if is_training else 'validation-*')
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    # Evaluation dataset can also be repeat as long as steps_per_eval is set.
    dataset = dataset.repeat()

    def fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024     # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Read the data from disk in parallel
    dataset = dataset.interleave(
        fetch_dataset, cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    if is_training:
      dataset = dataset.shuffle(1024)

    preprocess = functools.partial(self.dataset_parser, split=split)
    dataset = dataset.map(preprocess, tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder)

    if is_training and self.mixup_alpha > 0.0:
      if self.mixup_params['adaptive_mixup']:
        if 'mixup_coeff' not in self.mixup_params:
          # Hard target in the first epoch!
          self.mixup_params['mixup_coeff'] = tf.ones(
              [self.ensemble_size, self.mixup_params['num_classes']])
        adaptive_mixup_fn = functools.partial(
            augmix.adaptive_mixup, batch_size, self.mixup_params),
        dataset = dataset.map(
            adaptive_mixup_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
      else:
        aug_params = {
            'mixup_alpha': self.mixup_alpha,
            'same_mix_weight_per_batch': self.same_mix_weight_per_batch,
            'use_truncated_beta': self.use_truncated_beta,
            'use_random_shuffling': self.use_random_shuffling,
        }
        mixup_fn = functools.partial(augmix.mixup, batch_size, aug_params)
        dataset = dataset.map(
            mixup_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    # Optimize dataset performance.
    options = tf.data.Options()
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset


def load_corrupted_test_dataset(corruption_name,
                                corruption_intensity,
                                batch_size,
                                drop_remainder=True,
                                use_bfloat16=False):
  """Loads an ImageNet-C dataset."""
  corruption = corruption_name + '_' + str(corruption_intensity)

  dataset = tfds.load(
      name='imagenet2012_corrupted/{}'.format(corruption),
      split=tfds.Split.VALIDATION,
      decoders={
          'image': tfds.decode.SkipDecoding(),
      },
      with_info=False,
      as_supervised=True)

  def preprocess(image, label):
    image = tf.reshape(image, shape=[])
    image = resnet_preprocessing.preprocess_for_eval(image, use_bfloat16)
    label = tf.cast(label, dtype=tf.float32)
    return image, label

  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


# TODO(ghassen,trandustin): Push this metadata upstream to TFDS.
def load_corrupted_test_info():
  """Loads information for ImageNet-C."""
  corruption_types = [
      'gaussian_noise',
      'shot_noise',
      'impulse_noise',
      'defocus_blur',
      'glass_blur',
      'motion_blur',
      'zoom_blur',
      'snow',
      'frost',
      'fog',
      'brightness',
      'contrast',
      'elastic_transform',
      'pixelate',
      'jpeg_compression',
  ]
  max_intensity = 5
  return corruption_types, max_intensity


# TODO(baselines): Remove reliance on hard-coded metric names.
def aggregate_corrupt_metrics(metrics,
                              corruption_types,
                              max_intensity,
                              alexnet_errors_path=None,
                              fine_metrics=False):
  """Aggregates metrics across intensities and corruption types."""
  results = {
      'test/nll_mean_corrupted': 0.,
      'test/kl_mean_corrupted': 0.,
      'test/elbo_mean_corrupted': 0.,
      'test/accuracy_mean_corrupted': 0.,
      'test/ece_mean_corrupted': 0.,
      'test/member_acc_mean_corrupted': 0.,
      'test/member_ece_mean_corrupted': 0.
  }
  for intensity in range(1, max_intensity + 1):
    nll = np.zeros(len(corruption_types))
    kl = np.zeros(len(corruption_types))
    elbo = np.zeros(len(corruption_types))
    acc = np.zeros(len(corruption_types))
    ece = np.zeros(len(corruption_types))
    member_acc = np.zeros(len(corruption_types))
    member_ece = np.zeros(len(corruption_types))
    for i in range(len(corruption_types)):
      dataset_name = '{0}_{1}'.format(corruption_types[i], intensity)
      nll[i] = metrics['test/nll_{}'.format(dataset_name)].result()
      if 'test/kl_{}'.format(dataset_name) in metrics.keys():
        kl[i] = metrics['test/kl_{}'.format(dataset_name)].result()
      else:
        kl[i] = 0.
      if 'test/elbo_{}'.format(dataset_name) in metrics.keys():
        elbo[i] = metrics['test/elbo_{}'.format(dataset_name)].result()
      else:
        elbo[i] = 0.
      acc[i] = metrics['test/accuracy_{}'.format(dataset_name)].result()
      ece[i] = metrics['test/ece_{}'.format(dataset_name)].result()
      if 'test/member_acc_mean_{}'.format(dataset_name) in metrics.keys():
        member_acc[i] = metrics['test/member_acc_mean_{}'.format(
            dataset_name)].result()
      else:
        member_acc[i] = 0.
      if 'test/member_ece_mean_{}'.format(dataset_name) in metrics.keys():
        member_ece[i] = metrics['test/member_ece_mean_{}'.format(
            dataset_name)].result()
        member_ece[i] = 0.
      if fine_metrics:
        results['test/nll_{}'.format(dataset_name)] = nll[i]
        results['test/kl_{}'.format(dataset_name)] = kl[i]
        results['test/elbo_{}'.format(dataset_name)] = elbo[i]
        results['test/accuracy_{}'.format(dataset_name)] = acc[i]
        results['test/ece_{}'.format(dataset_name)] = ece[i]
    avg_nll = np.mean(nll)
    avg_kl = np.mean(kl)
    avg_elbo = np.mean(elbo)
    avg_accuracy = np.mean(acc)
    avg_ece = np.mean(ece)
    avg_member_acc = np.mean(member_acc)
    avg_member_ece = np.mean(member_ece)
    results['test/nll_mean_{}'.format(intensity)] = avg_nll
    results['test/kl_mean_{}'.format(intensity)] = avg_kl
    results['test/elbo_mean_{}'.format(intensity)] = avg_elbo
    results['test/accuracy_mean_{}'.format(intensity)] = avg_accuracy
    results['test/ece_mean_{}'.format(intensity)] = avg_ece
    results['test/nll_median_{}'.format(intensity)] = np.median(nll)
    results['test/kl_median_{}'.format(intensity)] = np.median(kl)
    results['test/elbo_median_{}'.format(intensity)] = np.median(elbo)
    results['test/accuracy_median_{}'.format(intensity)] = np.median(acc)
    results['test/ece_median_{}'.format(intensity)] = np.median(ece)
    results['test/nll_mean_corrupted'] += avg_nll
    results['test/kl_mean_corrupted'] += avg_kl
    results['test/elbo_mean_corrupted'] += avg_elbo
    results['test/accuracy_mean_corrupted'] += avg_accuracy
    results['test/ece_mean_corrupted'] += avg_ece
    results['test/member_acc_mean_{}'.format(intensity)] = avg_member_acc
    results['test/member_ece_mean_{}'.format(intensity)] = avg_member_ece
    results['test/member_acc_mean_corrupted'] += avg_member_acc
    results['test/member_ece_mean_corrupted'] += avg_member_ece

  results['test/nll_mean_corrupted'] /= max_intensity
  results['test/kl_mean_corrupted'] /= max_intensity
  results['test/elbo_mean_corrupted'] /= max_intensity
  results['test/accuracy_mean_corrupted'] /= max_intensity
  results['test/ece_mean_corrupted'] /= max_intensity
  results['test/member_acc_mean_corrupted'] /= max_intensity
  results['test/member_ece_mean_corrupted'] /= max_intensity

  if alexnet_errors_path:
    with tf.io.gfile.GFile(alexnet_errors_path, 'r') as f:
      df = pd.read_csv(f, index_col='intensity').transpose()
    alexnet_errors = df.to_dict()
    corrupt_error = {}
    for corruption in corruption_types:
      alexnet_normalization = alexnet_errors[corruption]['average']
      errors = np.zeros(max_intensity)
      for index in range(max_intensity):
        dataset_name = '{0}_{1}'.format(corruption, index + 1)
        errors[index] = 1. - metrics['test/accuracy_{}'.format(
            dataset_name)].result()
      average_error = np.mean(errors)
      corrupt_error[corruption] = average_error / alexnet_normalization
      results['test/corruption_error_{}'.format(
          corruption)] = 100 * corrupt_error[corruption]
    results['test/mCE'] = 100 * np.mean(list(corrupt_error.values()))
  return results


def drop_connect(inputs, is_training, survival_prob):
  """Drop the entire conv with given survival probability."""
  # "Deep Networks with Stochastic Depth", https://arxiv.org/pdf/1603.09382.pdf
  if not is_training:
    return inputs

  # Compute tensor.
  batch_size = tf.shape(inputs)[0]
  random_tensor = survival_prob
  random_tensor += tf.random.uniform([batch_size, 1, 1, 1], dtype=inputs.dtype)
  binary_tensor = tf.floor(random_tensor)
  # Unlike conventional way that multiply survival_prob at test time, here we
  # divide survival_prob at training time, such that no addition compute is
  # needed at test time.
  output = tf.math.divide(inputs, survival_prob) * binary_tensor
  return output


def conv_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for convolutional kernels.

  The main difference with tf.variance_scaling_initializer is that
  tf.variance_scaling_initializer uses a truncated normal with an uncorrected
  standard deviation, whereas here we use a normal distribution. Similarly,
  tf.initializers.variance_scaling uses a truncated normal with
  a corrected standard deviation.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  kernel_height, kernel_width, _, out_filters = shape
  fan_out = int(kernel_height * kernel_width * out_filters)
  return tf.random.normal(
      shape, mean=0.0, stddev=np.sqrt(2.0 / fan_out), dtype=dtype)


def dense_kernel_initializer(shape, dtype=None, partition_info=None):
  """Initialization for dense kernels.

  This initialization is equal to
    tf.variance_scaling_initializer(scale=1.0/3.0, mode='fan_out',
                                    distribution='uniform').
  It is written out explicitly here for clarity.

  Args:
    shape: shape of variable
    dtype: dtype of variable
    partition_info: unused

  Returns:
    an initialization for the variable
  """
  del partition_info
  init_range = 1.0 / np.sqrt(shape[1])
  return tf.random.uniform(shape, -init_range, init_range, dtype=dtype)
