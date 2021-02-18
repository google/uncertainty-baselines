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

import edward2 as ed
import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from uncertainty_baselines.datasets import resnet_preprocessing
tfd = tfp.distributions

# ImageNet statistics. Used to normalize the input to Efficientnet.
IMAGENET_MEAN = np.array([[[0.485, 0.456, 0.406]]], np.float32) * 255.
IMAGENET_STDDEV = np.array([[[0.229, 0.224, 0.225]]], np.float32) * 255.


# TODO(trandustin): Refactor similar to CIFAR code.
class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Resnet learning rate schedule."""

  def __init__(self, steps_per_epoch, initial_learning_rate, num_epochs,
               schedule):
    super(LearningRateSchedule, self).__init__()
    self.num_epochs = num_epochs
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.schedule = schedule

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    warmup_lr_multiplier, warmup_end_epoch = self.schedule[0]
    # Scale learning rate schedule by total epochs at vanilla settings.
    warmup_end_epoch = (warmup_end_epoch * self.num_epochs) // 90
    learning_rate = (
        self.initial_learning_rate * warmup_lr_multiplier * lr_epoch /
        warmup_end_epoch)
    for mult, start_epoch in self.schedule:
      start_epoch = (start_epoch * self.num_epochs) // 90
      learning_rate = tf.where(lr_epoch >= start_epoch,
                               self.initial_learning_rate * mult, learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
        'num_epochs': self.num_epochs,
        'schedule': self.schedule,
    }


class WarmupDecaySchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A wrapper for LearningRateSchedule that includes warmup steps."""

  def __init__(self, lr_schedule, warmup_steps):
    """Add warmup decay to a learning rate schedule.

    Args:
      lr_schedule: base learning rate scheduler
      warmup_steps: number of warmup steps

    """
    super(WarmupDecaySchedule, self).__init__()
    self._lr_schedule = lr_schedule
    self._warmup_steps = warmup_steps

  def __call__(self, step):
    lr = self._lr_schedule(step)
    if self._warmup_steps:
      initial_learning_rate = tf.convert_to_tensor(
          self._lr_schedule.initial_learning_rate, name='initial_learning_rate')
      dtype = initial_learning_rate.dtype
      global_step_recomp = tf.cast(step, dtype)
      warmup_steps = tf.cast(self._warmup_steps, dtype)
      warmup_lr = initial_learning_rate * global_step_recomp / warmup_steps
      lr = tf.cond(global_step_recomp < warmup_steps,
                   lambda: warmup_lr,
                   lambda: lr)
    return lr

  def get_config(self):
    config = self._lr_schedule.get_config()
    config.update({
        'warmup_steps': self._warmup_steps,
    })
    return config


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
        dataset = dataset.map(
            functools.partial(adaptive_mixup, batch_size, self.mixup_params),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      else:
        dataset = dataset.map(
            functools.partial(mixup, batch_size, self.mixup_alpha,
                              self.same_mix_weight_per_batch,
                              self.use_truncated_beta,
                              self.use_random_shuffling),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

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


def mixup(batch_size, alpha, same_mix_weight_per_batch, use_truncated_beta,
          use_random_shuffling, images, labels):
  """Applies Mixup regularization to a batch of images and labels.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    alpha: Float that controls the strength of Mixup regularization.
    same_mix_weight_per_batch: whether to use the same mix coef over the batch.
    use_truncated_beta: whether to sample from Beta_[0,1](alpha, alpha) or from
       the truncated distribution Beta_[1/2, 1](alpha, alpha).
    use_random_shuffling: Whether to pair images by random shuffling
      (default is a deterministic pairing by reversing the batch).
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  if same_mix_weight_per_batch:
    mix_weight = ed.Beta(alpha, alpha, sample_shape=[1, 1])
    mix_weight = tf.tile(mix_weight, [batch_size, 1])
  else:
    mix_weight = ed.Beta(alpha, alpha, sample_shape=[batch_size, 1])

  if use_truncated_beta:
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)

  images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  images_mix_weight = tf.cast(images_mix_weight, images.dtype)

  if not use_random_shuffling:
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
    mixup_index = tf.reverse(tf.range(batch_size), axis=[0])
  else:
    mixup_index = tf.random.shuffle(tf.range(batch_size))

  images_mix = (
      images * images_mix_weight + tf.gather(images, mixup_index) *
      (1. - images_mix_weight))
  mix_weight = tf.cast(mix_weight, labels.dtype)
  labels_mix = labels * mix_weight + tf.gather(labels,
                                               mixup_index) * (1. - mix_weight)
  return images_mix, labels_mix


def adaptive_mixup(batch_size, mixup_params, images, labels):
  """Applies CAMixup regularization to a batch of images and labels.

  Arguments:
    batch_size: The input batch size for images and labels.
    mixup_params: `Dict` to store the hparams of mixup.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  ensemble_size = mixup_params.get('ensemble_size', 1)
  mixup_coeff = mixup_params['mixup_coeff']
  scalar_labels = tf.argmax(labels, axis=1)
  alpha = tf.gather(mixup_coeff, scalar_labels, axis=-1)

  # Need to filter out elements in alpha which equal to 0.
  greater_zero_indicator = tf.cast(alpha > 0, alpha.dtype)
  less_one_indicator = tf.cast(alpha < 1, alpha.dtype)
  valid_alpha_indicator = tf.cast(
      greater_zero_indicator * less_one_indicator, tf.bool)

  dummy_alpha = 0.1 * tf.ones_like(alpha)
  sampled_alpha = tf.where(valid_alpha_indicator, alpha, dummy_alpha)
  mix_weight = tfd.Beta(sampled_alpha, sampled_alpha).sample()
  mix_weight = tf.where(valid_alpha_indicator, mix_weight, alpha)
  mix_weight = tf.reshape(mix_weight, [ensemble_size * batch_size, 1])
  mix_weight = tf.clip_by_value(mix_weight, 0, 1)
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  images_mix_weight = tf.reshape(mix_weight,
                                 [ensemble_size * batch_size, 1, 1, 1])

  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  images = tf.tile(images, [ensemble_size, 1, 1, 1])
  labels = tf.tile(labels, [ensemble_size, 1])
  images_mix_weight = tf.cast(images_mix_weight, images.dtype)
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
  labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
  return images_mix, labels_mix


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


# TODO(ywenxu): Check out `tf.keras.layers.experimental.SyncBatchNormalization.
# SyncBatchNorm on TPU. Orginal authored by hyhieu.
class SyncBatchNorm(tf.keras.layers.Layer):
  """BatchNorm that averages over ALL replicas. Only works for `NHWC` inputs."""

  def __init__(self, axis=3, momentum=0.99, epsilon=0.001,
               trainable=True, name='batch_norm', **kwargs):
    super(SyncBatchNorm, self).__init__(
        trainable=trainable, name=name, **kwargs)
    self.axis = axis
    self.momentum = momentum
    self.epsilon = epsilon

  def build(self, input_shape):
    """Build function."""
    dim = input_shape[-1]
    shape = [dim]

    self.gamma = self.add_weight(
        name='gamma',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        trainable=True)

    self.beta = self.add_weight(
        name='beta',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        trainable=True)

    self.moving_mean = self.add_weight(
        name='moving_mean',
        shape=shape,
        dtype=self.dtype,
        initializer='zeros',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

    self.moving_variance = self.add_weight(
        name='moving_variance',
        shape=shape,
        dtype=self.dtype,
        initializer='ones',
        synchronization=tf.VariableSynchronization.ON_READ,
        trainable=False,
        aggregation=tf.VariableAggregation.MEAN)

  def _get_mean_and_variance(self, x):
    """Cross-replica mean and variance."""
    replica_context = tf.distribute.get_replica_context()
    num_replicas_in_sync = replica_context.num_replicas_in_sync
    if num_replicas_in_sync <= 8:
      group_assignment = None
      num_replicas_per_group = tf.cast(num_replicas_in_sync, tf.float32)
    else:
      num_replicas_per_group = max(8, num_replicas_in_sync // 8)
      group_assignment = np.arange(num_replicas_in_sync, dtype=np.int32)
      group_assignment = group_assignment.reshape([-1, num_replicas_per_group])
      group_assignment = group_assignment.tolist()
      num_replicas_per_group = tf.cast(num_replicas_per_group, tf.float32)

    mean = tf.reduce_mean(x, axis=[0, 1, 2])
    mean = tf.cast(mean, tf.float32)
    mean = tf.tpu.cross_replica_sum(mean, group_assignment)
    mean = mean / num_replicas_per_group

    # Var[x] = E[x^2] - E[x]^2
    mean_sq = tf.reduce_mean(tf.square(x), axis=[0, 1, 2])
    mean_sq = tf.cast(mean_sq, tf.float32)
    mean_sq = tf.tpu.cross_replica_sum(mean_sq, group_assignment)
    mean_sq = mean_sq / num_replicas_per_group
    variance = mean_sq - tf.square(mean)

    def _assign(moving, normal):
      decay = tf.cast(1. - self.momentum, tf.float32)
      diff = tf.cast(moving, tf.float32) - tf.cast(normal, tf.float32)
      return moving.assign_sub(decay * diff)

    self.add_update(_assign(self.moving_mean, mean))
    self.add_update(_assign(self.moving_variance, variance))

    # TODO(ywenxu): Assuming bfloat16. Fix for non bfloat16 case.
    mean = tf.cast(mean, tf.bfloat16)
    variance = tf.cast(variance, tf.bfloat16)

    return mean, variance

  def call(self, inputs, training):
    """Call function."""
    if training:
      mean, variance = self._get_mean_and_variance(inputs)
    else:
      mean, variance = self.moving_mean, self.moving_variance
    x = tf.nn.batch_normalization(
        inputs,
        mean=mean,
        variance=variance,
        offset=self.beta,
        scale=self.gamma,
        variance_epsilon=tf.cast(self.epsilon, variance.dtype),
    )
    return x


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


class MovingAverage(tf.keras.optimizers.Optimizer):
  """Optimizer that computes a moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate)
  opt = MovingAverage(opt)

  opt.shadow_copy(model)
  ```

  At test time, swap the shadow variables to evaluate on the averaged weights:
  ```python
  opt.swap_weights()
  # Test eval the model here
  opt.swap_weights()
  ```
  """

  def __init__(self,
               optimizer,
               average_decay=0.99,
               start_step=0,
               dynamic_decay=True,
               name='moving_average',
               **kwargs):
    """Construct a new MovingAverage optimizer.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      average_decay: float. Decay to use to maintain the moving averages
        of trained variables.
      start_step: int. What step to start the moving average.
      dynamic_decay: bool. Whether to change the decay based on the number
        of optimizer updates. Decay will start at 0.1 and gradually increase
        up to `average_decay` after each optimizer update. This behavior is
        similar to `tf.train.ExponentialMovingAverage` in TF 1.x.
      name: Optional name for the operations created when applying
        gradients. Defaults to "moving_average".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}.
    """
    super(MovingAverage, self).__init__(name, **kwargs)
    self._optimizer = optimizer
    self._average_decay = average_decay
    self._start_step = tf.constant(start_step, tf.float32)
    self._dynamic_decay = dynamic_decay

  def shadow_copy(self, model):
    """Creates shadow variables for the given model weights."""
    for var in model.weights:
      self.add_slot(var, 'average', initializer='zeros')
    self._average_weights = [
        self.get_slot(var, 'average') for var in model.weights
    ]
    self._model_weights = model.weights

  @property
  def has_shadow_copy(self):
    """Whether this optimizer has created shadow variables."""
    return self._model_weights is not None

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access

  def apply_gradients(self, grads_and_vars, name=None):
    result = self._optimizer.apply_gradients(grads_and_vars, name)
    self.update_average(self._optimizer.iterations)
    return result

  @tf.function
  def update_average(self, step):
    step = tf.cast(step, tf.float32)
    if step < self._start_step:
      decay = tf.constant(0., tf.float32)
    elif self._dynamic_decay:
      decay = step - self._start_step
      decay = tf.minimum(self._average_decay, (1. + decay) / (10. + decay))
    else:
      decay = self._average_decay

    def _apply_moving(v_moving, v_normal):
      diff = v_moving - v_normal
      v_moving.assign_sub(tf.cast(1. - decay, v_moving.dtype) * diff)
      return v_moving

    def _update(strategy, v_moving_and_v_normal):
      for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(_update, args=(zip(self._average_weights,
                                             self._model_weights),))

  def swap_weights(self, strategy):
    """Swap the average and moving weights.

    This is a convenience method to allow one to evaluate the averaged weights
    at test time. Loads the weights stored in `self._average` into the model,
    keeping a copy of the original model weights. Swapping twice will return
    the original weights.

    Args:
      strategy: tf.distribute.Strategy to be used.
    """
    strategy.run(self._swap_weights, args=())

  def _swap_weights(self):
    def fn_0(a, b):
      a.assign_add(b)
      return a
    def fn_1(b, a):
      b.assign(a - b)
      return b
    def fn_2(a, b):
      a.assign_sub(b)
      return a

    def swap(strategy, a_and_b):
      """Swap `a` and `b` and mirror to all devices."""
      for a, b in a_and_b:
        strategy.extended.update(a, fn_0, args=(b,))  # a = a + b
        strategy.extended.update(b, fn_1, args=(a,))  # b = a - b
        strategy.extended.update(a, fn_2, args=(b,))  # a = a - b

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(
        swap, args=(zip(self._average_weights, self._model_weights),))

  def assign_average_vars(self, var_list):
    """Assign variables in var_list with their respective averages.

    Args:
      var_list: List of model variables to be assigned to their average.
    Returns:
      assign_op: The op corresponding to the assignment operation of
        variables to their average.
    """
    assign_op = tf.group([
        var.assign(self.get_slot(var, 'average')) for var in var_list
        if var.trainable
    ])
    return assign_op

  def _create_hypers(self):
    self._optimizer._create_hypers()  # pylint: disable=protected-access

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  @property
  def weights(self):
    return self._optimizer.weights

  # pylint: disable=protected-access
  @property
  def lr(self):
    return self._optimizer._get_hyper('learning_rate')

  @lr.setter
  def lr(self, lr):
    self._optimizer._set_hyper('learning_rate', lr)

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')

  @learning_rate.setter
  def learning_rate(self, learning_rate):  # pylint: disable=redefined-outer-name
    self._optimizer._set_hyper('learning_rate', learning_rate)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse(grad, var, indices)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse_duplicate_indices(
        grad, var, indices)
  # pylint: enable=protected-access

  def get_config(self):
    config = {
        'optimizer': tf.keras.optimizers.serialize(self._optimizer),
        'average_decay': self._average_decay,
        'start_step': self._start_step,
        'dynamic_decay': self._dynamic_decay,
    }
    base_config = super(MovingAverage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    return cls(optimizer, **config)
