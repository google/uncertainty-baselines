# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""Utilities for CIFAR-10 and CIFAR-100."""

import os
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def load_cifar100_c_input_fn(corruption_name,
                             corruption_intensity,
                             batch_size,
                             use_bfloat16,
                             path,
                             drop_remainder=True,
                             normalize=True,
                             standarize=True):
  """Loads CIFAR-100-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  filename = path + '{0}-{1}.tfrecords'.format(corruption_name,
                                               corruption_intensity)
  def preprocess(serialized_example):
    """Preprocess a serialized example for CIFAR100-C."""
    features = tf.io.parse_single_example(
        serialized_example,
        features={
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        })
    image = tf.io.decode_raw(features['image'], tf.uint8)
    image = tf.cast(tf.reshape(image, [32, 32, 3]), dtype)
    image = tf.image.convert_image_dtype(image, dtype)
    image = image / 255  # to convert into the [0, 1) range
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    elif standarize:
      # Normalize per-image using mean/stddev computed across pixels.
      image = tf.image.per_image_standardization(image)
    label = tf.cast(features['label'], dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


def load_cifar10_c_input_fn(corruption_name,
                            corruption_intensity,
                            batch_size,
                            use_bfloat16,
                            drop_remainder=True,
                            normalize=True):
  """Loads CIFAR-10-C dataset."""
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  corruption = corruption_name + '_' + str(corruption_intensity)
  def preprocess(image, label):
    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    dataset = tfds.load(name='cifar10_corrupted/{}'.format(corruption),
                        split=tfds.Split.TEST,
                        as_supervised=True)
    dataset = dataset.map(
        preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


# TODO(ghassen,trandustin): Push this metadata upstream to TFDS.
def load_corrupted_test_info(dataset):
  """Loads information for CIFAR-10-C."""
  if dataset == 'cifar10':
    corruption_types = [
        'gaussian_noise',
        'shot_noise',
        'impulse_noise',
        'defocus_blur',
        'frosted_glass_blur',
        'motion_blur',
        'zoom_blur',
        'snow',
        'frost',
        'fog',
        'brightness',
        'contrast',
        'elastic',
        'pixelate',
        'jpeg_compression',
    ]
  else:
    corruption_types = [
        'brightness',
        'contrast',
        'defocus_blur',
        'elastic_transform',
        'fog',
        'frost',
        'glass_blur',  # Called frosted_glass_blur in CIFAR-10.
        'gaussian_blur',
        'gaussian_noise',
        'impulse_noise',
        'jpeg_compression',
        'pixelate',
        'saturate',
        'shot_noise',
        'spatter',
        'speckle_noise',  # Does not exist for CIFAR-10.
        'zoom_blur',
    ]
  max_intensity = 5
  return corruption_types, max_intensity


def load_input_fn(split,
                  batch_size,
                  name,
                  use_bfloat16,
                  normalize=True,
                  drop_remainder=True,
                  repeat=False,
                  proportion=1.0,
                  data_dir=None):
  """Loads CIFAR dataset for training or testing.

  Args:
    split: tfds.Split.
    batch_size: The global batch size to use.
    name: A string indicates whether it is cifar10 or cifar100.
    use_bfloat16: data type, bfloat16 precision or float32.
    normalize: Whether to apply mean-std normalization on features.
    drop_remainder: bool.
    repeat: bool.
    proportion: float, the proportion of dataset to be used.
    data_dir: Directory where the dataset is stored to be loaded via tfds.load.
      Optional, useful for loading datasets stored on GCS.

  Returns:
    Input function which returns a locally-sharded dataset batch.
  """
  if use_bfloat16:
    dtype = tf.bfloat16
  else:
    dtype = tf.float32
  ds_info = tfds.builder(name).info
  image_shape = ds_info.features['image'].shape
  dataset_size = ds_info.splits['train'].num_examples

  def preprocess(image, label):
    """Image preprocessing function."""
    if split == tfds.Split.TRAIN:
      image = tf.image.resize_with_crop_or_pad(
          image, image_shape[0] + 4, image_shape[1] + 4)
      image = tf.image.random_crop(image, image_shape)
      image = tf.image.random_flip_left_right(image)

    image = tf.image.convert_image_dtype(image, dtype)
    if normalize:
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype)
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  def input_fn(ctx=None):
    """Returns a locally sharded (i.e., per-core) dataset batch."""
    if proportion == 1.0:
      dataset = tfds.load(
          name, split=split, data_dir=data_dir, as_supervised=True)
    else:
      new_name = '{}:3.*.*'.format(name)
      if split == tfds.Split.TRAIN:
        # use round instead of floor to resolve bug when e.g. using
        # proportion = 1 - 0.8 = 0.19999999
        new_split = 'train[:{}%]'.format(round(100 * proportion))
      elif split == tfds.Split.VALIDATION:
        new_split = 'train[-{}%:]'.format(round(100 * proportion))
      elif split == tfds.Split.TEST:
        new_split = 'test[:{}%]'.format(round(100 * proportion))
      else:
        raise ValueError('Provide valid split.')
      dataset = tfds.load(new_name, split=new_split, as_supervised=True)
    if split == tfds.Split.TRAIN or repeat:
      dataset = dataset.shuffle(buffer_size=dataset_size).repeat()

    dataset = dataset.map(preprocess,
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    if ctx and ctx.num_input_pipelines > 1:
      dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
    return dataset
  return input_fn


class LearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  """Learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               steps_per_epoch,
               initial_learning_rate,
               decay_ratio,
               decay_epochs,
               warmup_epochs):
    super(LearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = initial_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    learning_rate = self.initial_learning_rate
    if self.warmup_epochs >= 1:
      learning_rate *= lr_epoch / self.warmup_epochs
    decay_epochs = [self.warmup_epochs] + self.decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
      learning_rate = tf.where(
          lr_epoch >= start_epoch,
          self.initial_learning_rate * self.decay_ratio**index,
          learning_rate)
    return learning_rate

  def get_config(self):
    return {
        'steps_per_epoch': self.steps_per_epoch,
        'initial_learning_rate': self.initial_learning_rate,
    }


# TODO(baselines): Remove reliance on hard-coded metric names.
def aggregate_corrupt_metrics(metrics,
                              corruption_types,
                              max_intensity,
                              log_fine_metrics=False,
                              corrupt_diversity=None,
                              output_dir=None,
                              prefix='test'):
  """Aggregates metrics across intensities and corruption types.

  Args:
    metrics: Dictionary of tf.keras.metrics to be aggregated.
    corruption_types: List of corruption types.
    max_intensity: Int, of maximum intensity.
    log_fine_metrics: Bool, whether log fine metrics to main training script.
    corrupt_diversity: Dictionary of diversity metrics on corrupted datasets.
    output_dir: Str, the path to save the aggregated results.
    prefix: Str, the prefix before metrics such as 'test', 'lineareval'.

  Returns:
    Dictionary of aggregated results.

  """
  diversity_keys = ['disagreement', 'cosine_similarity', 'average_kl']
  results = {
      '{}/nll_mean_corrupted'.format(prefix): 0.,
      '{}/kl_mean_corrupted'.format(prefix): 0.,
      '{}/elbo_mean_corrupted'.format(prefix): 0.,
      '{}/accuracy_mean_corrupted'.format(prefix): 0.,
      '{}/ece_mean_corrupted'.format(prefix): 0.,
      '{}/member_acc_mean_corrupted'.format(prefix): 0.,
      '{}/member_ece_mean_corrupted'.format(prefix): 0.
  }
  fine_metrics_results = {}
  if corrupt_diversity is not None:
    for key in diversity_keys:
      results['corrupt_diversity/{}_mean_corrupted'.format(key)] = 0.

  for intensity in range(1, max_intensity + 1):
    nll = np.zeros(len(corruption_types))
    kl = np.zeros(len(corruption_types))
    elbo = np.zeros(len(corruption_types))
    acc = np.zeros(len(corruption_types))
    ece = np.zeros(len(corruption_types))
    member_acc = np.zeros(len(corruption_types))
    member_ece = np.zeros(len(corruption_types))
    disagreement = np.zeros(len(corruption_types))
    cosine_similarity = np.zeros(len(corruption_types))
    average_kl = np.zeros(len(corruption_types))

    for i in range(len(corruption_types)):
      dataset_name = '{0}_{1}'.format(corruption_types[i], intensity)
      nll[i] = metrics['{0}/nll_{1}'.format(prefix, dataset_name)].result()
      if '{0}/kl_{1}'.format(prefix, dataset_name) in metrics.keys():
        kl[i] = metrics['{0}/kl_{1}'.format(prefix, dataset_name)].result()
      else:
        kl[i] = 0.
      if '{0}/elbo_{1}'.format(prefix, dataset_name) in metrics.keys():
        elbo[i] = metrics['{0}/elbo_{1}'.format(prefix, dataset_name)].result()
      else:
        elbo[i] = 0.
      acc[i] = metrics['{0}/accuracy_{1}'.format(prefix, dataset_name)].result()
      ece[i] = metrics['{0}/ece_{1}'.format(prefix, dataset_name)].result()
      if '{0}/member_acc_mean_{1}'.format(prefix,
                                          dataset_name) in metrics.keys():
        member_acc[i] = metrics['{0}/member_acc_mean_{1}'.format(
            prefix, dataset_name)].result()
      else:
        member_acc[i] = 0.
      if '{0}/member_ece_mean_{1}'.format(prefix,
                                          dataset_name) in metrics.keys():
        member_ece[i] = metrics['{0}/member_ece_mean_{1}'.format(
            prefix, dataset_name)].result()
        member_ece[i] = 0.
      if corrupt_diversity is not None:
        disagreement[i] = (
            corrupt_diversity['corrupt_diversity/disagreement_{}'.format(
                dataset_name)].result())
        # Normalize the corrupt disagreement by its error rate.
        error = 1 - acc[i] + tf.keras.backend.epsilon()
        cosine_similarity[i] = (
            corrupt_diversity['corrupt_diversity/cosine_similarity_{}'.format(
                dataset_name)].result()) / error
        average_kl[i] = (
            corrupt_diversity['corrupt_diversity/average_kl_{}'.format(
                dataset_name)].result())
      if log_fine_metrics or output_dir is not None:
        fine_metrics_results['{0}/nll_{1}'.format(prefix,
                                                  dataset_name)] = nll[i]
        fine_metrics_results['{0}/kl_{1}'.format(prefix,
                                                 dataset_name)] = kl[i]
        fine_metrics_results['{0}/elbo_{1}'.format(prefix,
                                                   dataset_name)] = elbo[i]
        fine_metrics_results['{0}/accuracy_{1}'.format(prefix,
                                                       dataset_name)] = acc[i]
        fine_metrics_results['{0}/ece_{1}'.format(prefix,
                                                  dataset_name)] = ece[i]
        if corrupt_diversity is not None:
          fine_metrics_results['corrupt_diversity/disagreement_{}'.format(
              dataset_name)] = disagreement[i]
          fine_metrics_results['corrupt_diversity/cosine_similarity_{}'.format(
              dataset_name)] = cosine_similarity[i]
          fine_metrics_results['corrupt_diversity/average_kl_{}'.format(
              dataset_name)] = average_kl[i]
    avg_nll = np.mean(nll)
    avg_kl = np.mean(kl)
    avg_elbo = np.mean(elbo)
    avg_accuracy = np.mean(acc)
    avg_ece = np.mean(ece)
    avg_member_acc = np.mean(member_acc)
    avg_member_ece = np.mean(member_ece)
    results['{0}/nll_mean_{1}'.format(prefix, intensity)] = avg_nll
    results['{0}/kl_mean_{1}'.format(prefix, intensity)] = avg_kl
    results['{0}/elbo_mean_{1}'.format(prefix, intensity)] = avg_elbo
    results['{0}/accuracy_mean_{1}'.format(prefix, intensity)] = avg_accuracy
    results['{0}/ece_mean_{1}'.format(prefix, intensity)] = avg_ece
    results['{0}/nll_median_{1}'.format(prefix, intensity)] = np.median(nll)
    results['{0}/kl_median_{1}'.format(prefix, intensity)] = np.median(kl)
    results['{0}/elbo_median_{1}'.format(prefix, intensity)] = np.median(elbo)
    results['{0}/accuracy_median_{1}'.format(prefix,
                                             intensity)] = np.median(acc)
    results['{0}/ece_median_{1}'.format(prefix, intensity)] = np.median(ece)
    results['{0}/member_acc_mean_{1}'.format(prefix,
                                             intensity)] = avg_member_acc
    results['{0}/member_ece_mean_{1}'.format(prefix,
                                             intensity)] = avg_member_ece
    results['{}/nll_mean_corrupted'.format(prefix)] += avg_nll
    results['{}/kl_mean_corrupted'.format(prefix)] += avg_kl
    results['{}/elbo_mean_corrupted'.format(prefix)] += avg_elbo
    results['{}/accuracy_mean_corrupted'.format(prefix)] += avg_accuracy
    results['{}/ece_mean_corrupted'.format(prefix)] += avg_ece
    results['{}/member_acc_mean_corrupted'.format(prefix)] += avg_member_acc
    results['{}/member_ece_mean_corrupted'.format(prefix)] += avg_member_ece
    if corrupt_diversity is not None:
      avg_diversity_metrics = [np.mean(disagreement), np.mean(
          cosine_similarity), np.mean(average_kl)]
      for key, avg in zip(diversity_keys, avg_diversity_metrics):
        results['corrupt_diversity/{}_mean_{}'.format(
            key, intensity)] = avg
        results['corrupt_diversity/{}_mean_corrupted'.format(key)] += avg

  results['{}/nll_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/kl_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/elbo_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/accuracy_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/ece_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/member_acc_mean_corrupted'.format(prefix)] /= max_intensity
  results['{}/member_ece_mean_corrupted'.format(prefix)] /= max_intensity
  if corrupt_diversity is not None:
    for key in diversity_keys:
      results['corrupt_diversity/{}_mean_corrupted'.format(
          key)] /= max_intensity

  fine_metrics_results.update(results)
  if output_dir is not None:
    save_file_name = os.path.join(output_dir, 'corrupt_metrics.npz')
    with tf.io.gfile.GFile(save_file_name, 'w') as f:
      np.save(f, fine_metrics_results)

  if log_fine_metrics:
    return fine_metrics_results
  else:
    return results
