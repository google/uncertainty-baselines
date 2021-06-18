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

"""Utilities for CIFAR-10 and CIFAR-100."""

import os
from absl import flags
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# common flags
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', -1,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_string('data_dir', None,
                    'data_dir to be used for tfds dataset construction.'
                    'It is required when training with cloud TPUs')
flags.DEFINE_bool('download_data', False,
                  'Whether to download data locally when initializing a '
                  'dataset.')
flags.DEFINE_float('l2', 2e-4, 'L2 regularization coefficient.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('per_core_batch_size', 64,
                     'Batch size per TPU core/GPU. The number of new '
                     'datapoints gathered per batch is this number divided by '
                     'ensemble_size (we tile the batch by that # of times).')
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('train_epochs', 200, 'Number of training epochs.')
flags.DEFINE_float('train_proportion', default=1.0,
                   help='only use a proportion of training set.')
flags.register_validator('train_proportion',
                         lambda tp: tp > 0.0 and tp <= 1.0,
                         message='--train_proportion must be in (0, 1].')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')


def load_cifar100_c(corruption_name,
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

  dataset = tf.data.TFRecordDataset(filename, buffer_size=16 * 1000 * 1000)
  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset


def load_cifar10_c(corruption_name,
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
      # We use the convention of mean = np.mean(train_images, axis=(0,1,2))
      # and std = np.std(train_images, axis=(0,1,2)).
      mean = tf.constant([0.4914, 0.4822, 0.4465], dtype=dtype)
      std = tf.constant([0.2470, 0.2435, 0.2616], dtype=dtype)
      # Previously, std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
      # which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype).
      # However, we change convention to use the std over the entire training
      # set instead.
      image = (image - mean) / std
    label = tf.cast(label, dtype)
    return image, label

  dataset = tfds.load(name='cifar10_corrupted/{}'.format(corruption),
                      split=tfds.Split.TEST,
                      as_supervised=True)
  dataset = dataset.map(
      preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
  dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
  return dataset


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


# TODO(baselines): Remove reliance on hard-coded metric names.
def aggregate_corrupt_metrics(metrics,
                              corruption_types,
                              max_intensity=5,
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
      # TODO(dusenberrymw): rm.ECE returns a dictionary with a single item. Can
      # this be cleaned up?
      ece[i] = list(metrics['{0}/ece_{1}'.format(
          prefix, dataset_name)].result().values())[0]
      if '{0}/member_acc_mean_{1}'.format(prefix,
                                          dataset_name) in metrics.keys():
        member_acc[i] = metrics['{0}/member_acc_mean_{1}'.format(
            prefix, dataset_name)].result()
      else:
        member_acc[i] = 0.
      if '{0}/member_ece_mean_{1}'.format(prefix,
                                          dataset_name) in metrics.keys():
        member_ece[i] = list(metrics['{0}/member_ece_mean_{1}'.format(
            prefix, dataset_name)].result().values())[0]
      else:
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


def flatten_dictionary(x):
  """Flattens a dictionary where elements may itself be a dictionary.

  This function is helpful when using a collection of metrics, some of which
  include Robustness Metrics' metrics. Each metric in Robustness Metrics
  returns a dictionary with potentially multiple elements. This function
  flattens the dictionary of dictionaries.

  Args:
    x: Dictionary where keys are strings such as the name of each metric.

  Returns:
    Flattened dictionary.
  """
  outputs = {}
  for k, v in x.items():
    if isinstance(v, dict):
      if len(v.values()) == 1:
        # Collapse metric results like ECE's with dicts of len 1 into the
        # original key.
        outputs[k] = list(v.values())[0]
      else:
        # Flatten metric results like diversity's.
        for v_k, v_v in v.items():
          outputs[f'{k}/{v_k}'] = v_v
    else:
      outputs[k] = v
  return outputs
