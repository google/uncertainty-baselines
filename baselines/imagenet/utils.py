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

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
from uncertainty_baselines.datasets import resnet_preprocessing
tfd = tfp.distributions


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
    return {'features': image, 'labels': label}

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
        member_ece[i] = list(metrics['test/member_ece_mean_{}'.format(
            dataset_name)].result().values())[0]
      else:
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
