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

"""Utilities for UCI datasets."""

import collections
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf


class DataSpec(collections.namedtuple(
    'UCIDataSpec', 'path,desc,label,excluded')):

  __slots__ = []


# TODO(trandustin): Avoid hard-coding directory string so it's user-specified.
UCI_BASE_DIR = '/tmp/uci_datasets'
DATA_SPECS = {
    'boston_housing': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'boston_housing.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='MEDV',
        excluded=[]),
    'concrete_strength': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'concrete_strength.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='concrete_compressive_strength',
        excluded=[]),
    'energy_efficiency': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'energy_efficiency.csv'),
        desc=('This study looked into assessing the heating load and cooling '
              'load requirements of buildings (that is, energy efficiency) as '
              'a function of building parameters. **Heating load only**.'),
        label='Y1',
        excluded=['Y2']),
    'naval_propulsion': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'naval_propulsion.csv'),
        desc=('Data have been generated from a sophisticated simulator of a '
              'Gas Turbines (GT), mounted on a Frigate characterized by a '
              'Combined Diesel eLectric And Gas (CODLAG) propulsion plant '
              'type. **GT Turbine decay state coefficient only**'),
        label='GT Turbine decay state coefficient',
        excluded=['GT Compressor decay state coefficient']),
    'kin8nm': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'kin8nm.csv'),
        desc=('This is data set is concerned with the forward kinematics of '
              'an 8 link robot arm. Among the existing variants of this data '
              'set we have used the variant 8nm, which is known to be highly '
              'non-linear and medium noisy.'),
        label='y',
        excluded=[]),
    'power_plant': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'power_plant.csv'),
        desc=('The Boston housing data was collected in 1978 and each of the '
              '506 entries represent aggregated data about 14 features for '
              'homes from various suburbs in Boston, Massachusetts.'),
        label='PE',
        excluded=[]),
    'protein_structure': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'protein_structure.csv'),
        desc=('This is a data set of Physicochemical Properties of Protein '
              'Tertiary Structure. The data set is taken from CASP 5-9. There '
              'are 45730 decoys and size varying from 0 to 21 armstrong.'),
        label='RMSD',
        excluded=[]),
    'wine': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'wine.csv'),
        desc=('The dataset is related to red variant of the Portuguese '
              '"Vinho Verde" wine. **NB contains red wine examples only**'),
        label='quality',
        excluded=[]),
    'yacht_hydrodynamics': DataSpec(
        path=os.path.join(UCI_BASE_DIR, 'yacht_hydrodynamics.csv'),
        desc=('Delft data set, used to predict the hydodynamic performance of '
              'sailing yachts from dimensions and velocity.'),
        label='Residuary resistance per unit weight of displacement',
        excluded=[])
}


def get_uci_data(name):
  """Returns an array of features and a vector of labels for dataset `name`."""
  spec = DATA_SPECS.get(name)
  if spec is None:
    raise ValueError('Unknown dataset: {}. Available datasets:\n{}'.format(
        name, '\n'.join(DATA_SPECS.keys())))
  with tf.io.gfile.GFile(spec.path) as f:
    df = pd.read_csv(f)
  labels = df.pop(spec.label).as_matrix().astype(np.float32)
  for ex in spec.excluded:
    _ = df.pop(ex)
  features = df.as_matrix().astype(np.float32)
  return features, labels


def load(name):
  """Loads dataset as numpy array."""
  x, y = get_uci_data(name)
  if len(y.shape) == 1:
    y = y[:, None]
  train_test_split = 0.8
  random_permutation = np.random.permutation(x.shape[0])
  n_train = int(x.shape[0] * train_test_split)
  train_ind = random_permutation[:n_train]
  test_ind = random_permutation[n_train:]
  x_train, y_train = x[train_ind, :], y[train_ind, :]
  x_test, y_test = x[test_ind, :], y[test_ind, :]

  x_mean, x_std = np.mean(x_train, axis=0), np.std(x_train, axis=0)
  y_mean = np.mean(y_train, axis=0)
  epsilon = tf.keras.backend.epsilon()
  x_train = (x_train - x_mean) / (x_std + epsilon)
  x_test = (x_test - x_mean) / (x_std + epsilon)
  y_train, y_test = y_train - y_mean, y_test - y_mean
  return x_train, y_train, x_test, y_test


def ensemble_metrics(x,
                     y,
                     model,
                     log_likelihood_fn,
                     n_samples=1,
                     weight_files=None):
  """Evaluate metrics of an ensemble.

  Args:
    x: numpy array of inputs
    y: numpy array of labels
    model: tf.keras.Model.
    log_likelihood_fn: function which takes tuple of x, y and returns batched
      tuple output of the log prob and mean error.
    n_samples: number of Monte Carlo samples to draw per ensemble member (each
      weight file).
    weight_files: to draw samples from multiple weight sets, specify a list of
      weight files to load. These files must have been generated through
      keras's model.save_weights(...).

  Returns:
    metrics_dict: dictionary containing the metrics
  """
  if weight_files is None:
    ensemble_logprobs = [log_likelihood_fn([x, y])[0] for _ in range(n_samples)]
    ensemble_error = [log_likelihood_fn([x, y])[1] for _ in range(n_samples)]
  else:
    ensemble_logprobs = []
    ensemble_error = []
    for filename in weight_files:
      model.load_weights(filename)
      ensemble_logprobs.extend([
          log_likelihood_fn([x, y])[0] for _ in range(n_samples)])
      ensemble_error.extend([
          log_likelihood_fn([x, y])[1] for _ in range(n_samples)])

  results = {}
  ensemble_logprobs = np.array(ensemble_logprobs)
  results['log_likelihood'] = np.mean(ensemble_logprobs)
  results['mse'] = np.mean(np.square(ensemble_error))
  probabilistic_log_likelihood = np.mean(
      scipy.special.logsumexp(
          np.sum(ensemble_logprobs, axis=2)
          if len(ensemble_logprobs.shape) > 2 else ensemble_logprobs,
          b=1. / ensemble_logprobs.shape[0],
          axis=0),
      axis=0)
  results['probabilistic_log_likelihood'] = probabilistic_log_likelihood
  ensemble_error = np.stack([np.array(l) for l in ensemble_error])
  results['probabilistic_mse'] = np.mean(
      np.square(np.mean(ensemble_error, axis=0)))
  return results
