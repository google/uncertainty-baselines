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

"""Uncertainty Utilities.

A set of model wrappers and evaluation utilities to determine the robustness
and quality of uncertainty estimates of the given model.
"""

import functools
import pdb
from typing import Dict

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.stats import bernoulli

tfd = tfp.distributions


"""
Model Wrappers:
Decompose uncertainty into aleatoric and epistemic portions using 
the mutual information between the parameters and the predicted output r.v. 
"""


def predict_and_decompose_uncertainty_tf(mc_samples: tf.Tensor):
  """Given a set of MC samples, decomposes uncertainty into
    aleatoric and epistemic parts.

  Args:
    mc_samples: `tf.Tensor`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [T, B] where T is the number of samples and B
      is the batch size.

  Returns:
    Dict: {
      mean: `tf.Tensor`, predictive mean, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
  """
  per_sample_entropies = tfd.Bernoulli(probs=mc_samples).entropy()
  expected_entropy = tf.reduce_mean(per_sample_entropies, axis=0)

  # Bernoulli output distribution
  predictive_dist = tfd.Bernoulli(probs=tf.reduce_mean(mc_samples, axis=0))

  predictive_entropy = predictive_dist.entropy()
  predictive_variance = predictive_dist.variance()
  predictive_mean = predictive_dist.mean()

  # tf.print(tf.shape(predictive_mean))
  # tf.print(tf.shape(predictive_entropy))
  # tf.print(tf.shape(predictive_variance))
  # tf.print(tf.shape(predictive_entropy - expected_entropy))
  # tf.print(tf.shape(expected_entropy))

  return {
    'prediction': predictive_mean,
    'predictive_entropy': predictive_entropy,
    'predictive_variance': predictive_variance,
    'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
    'aleatoric_uncertainty': expected_entropy
  }


def predict_and_decompose_uncertainty(mc_samples: np.ndarray):
  """Given a set of MC samples, decomposes uncertainty into
    aleatoric and epistemic parts.

  Args:
    mc_samples: `np.ndarray`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [T, B] where T is the number of samples and B
      is the batch size.

  Returns:
    Dict: {
      mean: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  num_samples = mc_samples.shape[0]
  try:
    per_sample_entropies = np.array([
      bernoulli(mc_samples[i, :]).entropy() for i in range(num_samples)])
  except KeyboardInterrupt:
    raise
  except:
    print(mc_samples)
    print(mc_samples.shape)
    print(num_samples)
    print(bernoulli(mc_samples[0, :]))

  expected_entropy = per_sample_entropies.mean(axis=0)

  # Bernoulli output distribution
  predictive_dist = bernoulli(mc_samples.mean(axis=0))

  predictive_entropy = predictive_dist.entropy()
  predictive_variance = predictive_dist.std() ** 2
  predictive_mean = predictive_dist.mean()

  return {
    'prediction': predictive_mean,
    'predictive_entropy': predictive_entropy,
    'predictive_variance': predictive_variance,
    'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
    'aleatoric_uncertainty': expected_entropy
  }


def variational_predict_and_decompose_uncertainty(
  x,
  model,
  training_setting,
  num_samples
):
  """Monte Carlo uncertainty estimator for a variational model,
    decomposes uncertainty into aleatoric and epistemic parts.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- e.g., MC Dropout, MFVI, Radial BNNs.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at test time
  # See note in docstring regarding `training` mode
  list_samples = []
  nb_trials = 0
  while len(list_samples) < num_samples:
    nb_trials += 1
    new_vals = model(x, training=training_setting)
    if np.isnan(new_vals).sum() == 0:
      list_samples.append(new_vals)
    if nb_trials == 20:
      raise ValueError(f"The model always returns nan!! {list_samples}")

  mc_samples = np.asarray([list_samples]).reshape(-1, b)

  return predict_and_decompose_uncertainty(mc_samples=mc_samples)


def variational_predict_and_decompose_uncertainty_tf(
  x,
  model,
  training_setting,
  num_samples
):
  """Monte Carlo uncertainty estimator for a variational model,
    decomposes uncertainty into aleatoric and epistemic parts.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- e.g., MC Dropout, MFVI, Radial BNNs.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, predictive mean, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b = tf.shape(x)[0]

  # Monte Carlo samples from different dropout mask at test time
  # See note in docstring regarding `training` mode
  if num_samples > 1:
    mc_samples = tf.convert_to_tensor(
      [model(x, training=training_setting) for _ in range(num_samples)])
    # mc_samples = tf.TensorArray(tf.float32, size=num_samples)
    # for i in tf.range(num_samples):
    #   probs = model(x, training=training_setting)
    #   mc_samples = mc_samples.write(i, probs)
    #
    # mc_samples = mc_samples.stack()
  else:
    mc_samples = model(x, training=training_setting)

  # Long-form Pythonic
  # mc_samples = []
  #
  # for _ in range(num_samples):
  #   print('retracing')
  #   mc_samples.append(model(x, training=training_setting))
  #
  # mc_samples = tf.convert_to_tensor(mc_samples)

  # TPU-friendly
  # mc_samples = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  # for _ in tf.range(num_samples):
  #   probs = model(x, training=training_setting)
  #   mc_samples = mc_samples.write(mc_samples.size(), probs)
  #
  # mc_samples = mc_samples.stack()
  mc_samples = tf.reshape(mc_samples, [-1, b])

  # tf.print('Mc samples shape')
  # tf.print(tf.shape(mc_samples))

  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)


def variational_ensemble_predict_and_decompose_uncertainty(
  x,
  models,
  training_setting,
  num_samples
):
  """Monte Carlo uncertainty estimator for ensembles of variational models.
    Decomposes uncertainty into aleatoric and epistemic parts.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- MC Dropout, MFVI, Radial BNNs.
  This estimator is for ensembles of the above methods.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at
  # test time from different models
  # See note in docstring regarding `training` mode
  # pylint: disable=g-complex-comprehension
  list_samples = []
  nb_trials = 0
  i = 0
  while len(list_samples) < num_samples * len(models):
    nb_trials += 1
    model_index = i % len(models)
    model = models[model_index]
    new_vals = model(x, training=training_setting)
    if np.isnan(new_vals).sum() == 0:
      list_samples.append(new_vals)
    if nb_trials == 20:
      raise ValueError(f"The model always returns nan!! {list_samples}")
    i += 1

  mc_samples = np.asarray(list_samples).reshape(-1, b)
  # pylint: enable=g-complex-comprehension

  # TODO: TPU friendly implementation
  # mc_samples = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  #
  # k = len(models)
  #
  # for model_index in tf.range(k):
  #   model = models[model_index]
  #   for _ in range()
  # for _ in tf.range(num_samples):
  #   probs = model(x, training=training_setting)
  #   mc_samples = mc_samples.write(mc_samples.size(), probs)
  #
  # mc_samples = mc_samples.stack()
  # mc_samples = tf.reshape(mc_samples, [-1, b])

  return predict_and_decompose_uncertainty(mc_samples=mc_samples)


def variational_ensemble_predict_and_decompose_uncertainty_tf(
  x,
  models,
  training_setting,
  num_samples
):
  """Monte Carlo uncertainty estimator for ensembles of variational models.
    Decomposes uncertainty into aleatoric and epistemic parts.

  Should work for all variational methods which sample from model posterior
  in each forward pass -- MC Dropout, MFVI, Radial BNNs.
  This estimator is for ensembles of the above methods.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, predictive mean, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b = x.shape[0]

  # Monte Carlo samples from different dropout mask at
  # test time from different models
  # See note in docstring regarding `training` mode
  # pylint: disable=g-complex-comprehension
  mc_samples = tf.convert_to_tensor([
      model(x, training=training_setting)
      for _ in range(num_samples)
      for model in models
  ])
  # pylint: enable=g-complex-comprehension

  mc_samples = tf.reshape(mc_samples, [-1, b])
  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)


def deterministic_predict_and_decompose_uncertainty(
  x,
  model,
  training_setting
):
  """
  Wrapper for simple sigmoid uncertainty estimator -- returns None for
    aleatoric and epistemic uncertainty, as we cannot obtain these.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: None
      aleatoric_uncertainty: None
    }
  """
  mean, predictive_entropy, predictive_variance = deterministic_predict(
    x=x, model=model, training_setting=training_setting)

  return {
    'prediction': mean,
    'predictive_entropy': predictive_entropy,
    'predictive_variance': predictive_variance,
    'epistemic_uncertainty': None,
    'aleatoric_uncertainty': None
  }


def deterministic_predict_and_decompose_uncertainty_tf(
  x,
  model,
  training_setting
):
  """
  Wrapper for simple sigmoid uncertainty estimator -- returns None for
    aleatoric and epistemic uncertainty, as we cannot obtain these.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: None
      aleatoric_uncertainty: None
    }
  """
  mean, predictive_entropy, predictive_variance = deterministic_predict_tf(
    x=x, model=model, training_setting=training_setting)

  return {
    'prediction': mean,
    'predictive_entropy': predictive_entropy,
    'predictive_variance': predictive_variance,
    'epistemic_uncertainty': None,
    'aleatoric_uncertainty': None
  }


def deep_ensemble_predict_and_decompose_uncertainty(
  x,
  models,
  training_setting
):
  """Monte Carlo uncertainty estimator for ensembles of deterministic models,
    decomposes uncertainty into aleatoric and epistemic parts.

  For example, this method should be used with Deep Ensembles (ensembles of
    deterministic neural networks, with different data/model seeds).

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different deterministic models
  mc_samples = np.asarray(
      [model(x, training=training_setting) for model in models]).reshape(-1, b)

  return predict_and_decompose_uncertainty(mc_samples=mc_samples)


def deep_ensemble_predict_and_decompose_uncertainty_tf(
  x,
  models,
  training_setting
):
  """Monte Carlo uncertainty estimator for ensembles of deterministic models,
    decomposes uncertainty into aleatoric and epistemic parts.

  For example, this method should be used with Deep Ensembles (ensembles of
    deterministic neural networks, with different data/model seeds).

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode.
      See note in docstring at top of file.

  Returns:
    Dict: {
      prediction: `tf.Tensor`, predictive mean, with shape [B].
      predictive_entropy: `tf.Tensor`, predictive entropy, with shape [B].
      predictive_variance: `tf.Tensor`, predictive variance, with shape [B].
      epistemic_uncertainty: `tf.Tensor`, mutual info, with shape [B].
      aleatoric_uncertainty: `tf.Tensor`, expected entropy, with shape [B].
    }
    """

  # Get shapes of data
  b = x.shape[0]

  # Monte Carlo samples from different deterministic models
  mc_samples = tf.convert_to_tensor(
      [model(x, training=training_setting) for model in models])
  mc_samples = tf.reshape(mc_samples, [-1, b])
  return predict_and_decompose_uncertainty_tf(mc_samples=mc_samples)


"""
Model Wrappers: obtain predictive entropy or predictive stddev 
along with the predictive mean.
"""


# TODO(@nband): low priority -- add TF versions of these wrappers

# def dropout_predict(x,
#                     model,
#                     training_setting,
#                     num_samples,
#                     uncertainty_type='entropy'):
#   """Monte Carlo Dropout uncertainty estimator.
#
#   Should work also with Variational Inference and Radial BNNs.
#
#   Args:
#     x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
#       where B the batch size and H, W the input images height and width
#       accordingly.
#     model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
#       input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
#       and also accepts boolean argument `training` for disabling e.g.,
#       BatchNorm, Dropout at test time.
#     training_setting: bool, if True, run model prediction in training mode. See
#       note in docstring at top of file.
#     num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
#       dropout) used for the calculation of predictive mean and uncertainty.
#     uncertainty_type: (optional) `str`, type of uncertainty; returns one of
#       {"entropy", "stddev"}.
#
#   Returns:
#     mean: `numpy.ndarray`, predictive mean, with shape [B].
#     uncertainty: `numpy.ndarray`, uncertainty in prediction,
#       with shape [B].
#   """
#   # Get shapes of data
#   b, _, _, _ = x.shape
#
#   # Monte Carlo samples from different dropout mask at test time
#   # See note in docstring regarding `training` mode
#   mc_samples = np.asarray([
#       model(x, training=training_setting) for _ in range(num_samples)
#   ]).reshape(-1, b)
#
#   # Bernoulli output distribution
#   dist = bernoulli(mc_samples.mean(axis=0))
#
#   return get_dist_mean_and_uncertainty(
#       dist=dist, uncertainty_type=uncertainty_type)
#
#
# def dropout_ensemble_predict(x,
#                              models,
#                              training_setting,
#                              num_samples,
#                              uncertainty_type='entropy'):
#   """Ensembles of Monte Carlo Dropout uncertainty estimator.
#
#   Args:
#     x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
#       where B the batch size and H, W the input images height and width
#       accordingly.
#     models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
#       each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
#       probability [0.0, 1.0], and also accepts boolean argument `training` for
#       disabling e.g., BatchNorm, Dropout at test time.
#     training_setting: bool, if True, run model prediction in training mode. See
#       note in docstring at top of file.
#     num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
#       dropout) used for each model in the ensemble, in the calculation of
#       predictive mean and uncertainty.
#     uncertainty_type: (optional) `str`, type of uncertainty; returns one of
#       {"entropy", "stddev"}.
#
#   Returns:
#     mean: `numpy.ndarray`, predictive mean, with shape [B].
#     uncertainty: `numpy.ndarray`, uncertainty in prediction,
#       with shape [B].
#   """
#   # Get shapes of data
#   b, _, _, _ = x.shape
#
#   # Monte Carlo samples from different dropout mask at
#   # test time from different models
#   # See note in docstring regarding `training` mode
#   # pylint: disable=g-complex-comprehension
#   mc_samples = np.asarray([
#       model(x, training=training_setting)
#       for _ in range(num_samples)
#       for model in models
#   ]).reshape(-1, b)
#   # pylint: enable=g-complex-comprehension
#
#   # Bernoulli output distribution
#   dist = bernoulli(mc_samples.mean(axis=0))
#
#   return get_dist_mean_and_uncertainty(
#       dist=dist, uncertainty_type=uncertainty_type)
#
#
def deterministic_predict(x,
                          model,
                          training_setting):
  """Simple sigmoid uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Single forward pass from the deterministic model
  p = model(x, training=training_setting)

  # Bernoulli output distribution
  dist = bernoulli(p)

  return get_dist_mean_and_uncertainty(dist=dist)


def deterministic_predict_tf(x, model, training_setting):
  """Simple sigmoid uncertainty estimator.

  Args:
    x: `tf.Tensor`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
  Returns:
    mean: `tf.Tensor`, predictive mean, with shape [B].
    uncertainty: `tf.Tensor`, uncertainty in prediction,
      with shape [B].
  """
  # Single forward pass from the deterministic model
  p = model(x, training=training_setting)

  # Bernoulli output distribution
  dist = tfd.Bernoulli(probs=p)

  return get_dist_mean_and_uncertainty_tf(dist=dist)
#
#
# def deep_ensemble_predict(x,
#                           models,
#                           training_setting,
#                           uncertainty_type='entropy'):
#   """Deep Ensembles uncertainty estimator.
#
#   Args:
#     x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
#       where B the batch size and H, W the input images height and width
#       accordingly.
#     models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
#       each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
#       probability [0.0, 1.0], and also accepts boolean argument `training` for
#       disabling e.g., BatchNorm, Dropout at test time.
#     training_setting: bool, if True, run model prediction in training mode. See
#       note in docstring at top of file.
#     uncertainty_type: (optional) `str`, type of uncertainty; returns one of
#       {"entropy", "stddev"}.
#
#   Returns:
#     mean: `numpy.ndarray`, predictive mean, with shape [B].
#     uncertainty: `numpy.ndarray`, uncertainty in prediction,
#       with shape [B].
#   """
#   # Get shapes of data
#   b, _, _, _ = x.shape
#
#   # Monte Carlo samples from different deterministic models
#   mc_samples = np.asarray(
#       [model(x, training=training_setting) for model in models]).reshape(-1, b)
#
#   # Bernoulli output distribution
#   dist = bernoulli(mc_samples.mean(axis=0))
#
#   return get_dist_mean_and_uncertainty(
#       dist=dist, uncertainty_type=uncertainty_type)
#
#


def binary_entropy_jax(array):
  import jax
  return jax.scipy.special.entr(array) + jax.scipy.special.entr(1 - array)


def fsvi_predict_and_decompose_uncertainty(
        x,
        model,
        rng_key,
        training_setting,
        num_samples,
        params,
        state,
):
  """
  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
    where B the batch size and H, W the input images height and width
    accordingly.
    model: a probabilistic model (e.g., `tensorflow.keras.model`) which accepts
      input with shape [B, H, W, 3] and outputs sigmoid probability [0.0, 1.0],
      and also accepts boolean argument `training` for disabling e.g.,
      BatchNorm, Dropout at test time, as well as rng_key as random key for the
      forward passes.
    rng_key: `jax.numpy.ndarray`, jax random key for the forward passes.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: int, the number of MC samples for each member of ensenble
    params: parameters of haiku model
    state: state of haiku model

  Returns:
    Dict: {
      mean: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  # mc_samples has shape [T, B]
  preds_y_samples, _, _ = model.predict_y_multisample_jitted(
    params=params,
    state=state,
    inputs=x,
    rng_key=rng_key,
    n_samples=num_samples,
    is_training=training_setting,
  )
  mc_samples = preds_y_samples[:, :, 1]

  return predict_and_decompose_uncertainty_jax(mc_samples=mc_samples)


def fsvi_ensemble_predict_and_decompose_uncertainty(
      x,
      model,
      rng_key,
      training_setting,
      num_samples,
      params,
      state,
):
  import jax.numpy as jnp
  """
  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
    where B the batch size and H, W the input images height and width
    accordingly.
    model: a list of FSVI Model objects
    rng_key: `jax.numpy.ndarray`, jax random key for the forward passes.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: int, the number of MC samples for each member of ensenble
    params: parameters of haiku model
    state: state of haiku model

  Returns:
    Dict: {
      mean: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  # mc_samples has shape [T, B]
  list_mc_samples = []
  for i, m in enumerate(model):
    preds_y_samples, _, _ = m.predict_y_multisample_jitted(
      params=params[i],
      state=state[i],
      inputs=x,
      rng_key=rng_key,
      n_samples=num_samples,
      is_training=training_setting,
    )
    list_mc_samples.append(preds_y_samples[:, :, 1])

  mc_samples = jnp.concatenate(list_mc_samples)

  return predict_and_decompose_uncertainty_jax(mc_samples=mc_samples)



def predict_and_decompose_uncertainty_jax(mc_samples):
  """Given a set of MC samples, decomposes uncertainty into
    aleatoric and epistemic parts.

  Args:
    mc_samples: `np.ndarray`, Monte Carlo samples from a sigmoid predictive
      distribution, shape [T, B] where T is the number of samples and B
      is the batch size.

  Returns:
    Dict: {
      mean: `numpy.ndarray`, predictive mean, with shape [B].
      predictive_entropy: `numpy.ndarray`, predictive entropy, with shape [B].
      predictive_variance: `numpy.ndarray`, predictive variance, with shape [B].
      epistemic_uncertainty: `numpy.ndarray`, mutual info, with shape [B].
      aleatoric_uncertainty: `numpy.ndarray`, expected entropy, with shape [B].
    }
  """
  expected_entropy = binary_entropy_jax(mc_samples).mean(axis=0)

  # Bernoulli output distribution
  predictive_mean = mc_samples.mean(axis=0)
  predictive_entropy = binary_entropy_jax(predictive_mean)
  predictive_variance = predictive_mean * (1 - predictive_mean)

  return {
    'prediction': predictive_mean,
    'predictive_entropy': predictive_entropy,
    'predictive_variance': predictive_variance,
    'epistemic_uncertainty': predictive_entropy - expected_entropy,  # MI
    'aleatoric_uncertainty': expected_entropy
  }


def get_dist_mean_and_uncertainty(dist: bernoulli):
  """Compute the mean and uncertainty.

  From a scipy.stats.bernoulli predictive distribution, compute the predictive
  mean and uncertainty (entropy and variance).

  Args:
    dist: `scipy.stats.bernoulli`, predictive distribution constructed from
      probabilistic model samples for some input batch.

  Returns:
    mean: `np.ndarray`, predictive mean, with shape [B].
    predictive_entropy: `numpy.ndarray`, total uncertainty by predictive
      entropy, with shape [B].
    predictive_variance: `numpy.ndarray`, total uncertainty by predictive
      variance, with shape [B].
  """
  mean = dist.mean()
  predictive_entropy = dist.entropy()
  predictive_variance = dist.std() ** 2
  return mean, predictive_entropy, predictive_variance


def get_dist_mean_and_uncertainty_tf(dist: tfd.Bernoulli):
  """Compute the mean and uncertainty.

  From a tensorflow_probability.distributions.Bernoulli predictive distribution,
   compute the predictive mean and uncertainty (entropy and variance).

  Args:
    dist: `tfd.Bernoulli`, predictive distribution constructed from
      probabilistic model samples for some input batch.

  Returns:
    mean: `tf.Tensor`, predictive mean, with shape [B].
    predictive_entropy: `tf.Tensor`, total uncertainty by predictive
      entropy, with shape [B].
    predictive_variance: `tf.Tensor`, total uncertainty by predictive
      variance, with shape [B].
  """
  mean = dist.mean()
  predictive_entropy = dist.entropy()
  predictive_variance = dist.variance()
  return mean, predictive_entropy, predictive_variance


# RETINOPATHY_MODEL_TO_TOTAL_UNCERTAINTY_ESTIMATOR = {
#     'deterministic': deterministic_predict,
#     'dropout': dropout_predict,
#     'dropoutensemble': dropout_ensemble_predict,
#     'ensemble': deep_ensemble_predict,
#     'radial': dropout_predict,
#     'variational_inference': dropout_predict
# }

# Format:
# (model_type, use_ensemble): predict_and_decompose_uncertainty_fn
RETINOPATHY_MODEL_TO_DECOMPOSED_UNCERTAINTY_ESTIMATOR = {
  ('deterministic', False): deterministic_predict_and_decompose_uncertainty,
  ('deterministic', True): deep_ensemble_predict_and_decompose_uncertainty,
  ('dropout', False): variational_predict_and_decompose_uncertainty,
  ('dropout', True): variational_ensemble_predict_and_decompose_uncertainty,
  ('radial', False): variational_predict_and_decompose_uncertainty,
  ('radial', True): variational_ensemble_predict_and_decompose_uncertainty,
  ('variational_inference', False): (
    variational_predict_and_decompose_uncertainty),
  ('variational_inference', True): (
    variational_ensemble_predict_and_decompose_uncertainty),
  ('rank1', False): variational_predict_and_decompose_uncertainty,
  ('rank1', True): variational_ensemble_predict_and_decompose_uncertainty,
  ('swag', False): None,  # SWAG requires sampling outside the dataset loop
  ('swag', True): None,
  ('fsvi', False): fsvi_predict_and_decompose_uncertainty,
  ('fsvi', True): fsvi_ensemble_predict_and_decompose_uncertainty,
}

# (model_type, use_ensemble): predict_and_decompose_uncertainty_fn
# Need these for use with TensorFlow TPU and GPU strategies
RETINOPATHY_MODEL_TO_TF_DECOMPOSED_UNCERTAINTY_ESTIMATOR = {
  ('deterministic', False): deterministic_predict_and_decompose_uncertainty_tf,
  ('deterministic', True): deep_ensemble_predict_and_decompose_uncertainty_tf,
  ('dropout', False): variational_predict_and_decompose_uncertainty_tf,
  ('dropout', True): variational_ensemble_predict_and_decompose_uncertainty_tf,
  ('radial', False): variational_predict_and_decompose_uncertainty_tf,
  ('radial', True): variational_ensemble_predict_and_decompose_uncertainty_tf,
  ('variational_inference', False): (
    variational_predict_and_decompose_uncertainty_tf),
  ('variational_inference', True): (
    variational_ensemble_predict_and_decompose_uncertainty_tf),

  # Rank 1 BNNs also have default functionality for mixture posteriors
  ('rank1', False): (
    variational_predict_and_decompose_uncertainty_tf),
  ('rank1', True): (
    variational_ensemble_predict_and_decompose_uncertainty_tf),
  ('swag', False): None,  # SWAG requires sampling outside the dataset loop
  ('swag', True): None,
  # ('fsvi', False): variational_predict_and_decompose_uncertainty_tf,
}


"""
Prediction and Loss computation.
"""


def wrap_retinopathy_estimator(
    estimator, use_mixed_precision, return_logits=False, numpy_outputs=True
):
  """Models used in the Diabetic Retinopathy baseline output logits by default.

  Apply conversion if necessary based on mixed precision setting, and apply
  a sigmoid to obtain sigmoid probability [0.0, 1.0] for the model.

  Args:
    estimator: a `tensorflow.keras.model` probabilistic model, that accepts
      input with shape [B, H, W, 3] and outputs logits
    use_mixed_precision: bool, whether to use mixed precision.
    return_logits: bool, optionally return logits.
    numpy_outputs: bool, convert outputs to numpy.

  Returns:
     wrapped estimator, outputting sigmoid probabilities.
  """
  def estimator_wrapper(inputs, training, estimator):
    logits = estimator(inputs, training=training)
    if use_mixed_precision:
      logits = tf.cast(logits, tf.float32)
    probs = tf.squeeze(tf.nn.sigmoid(logits))

    if numpy_outputs and return_logits:
      return probs.numpy(), logits.numpy()
    elif return_logits:
      return probs, logits
    elif numpy_outputs:
      return probs.numpy()
    else:
      return probs

  return functools.partial(estimator_wrapper, estimator=estimator)


def negative_log_likelihood_metric(labels, probs):
  """Wrapper computing NLL for the Diabetic Retinopathy classification task.

  Args:
    labels: the ground truth labels, with shape `[batch_size, d0, .., dN]`.
    probs: the predicted values, with shape `[batch_size, d0, .., dN]`.

  Returns:
    Binary NLL.
  """
  return tf.reduce_mean(
      tf.keras.losses.binary_crossentropy(
          y_true=tf.expand_dims(labels, axis=-1),
          y_pred=tf.expand_dims(probs, axis=-1),
          from_logits=False))


def get_uncertainty_estimator(model_type, use_ensemble, use_tf):
  # if model_type == 'rank1' and use_ensemble:
  #   raise NotImplementedError  # Special code for using ensemble
  #
  if model_type == 'swag':
    raise NotImplementedError  # Special eval loop
  try:
    if use_tf:
      uncertainty_estimator_fn = (
        RETINOPATHY_MODEL_TO_TF_DECOMPOSED_UNCERTAINTY_ESTIMATOR[
          (model_type, use_ensemble)])
    else:
      uncertainty_estimator_fn = (
        RETINOPATHY_MODEL_TO_DECOMPOSED_UNCERTAINTY_ESTIMATOR[
          (model_type, use_ensemble)])
  except KeyError:
    raise NotImplementedError(
        'Unsupported model type. Try implementing a wrapper to retrieve '
        'aleatoric, epistemic, and total uncertainty in X.py.')

  return uncertainty_estimator_fn
