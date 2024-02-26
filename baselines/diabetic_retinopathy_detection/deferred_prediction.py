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

"""Deffered prediction utilities.

A set of model wrappers and evaluation utilities to perform Deferred
Prediction. In this task, the model uses pointwise uncertainty estimates (e.g.,
the entropy of the predictive distribution) to decide which test points to
`defer`.

In the context of Diabetic Retinopathy, this might mean that a diagnostic
model will refuse to classify points with the highest uncertainty, and instead
will refer them to a human specialist for manual diagnosis.
Throughout the methods, the `fractions` array specifies the proportion of the
test set that must be `retained`, i.e., that the model must classify and cannot
refer to a specialist. Over this portion, we compute metrics. A model that
has well-calibrated uncertainty should perform particularly well compared
to other models as the proportion of the test set that must be retained
decreases.

Note the `training_setting` flag in the model wrappers. For our model
implementations, we have explicitly configured dropout to sample new
masks at test time regardless of the model setting (e.g., see apply_dropout
in resnet50_dropout.py). Therefore, we set training=False to run BatchNorm
in inference mode, even when evaluating e.g., Monte Carlo Dropout.

This will often not be the case for those using MC Dropout with standard
models (e.g. PyTorch ResNet50) -- users should alter their model or pass
training_setting = True in this case.
"""

import collections
import datetime
import functools
import inspect
import os
from typing import Any, Callable, Dict, Optional, Sequence, Text, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import bernoulli
import tensorflow as tf
import tensorflow_datasets as tfds


def dropout_predict(x,
                    model,
                    training_setting,
                    num_samples,
                    uncertainty_type='entropy'):
  """Monte Carlo Dropout uncertainty estimator.

  Should work also with Variational Inference and Radial BNNs.

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
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for the calculation of predictive mean and uncertainty.
    uncertainty_type: (optional) `str`, type of uncertainty; returns one of
      {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at test time
  # See note in docstring regarding `training` mode
  mc_samples = np.asarray([
      model(x, training=training_setting) for _ in range(num_samples)
  ]).reshape(-1, b)

  # Bernoulli output distribution
  dist = bernoulli(mc_samples.mean(axis=0))

  return get_dist_mean_and_uncertainty(
      dist=dist, uncertainty_type=uncertainty_type)


def dropout_ensemble_predict(x,
                             models,
                             training_setting,
                             num_samples,
                             uncertainty_type='entropy'):
  """Ensembles of Monte Carlo Dropout uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    num_samples: `int`, number of Monte Carlo samples (i.e. forward passes from
      dropout) used for each model in the ensemble, in the calculation of
      predictive mean and uncertainty.
    uncertainty_type: (optional) `str`, type of uncertainty; returns one of
      {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different dropout mask at
  # test time from different models
  # See note in docstring regarding `training` mode
  # pylint: disable=g-complex-comprehension
  mc_samples = np.asarray([
      model(x, training=training_setting)
      for _ in range(num_samples)
      for model in models
  ]).reshape(-1, b)
  # pylint: enable=g-complex-comprehension

  # Bernoulli output distribution
  dist = bernoulli(mc_samples.mean(axis=0))

  return get_dist_mean_and_uncertainty(
      dist=dist, uncertainty_type=uncertainty_type)


def deterministic_predict(x,
                          model,
                          training_setting,
                          uncertainty_type='entropy'):
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
    uncertainty_type: (optional) `str`, type of uncertainty; returns one of
      {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Single forward pass from the deterministic model
  p = model(x, training=training_setting)

  # Bernoulli output distribution
  dist = bernoulli(p)

  return get_dist_mean_and_uncertainty(
      dist=dist, uncertainty_type=uncertainty_type)


def deep_ensemble_predict(x,
                          models,
                          training_setting,
                          uncertainty_type='entropy'):
  """Deep Ensembles uncertainty estimator.

  Args:
    x: `numpy.ndarray`, datapoints from input space, with shape [B, H, W, 3],
      where B the batch size and H, W the input images height and width
      accordingly.
    models: `iterable` of probabilistic models (e.g., `tensorflow.keras.model`),
      each of which accepts input with shape [B, H, W, 3] and outputs sigmoid
      probability [0.0, 1.0], and also accepts boolean argument `training` for
      disabling e.g., BatchNorm, Dropout at test time.
    training_setting: bool, if True, run model prediction in training mode. See
      note in docstring at top of file.
    uncertainty_type: (optional) `str`, type of uncertainty; returns one of
      {"entropy", "stddev"}.

  Returns:
    mean: `numpy.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Get shapes of data
  b, _, _, _ = x.shape

  # Monte Carlo samples from different deterministic models
  mc_samples = np.asarray(
      [model(x, training=training_setting) for model in models]).reshape(-1, b)

  # Bernoulli output distribution
  dist = bernoulli(mc_samples.mean(axis=0))

  return get_dist_mean_and_uncertainty(
      dist=dist, uncertainty_type=uncertainty_type)


def get_dist_mean_and_uncertainty(dist: bernoulli, uncertainty_type: str):
  """Compute the mean and uncertainty.

  From a scipy.stats.bernoulli predictive distribution, compute the predictive
  mean and uncertainty (entropy or stddev).

  Args:
    dist: `scipy.stats.bernoulli`, predictive distribution constructed from
      probabilistic model samples for some input batch.
    uncertainty_type: (optional) `str`, type of uncertainty; returns one of
      {"entropy", "stddev"}.

  Returns:
    mean: `np.ndarray`, predictive mean, with shape [B].
    uncertainty: `numpy.ndarray`, uncertainty in prediction,
      with shape [B].
  """
  # Predictive mean calculation
  mean = dist.mean()

  # Use predictive entropy for uncertainty
  if uncertainty_type == 'entropy':
    uncertainty = dist.entropy()
  # Use predictive standard deviation for uncertainty
  elif uncertainty_type == 'stddev':
    uncertainty = dist.std()
  else:
    raise ValueError(
        f'Unrecognized type of uncertainty '
        f"{uncertainty_type} provided, use one of 'entropy', 'stddev'.")

  return mean, uncertainty


RETINOPATHY_MODEL_TO_UNCERTAINTY_ESTIMATOR = {
    'deterministic': deterministic_predict,
    'dropout': dropout_predict,
    'dropoutensemble': dropout_ensemble_predict,
    'ensemble': deep_ensemble_predict,
    'radial': dropout_predict,
    'variational_inference': dropout_predict
}


def evaluate(
    estimator: Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]],
    dataset: tf.data.Dataset,
    metrics: Dict[Text, Callable[[np.ndarray, np.ndarray], float]],
    fractions: np.array = np.asarray([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]),
) -> Dict[Text, float]:
  """Evaluates an `estimator` on the `dataset`.

  Supports deferred prediction, in which the estimator is allowed to "defer",
    or avoid prediction on, the (1 - fraction) least confident examples as
    determined by the predictive uncertainty of the estimator.

  Args:
    estimator: `lambda x: mu_x, uncertainty_x`, an uncertainty estimation
      function, which returns `mean_x` and predictive `uncertainty_x`.
    dataset: `tf.data.Dataset`, on which dataset to perform evaluation.
    metrics: `Dict[Text, Callable[[np.ndarray, np.ndarray], float]], keys are
      the name of the metric, and the value is a callable which provides a score
      given ground truths and predictions.
    fractions: `np.array`, specifies the proportion of the evaluation set to
      "retain" (i.e. not defer).

  Returns:
    Dict with format key: str = f'metric_retain_{fraction}', and
    value: `pandas.DataFrame` with columns ["retained_data", "mean", "std"],
    that summarizes the scores at different data retained fractions.
  """
  # Containers used for caching performance evaluation
  y_true = list()
  y_pred = list()
  y_uncertainty = list()

  # Convert to NumPy iterator if necessary
  ds = dataset if inspect.isgenerator(dataset) else tfds.as_numpy(dataset)

  for x, y in ds:
    # Sample from probabilistic model
    mean, uncertainty = estimator(x)

    # Cache predictions
    y_true.append(y)
    y_pred.append(mean)
    y_uncertainty.append(uncertainty)

  # Use vectorized NumPy containers
  y_true = np.concatenate(y_true).flatten()
  y_pred = np.concatenate(y_pred).flatten()
  y_uncertainty = np.concatenate(y_uncertainty).flatten()

  # pylint: disable=g-complex-comprehension
  # Make sure to close over metric_fn.
  return {
      metric: evaluate_metric(
          y_true,
          y_pred,
          y_uncertainty,
          fractions,
          lambda y_true, y_pred, fn=metric_fn: fn(y_true, y_pred).numpy(),
          name=metric) for (metric, metric_fn) in metrics.items()
  }
  # pylint: enable=g-complex-comprehension


def evaluate_metric(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_uncertainty: np.ndarray,
                    fractions: Sequence[float],
                    metric_fn: Callable[[np.ndarray, np.ndarray],
                                        Union[float, tf.Tensor]],
                    name: str = None,
                    return_df: bool = True):
  """Evaluate model predictive dist on `metric_fn` at data retain `fractions`.

  Args:
    y_true: `numpy.ndarray`, the ground truth labels, with shape [N].
    y_pred: `numpy.ndarray`, the model predictions, with shape [N].
    y_uncertainty: `numpy.ndarray`, the model uncertainties, with shape [N].
    fractions: `iterable`, the percentages of data to retain for calculating
      `metric_fn`.
    metric_fn: `lambda(y_true, y_pred) -> float`, a metric function that
      provides a score given ground truths and predictions.
    name: (optional) `str`, the name of the method.
    return_df: Whether or not to return a numpy array or a pandas dataframe.

  Returns:
    if return_df:
      A `pandas.DataFrame` with columns ["retained_data", "mean", "std"],
        that summarizes the scores at different data retained fractions.
    else:
      `np.array`, contains result of metric evaluation for each fraction
        in fractions.
  """
  n = y_true.shape[0]

  # Sorts indexes by ascending uncertainty
  i_uncertainties = np.argsort(y_uncertainty)

  # Score containers
  mean = np.empty_like(fractions)
  # TODO(nband): do bootstrap sampling and estimate standard error
  std = np.zeros_like(fractions)

  for i, frac in enumerate(fractions):
    # Keep only the %-frac of lowest uncertainties.
    idx = np.zeros(n, dtype=bool)
    idx[i_uncertainties[:int(n * frac)]] = True
    mean[i] = metric_fn(y_true[idx], y_pred[idx])

  if return_df:
    # Store
    df = pd.DataFrame(dict(retained_data=fractions, mean=mean, std=std))
    df.name = name
    return df
  else:
    return mean


def wrap_retinopathy_estimator(estimator, use_mixed_precision):
  """Models used in the Diabetic Retinopathy baseline output logits by default.

  Apply conversion if necessary based on mixed precision setting, and apply
  a sigmoid to obtain sigmoid probability [0.0, 1.0] for the model.

  Args:
    estimator: a `tensorflow.keras.model` probabilistic model, that accepts
      input with shape [B, H, W, 3] and outputs logits
    use_mixed_precision: bool, whether to use mixed precision.

  Returns:
     wrapped estimator, outputting sigmoid probabilities.
  """
  def estimator_wrapper(inputs, training, estimator):
    logits = estimator(inputs, training=training)
    if use_mixed_precision:
      logits = tf.cast(logits, tf.float32)
    probs = tf.squeeze(tf.nn.sigmoid(logits))
    return probs.numpy()

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


def update_metrics_keras(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_uncertainty: np.ndarray,
    metrics_dict: Dict[str, Any],
    test_metric_fns: Dict[str, Optional[Callable[[Any, Any], Any]]],
    fractions: Sequence[float]):
  """Update per-epoch metrics using respective tf.keras.metrics-like class instances.

    Relies on the tf.keras.metrics-like objects having methods
      (i) update_state()
      (ii) result(), which is not called in this method, but will be
        later to perform metric reduction as needed.

  Given the metrics computed thus far in training (`metrics_dict`),
  the deferred prediction fractions, which specify the proportion of the
  data retained (`fractions`), update and return the metrics_dict.

  Args:
    y_true: `numpy.ndarray`, the ground truth labels, with shape [N].
    y_pred: `numpy.ndarray`, the model predictions, with shape [N].
    y_uncertainty: `numpy.ndarray`, the model uncertainties, with shape [N].
    metrics_dict: `Dict`, the metrics computed thus far in training. Unlike the
      rest of our deferred prediction methods, we here expect
      tf.keras.metrics-like values which each have method .update_state().
    test_metric_fns: `Dict[str, Union[None, Callable]]`, the names of the
      metrics to be updated. If the respective value is included, we use this
      function to process the `tf.Tensor` `y_true` and `y_pred`.
    fractions: `Sequence[float]`, the deferred prediction fractions, which
      specify the proportion of the data retained.
  """
  n = y_true.shape[0]

  # Sorts indexes by ascending uncertainty
  i_uncertainties = np.argsort(y_uncertainty)

  for retain_fraction in fractions:
    # TODO(nband): do bootstrap sampling and estimate standard error

    # Keep only the %-frac of lowest uncertainties
    idx = np.zeros(n, dtype=bool)
    idx[i_uncertainties[:int(n * retain_fraction)]] = True

    y_true_frac = y_true[idx]
    y_pred_frac = y_pred[idx]

    for metric_key, metric_fn in test_metric_fns.items():
      retain_metric_key = f'test_retain_{retain_fraction}/{metric_key}'

      # Special treatment for rm.metrics.ExpectedCalibrationError
      if metric_key == 'ece':
        metrics_dict[retain_metric_key].add_batch(
            y_pred_frac, label=y_true_frac)
      elif metric_fn is None:
        metrics_dict[retain_metric_key].update_state(y_true_frac, y_pred_frac)
      else:
        mean = metric_fn(y_true_frac, y_pred_frac)
        metrics_dict[retain_metric_key].update_state(mean)


def store_keras_metrics(metrics_dict: Dict[str, Any],
                        model_type: str,
                        model_results_path: str,
                        train_seed: int,
                        eval_seed: int,
                        return_parsed_dict: bool = True):
  """Parses a dict of metrics for values obtained at various retain proportions.

  Stores results to a DataFrame at the specified path (or updates a DataFrame
  that already exists at the path).

  Optionally, returns a dict with retain proportions separated,
  which allows for more natural logging of tf.Summary values and downstream
  TensorBoard visualization.

  Args:
    metrics_dict: `Dict`, metrics computed through deferred prediction
      evaluation, e.g., through `run_deferred_prediction.py`.
    model_type: `str`, type of model that was evaluated (e.g., 'deterministic').
      See run_deferred_prediction.py.
    model_results_path: `str`, path at which model results DataFrame should be
      stored.
    train_seed: `int`, seed used for training the model currently being
      evaluated.
    eval_seed: `int`, seed used in evaluating the current model (e.g., for MC
      dropout mask sampling).
    return_parsed_dict: `bool`, will return a dict with retain proportions
      separated, which is easier to use for logging to TensorBoard.

  Returns:
    `Optional[Dict]`
  """
  results = []
  if return_parsed_dict:
    parsed_dict = collections.defaultdict(list)

  for name, metric in metrics_dict.items():
    prefix, metric_name = name.split('/')

    # Only consider metrics collected at each deferred prediction fraction
    if prefix.endswith('test'):
      continue

    retain_proportion = float(prefix.split('_')[-1])
    metric_tensor = metric.result()
    metric_scalar = metric_tensor.numpy()
    results.append((metric_name, retain_proportion, metric_scalar))

    if return_parsed_dict:
      parsed_dict[metric_name].append((retain_proportion, metric_tensor))

  new_results_df = pd.DataFrame(
      results, columns=['metric', 'retain_proportion', 'value'])

  # Add metadata
  new_results_df['model_type'] = model_type
  new_results_df['train_seed'] = train_seed
  new_results_df['eval_seed'] = eval_seed
  new_results_df['run_datetime'] = datetime.datetime.now()
  new_results_df['run_datetime'] = pd.to_datetime(
      new_results_df['run_datetime'])

  model_results_path = os.path.join(model_results_path, 'results.tsv')

  # Update or initialize results DataFrame
  try:
    with tf.io.gfile.GFile(model_results_path, 'r') as f:
      previous_results_df = pd.read_csv(f, sep='\t')
    results_df = pd.concat([previous_results_df, new_results_df])
    action_str = 'updated'
  except (FileNotFoundError, tf.errors.NotFoundError):
    print(f'No previous results found at path {model_results_path}. '
          f'Storing a new results dataframe.')
    results_df = new_results_df
    action_str = 'stored initial'

  # Store to file
  with tf.io.gfile.GFile(model_results_path, 'w') as f:
    results_df.to_csv(path_or_buf=f, sep='\t', index=False)

  print(f'Successfully {action_str} results dataframe at {model_results_path}.')

  if return_parsed_dict:
    for metric_name in parsed_dict.keys():
      # Sort by ascending retain proportion
      parsed_dict[metric_name] = sorted(parsed_dict[metric_name])
    return parsed_dict
