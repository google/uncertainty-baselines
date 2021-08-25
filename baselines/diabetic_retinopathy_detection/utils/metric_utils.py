import functools
import logging
from typing import Any, Dict, List, Optional

import robustness_metrics as rm
import tensorflow as tf
from tabulate import tabulate

"""Utils for initializing, updating, and logging metrics."""


def get_test_metrics(
    test_metric_classes: Dict[str, Any],
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Add a unique test metric for each fraction of data retained in deferred
    prediction.

    If deferred_prediction_fractions is unspecified (= None), we are simply
    evaluating on the full test set, and don't specify the suffix `retain_X`.

  Args:
    test_metric_classes: dict, keys are the name of the test metric and values
      are the classes, which we use for instantiation.
    deferred_prediction_fractions: List[float], used to create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on the
      remaining "retained" X% of points.

  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  def update_test_metrics(retain_fraction):
    for metric_key, metric_class in test_metric_classes.items():
      if retain_fraction is None:
        retain_metric_key = f'test/{metric_key}'
      else:
        retain_metric_key = f'test_retain_{retain_fraction}/{metric_key}'

      test_metrics.update({retain_metric_key: metric_class()})

  if deferred_prediction_fractions is None:
    deferred_prediction_fractions = [None]

  # Update test metrics dict, for each retain fraction and metric
  for fraction in deferred_prediction_fractions:
    update_test_metrics(retain_fraction=fraction)

  return test_metrics


# def get_diabetic_retinopathy_test_metric_fns(use_tpu: bool):
#   """Construct dict with keys = metric name, and values described below.
#
#   Possible values:
#     None: if no additional preprocessing should be applied to the y_true
#       and y_pred tensors before metric.update_state(y_true, y_pred) is called
#       each epoch.
#     metric_fn: `Callable[[tf.Tensor, tf.Tensor], tf.Tensor]`, which will be
#       used for preprocessing the y_true and y_pred tensors, such that the
#       output of the metric_fn will be used in metric.update_state(output)
#       during each epoch.
#   Args:
#     use_tpu: bool, ECE is not yet supported on TPU.
#
#   Returns:
#     test_metric_fns, dict containing our test metrics and preprocessing methods
#   """
#   test_metric_fns = {
#       'negative_log_likelihood': negative_log_likelihood_metric,
#       'accuracy': None,
#       'auprc': None,
#       'auroc': None
#   }
#
#   if not use_tpu:
#     test_metric_fns['ece'] = None
#
#   return test_metric_fns


def get_diabetic_retinopathy_test_metric_classes(use_tpu: bool, num_bins: int):
  """Retrieve the relevant metric classes for test set evaluation.

  Do not yet instantiate the test metrics - this must be done for each
  fraction in deferred prediction.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.

  Returns:
    dict, test_metric_classes
  """
  test_metric_classes = {
      'negative_log_likelihood': tf.keras.metrics.Mean,
      'accuracy': tf.keras.metrics.BinaryAccuracy
  }

  if use_tpu:
    test_metric_classes.update({
        'auprc': functools.partial(tf.keras.metrics.AUC, curve='PR'),
        'auroc': functools.partial(tf.keras.metrics.AUC, curve='ROC')
    })
  else:
    test_metric_classes.update({
        'ece': functools.partial(
            rm.metrics.ExpectedCalibrationError, num_bins=num_bins)
    })

  return test_metric_classes


def get_diabetic_retinopathy_base_test_metrics(
    use_tpu: bool,
    num_bins: int,
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize base test metrics for non-ensemble Diabetic Retinopathy.

  predictors.

  Should be called within the distribution strategy scope (e.g. see
  eval_deferred_prediction.py script).

  Note:
    We disclude AUCs in non-TPU case, which must be defined and added to this
    dict outside the strategy scope. See the
    `get_diabetic_retinopathy_cpu_eval_metrics` method for initializing AUC.
    We disclude ECE in TPU case, which currently throws an XLA error on TPU.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on the
      remaining "retained" X% of points. If default (None), just create one copy
      of each metric for evaluation on full test set (no deferral).

  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  # Retrieve unitialized test metric classes
  test_metric_classes = get_diabetic_retinopathy_test_metric_classes(
      use_tpu=use_tpu, num_bins=num_bins)

  # Create a test metric for each deferred prediction fraction
  test_metrics.update(
      get_test_metrics(test_metric_classes, deferred_prediction_fractions))

  return test_metrics


def get_diabetic_retinopathy_base_metrics(
    use_tpu: bool,
    num_bins: int,
    available_splits: List[str],
    use_validation: bool = True,
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize base metrics for non-ensemble Diabetic Retinopathy predictors.

  Should be called within the distribution strategy scope (e.g. see
  deterministic.py script).

  Note:
    We disclude AUCs in non-TPU case, which must be defined and added to this
    dict outside the strategy scope. See the
    `get_diabetic_retinopathy_cpu_eval_metrics` method for initializing AUCs.
    We disclude ECE in TPU case, which currently throws an XLA error on TPU.

  Args:
    use_tpu: bool, is run using TPU.
    num_bins: int, number of ECE bins.
    use_validation: whether to use a validation split.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on
      the remaining "retained" X% of points.
      If default (None), just create one copy of each metric for evaluation
        on full test set (no deferral).

  Returns:
    dict, metrics
  """
  validation_datasets = [key for key in available_splits
                         if 'validation' in key]
  test_datasets = [key for key in available_splits
                   if 'test' in key]
  eval_datasets = test_datasets
  if use_validation:
    eval_datasets = validation_datasets + eval_datasets

  metrics = {
      'train/negative_log_likelihood': tf.keras.metrics.Mean(),
      'train/accuracy': tf.keras.metrics.BinaryAccuracy(),
      'train/loss': tf.keras.metrics.Mean(),  # NLL + L2
  }
  for eval_split in eval_datasets:
    metrics.update({
      f'{eval_split}/negative_log_likelihood': tf.keras.metrics.Mean(),
      f'{eval_split}/accuracy': tf.keras.metrics.BinaryAccuracy(),
    })

  if use_tpu:
    # AUC does not yet work within GPU strategy scope, but does for TPU
    metrics.update({
        'train/auprc': tf.keras.metrics.AUC(curve='PR'),
        'train/auroc': tf.keras.metrics.AUC(curve='ROC'),
    })
    for eval_split in eval_datasets:
      metrics.update({
        f'{eval_split}/auprc': tf.keras.metrics.AUC(curve='PR'),
        f'{eval_split}/auroc': tf.keras.metrics.AUC(curve='ROC')
      })
  else:
    # ECE does not yet work on TPU
    metrics.update({
        'train/ece': rm.metrics.ExpectedCalibrationError(num_bins=num_bins),
    })
    for eval_split in eval_datasets:
      metrics.update({
        f'{eval_split}/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=num_bins)
      })

  # # Retrieve unitialized test metric classes
  # test_metric_classes = get_diabetic_retinopathy_test_metric_classes(
  #     use_tpu=use_tpu, num_bins=num_bins)
  #
  # # Create a test metric for each deferred prediction fraction
  # metrics.update(
  #     get_test_metrics(test_metric_classes, deferred_prediction_fractions))

  return metrics


def get_diabetic_retinopathy_cpu_test_metrics(
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize test metrics for non-ensemble Diabetic Retinopathy predictors.

  that must be initialized outside of the accelerator scope for CPU evaluation.

  Note that this method will cause an error on TPU.

  Args:
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on the
      remaining "retained" X% of points. If default (None), just create one copy
      of each metric for evaluation on full test set (no deferral).

  Returns:
    dict, test_metrics
  """
  test_metrics = {}

  # Do not yet instantiate the test metrics -
  # this must be done for each fraction in deferred prediction
  test_metric_classes = {
      'auprc': functools.partial(tf.keras.metrics.AUC, curve='PR'),
      'auroc': functools.partial(tf.keras.metrics.AUC, curve='ROC')
  }

  # Create a test metric for each deferred prediction fraction
  test_metrics.update(
      get_test_metrics(test_metric_classes, deferred_prediction_fractions))

  return test_metrics


def get_diabetic_retinopathy_cpu_metrics(
    available_splits: List[str],
    use_validation: bool = True,
    deferred_prediction_fractions: Optional[List[float]] = None):
  """Initialize metrics for non-ensemble Diabetic Retinopathy predictors.

  that must be initialized outside of the accelerator scope for CPU evaluation.

  Note that this method will cause an error on TPU.

  Args:
    use_validation: whether to use a validation split.
    deferred_prediction_fractions: Optional[List[float]], create a copy of each
      metric for each of the specified fractions. Each metric will be evaluated
      at each deferral percentage, i.e., using an uncertainty score to defer
      prediction on (100 - X)% of the test set, and evaluating the metric on the
      remaining "retained" X% of points. If default (None), just create one copy
      of each metric for evaluation on full test set (no deferral).

  Returns:
    dict, metrics
  """
  metrics = {
      'train/auprc': tf.keras.metrics.AUC(curve='PR'),
      'train/auroc': tf.keras.metrics.AUC(curve='ROC')
  }

  validation_datasets = [key for key in available_splits
                         if 'validation' in key]
  test_datasets = [key for key in available_splits
                   if 'test' in key]
  eval_datasets = test_datasets
  if use_validation:
    eval_datasets = validation_datasets + eval_datasets

  for eval_split in eval_datasets:
    metrics.update({
      f'{eval_split}/auprc': tf.keras.metrics.AUC(curve='PR'),
      f'{eval_split}/auroc': tf.keras.metrics.AUC(curve='ROC')
    })

  # # Do not yet instantiate the test metrics -
  # # this must be done for each fraction in deferred prediction
  # test_metric_classes = {
  #     'auprc': functools.partial(tf.keras.metrics.AUC, curve='PR'),
  #     'auroc': functools.partial(tf.keras.metrics.AUC, curve='ROC')
  # }
  #
  # # Create a test metric for each deferred prediction fraction
  # metrics.update(
  #     get_test_metrics(test_metric_classes, deferred_prediction_fractions))

  return metrics


def log_epoch_metrics_new(metrics, eval_results, use_tpu, dataset_splits):
  metrics_to_return = {}

  if 'train' in dataset_splits:
    train_columns = ['Train Loss (NLL+L2)', 'Accuracy', 'AUPRC', 'AUROC']
    train_metrics = ['loss', 'accuracy', 'auprc', 'auroc']
    train_values = [metrics['train/loss'].result(),
                    metrics['train/accuracy'].result() * 100,
                    metrics['train/auprc'].result() * 100,
                    metrics['train/auroc'].result() * 100]
    if not use_tpu:
      train_columns.append('ECE')
      train_values.append(metrics['train/ece'].result()['ece'] * 100)
      train_metrics.append('ece')

    train_table = tabulate(
      [train_values], train_columns, tablefmt="simple", floatfmt="8.4f")
    print(train_table)

    # Log to the metrics dict which we will return (for TensorBoard)
    for train_metric, train_value in zip(train_metrics, train_values):
      metrics_to_return[f'train/{train_metric}'] = train_value

  # Standard evaluation, robustness, and uncertainty quantification metrics
  eval_columns = [
    'Eval Dataset',
    'NLL', 'Accuracy', 'AUPRC', 'AUROC', 'ECE',
    'OOD AUROC', 'OOD AUPRC',
    'R-Accuracy AUC', 'R-NLL AUC', 'R-AUROC AUC', 'R-AUPRC AUC',
    'Balanced R-Accuracy AUC', 'Balanced R-NLL AUC',
    'Balanced R-AUROC AUC', 'Balanced R-AUPRC AUC']
  eval_metrics = [
    'negative_log_likelihood', 'accuracy', 'auprc', 'auroc', 'ece',
    'ood_detection_auroc', 'ood_detection_auprc',
    'retention_accuracy_auc', 'retention_nll_auc',
    'retention_auroc_auc', 'retention_auprc_auc',
    'balanced_retention_accuracy_auc', 'balanced_retention_nll_auc',
    'balanced_retention_auroc_auc', 'balanced_retention_auprc_auc']

  eval_values = list()
  for dataset_key, results_dict in eval_results.items():
    dataset_values = list()
    dataset_values.append(dataset_key)

    # Add all the relevant metrics from the per-dataset split results dict
    for eval_metric in eval_metrics:
      dataset_key_and_metric = f'{dataset_key}/{eval_metric}'
      eval_value = results_dict[dataset_key_and_metric]
      dataset_values.append(eval_value)

      # Add to the metrics dict which we will return (for TensorBoard logging)
      metrics_to_return[dataset_key_and_metric] = eval_value

    eval_values.append(dataset_values)

  eval_table = tabulate(eval_values, eval_columns,
                        tablefmt="simple", floatfmt="8.4f")
  print('\n')
  print(eval_table)
  return metrics_to_return


def log_epoch_metrics(metrics, use_tpu, dataset_splits):
  """Log epoch metrics -- different metrics supported depending on TPU use.

  Args:
    metrics: dict, contains all train/test metrics evaluated for the run.
    use_tpu: bool, is run using TPU.
    dataset_splits: list, all splits on which we computed metrics
  """
  evaluation_splits = [split for split in dataset_splits if
                       'validation' in split or 'test' in split]
  if use_tpu:
    if 'train' in dataset_splits:
      logging.info(
        'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
        'AUPRC: %.2f%%, AUROC: %.2f%%',
        metrics['train/loss'].result(),
        metrics['train/accuracy'].result() * 100,
        metrics['train/auprc'].result() * 100,
        metrics['train/auroc'].result() * 100)
    for split in evaluation_splits:
      logging.info(
        f'{split} NLL: %.4f, Accuracy: %.2f%%, '
        f'AUPRC: %.2f%%, AUROC: %.2f%%',
        metrics[f'{split}/negative_log_likelihood'].result(),
        metrics[f'{split}/accuracy'].result() * 100,
        metrics[f'{split}/auprc'].result() * 100,
        metrics[f'{split}/auroc'].result() * 100)
  else:
    if 'train' in dataset_splits:
      logging.info(
        'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, '
        'AUPRC: %.2f%%, AUROC: %.2f%%, ECE: %.2f%%',
        metrics['train/loss'].result(),
        metrics['train/accuracy'].result() * 100,
        metrics['train/auprc'].result() * 100,
        metrics['train/auroc'].result() * 100,
        metrics['train/ece'].result()['ece'] * 100)
    for split in evaluation_splits:
      logging.info(
        f'{split} NLL: %.4f, Accuracy: %.2f%%, '
        f'AUPRC: %.2f%%, AUROC: %.2f%%, ECE: %.2f%%',
        metrics[f'{split}/negative_log_likelihood'].result(),
        metrics[f'{split}/accuracy'].result() * 100,
        metrics[f'{split}/auprc'].result() * 100,
        metrics[f'{split}/auroc'].result() * 100,
        metrics[f'{split}/ece'].result()['ece'] * 100)


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
