import datetime
import time
from typing import Union, List

import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import torch
from absl import logging
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             roc_curve, precision_recall_curve, auc)
from sklearn.preprocessing import LabelBinarizer
# from swag_utils import utils as swag_utils

from .metric_utils import log_epoch_metrics_new
from .results_storage_utils import (
  create_eval_results_dir, store_eval_results, add_joint_dicts,
  store_eval_metadata)


# Want to avoid using .numpy() within TPU or GPU strategy scope.
@tf.function
def eval_tpu_step(
  dataset_iterator, dataset_steps, strategy, estimator,
  estimator_args, uncertainty_estimator_fn, is_deterministic
):
  def step_fn(inputs):
    images = inputs['features']
    labels = inputs['labels']

    # Compute prediction, total, aleatoric, and epistemic
    # uncertainty estimates
    pred_and_uncert = uncertainty_estimator_fn(
      images, estimator, training_setting=False, **estimator_args)

    # Return a tuple
    y_true = labels
    y_pred = pred_and_uncert['prediction']
    y_pred_entropy = pred_and_uncert['predictive_entropy']
    y_pred_variance = pred_and_uncert['predictive_variance']

    if not is_deterministic:
      y_aleatoric_uncert = pred_and_uncert['aleatoric_uncertainty']
      y_epistemic_uncert = pred_and_uncert['epistemic_uncertainty']
    else:
      y_aleatoric_uncert = tf.zeros(0)
      y_epistemic_uncert = tf.zeros(0)

    return (
      y_true, y_pred, y_pred_entropy, y_pred_variance,
      y_aleatoric_uncert, y_epistemic_uncert)

  # Containers for storage of
  # predictions, ground truth, uncertainty estimates
  # Construct tf.TensorArrays to store model results

  # Cannot handle strings on TPU
  # names = tf.TensorArray(tf.string, size=0, dynamic_size=True)
  y_true = tf.TensorArray(tf.int32, size=0, dynamic_size=True)
  y_pred = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  y_pred_entropy = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  y_pred_variance = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  #
  # Have to define even if we don't use them
  y_aleatoric_uncert = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
  y_epistemic_uncert = tf.TensorArray(tf.float32, size=0, dynamic_size=True)

  for _ in tf.range(dataset_steps):
    result = strategy.run(step_fn, args=(next(dataset_iterator),))

    # Parse results
    (y_true_, y_pred_, y_pred_entropy_, y_pred_variance_,
     y_aleatoric_uncert_, y_epistemic_uncert_) = result

    # Convert from Per-Replica object to tuple
    if strategy.num_replicas_in_sync > 1:
      y_true_ = y_true_.values
      y_pred_ = y_pred_.values
      y_pred_entropy_ = y_pred_entropy_.values
      y_pred_variance_ = y_pred_variance_.values

      if not is_deterministic:
        y_aleatoric_uncert_ = y_aleatoric_uncert_.values
        y_epistemic_uncert_ = y_epistemic_uncert_.values

    # Iterate through per-batch results
    # This is written in a very un-Pythonic manner to have updates only
    # rely on arguments successfully passed to TPU scope
    for batch_result in y_true_:
      y_true = y_true.write(y_true.size(), batch_result)
    for batch_result in y_pred_:
      y_pred = y_pred.write(y_pred.size(), batch_result)
    for batch_result in y_pred_entropy_:
      y_pred_entropy = y_pred_entropy.write(y_pred_entropy.size(), batch_result)
    for batch_result in y_pred_variance_:
      y_pred_variance = y_pred_variance.write(
        y_pred_variance.size(), batch_result)

    if not is_deterministic:
      for batch_result in y_aleatoric_uncert_:
        y_aleatoric_uncert = y_aleatoric_uncert.write(
          y_aleatoric_uncert.size(), batch_result)
      for batch_result in y_epistemic_uncert_:
        y_epistemic_uncert = y_epistemic_uncert.write(
          y_epistemic_uncert.size(), batch_result)

  results_arrs = {
    'y_true': y_true.stack(),
    'y_pred': y_pred.stack(),
    'y_pred_entropy': y_pred_entropy.stack(),
    'y_pred_variance': y_pred_variance.stack()
  }
  if not is_deterministic:
    results_arrs['y_aleatoric_uncert'] = y_aleatoric_uncert.stack()
    results_arrs['y_epistemic_uncert'] = y_epistemic_uncert.stack()

  return results_arrs


def evaluate_model_on_datasets_tpu(
  strategy, datasets, steps, estimator, estimator_args,
  uncertainty_estimator_fn, eval_batch_size, is_deterministic=False,
  backend="tf", eval_step_jax=None,
):

  # Need to collect these so we can form joint datasets:
  # e.g., joint_test = in_domain_test UNION ood_test
  dataset_split_to_containers = {}

  for dataset_split, dataset in datasets.items():
    # Begin iteration for this dataset split
    start_time = time.time()
    dataset_iterator = iter(dataset)
    dataset_steps = steps[dataset_split]
    logging.info(f'Evaluating split {dataset_split}.')
    if backend == "jax":
      eval_epoch_arrs = eval_step_jax(
        dataset_iterator, dataset_steps, is_deterministic)
    elif backend == "tf":
      eval_epoch_arrs = eval_tpu_step(
        dataset_iterator, tf.convert_to_tensor(dataset_steps), strategy,
        estimator, estimator_args, uncertainty_estimator_fn, is_deterministic)
    else:
      raise NotImplementedError(f"backend {backend} is not supported yet.")

    # Update metadata
    time_elapsed = time.time() - start_time
    dataset_split_to_containers[dataset_split] = {}
    dataset_split_dict = dataset_split_to_containers[dataset_split]
    dataset_split_dict['total_ms_elapsed'] = time_elapsed * 1e6
    dataset_split_dict['dataset_size'] = dataset_steps * eval_batch_size

    # Use vectorized NumPy containers
    for eval_key, eval_arr in eval_epoch_arrs.items():
      dataset_split_dict[eval_key] = np.concatenate(eval_arr).flatten()
      # print(
      #   f'Concatenated {eval_key} into shape '
      #   f'{dataset_split_dict[eval_key].shape}')

    dataset_split_dict['y_pred'] = dataset_split_dict[
      'y_pred'].astype('float64')

  # Add Joint Dicts
  dataset_split_to_containers = add_joint_dicts(
    dataset_split_to_containers, is_deterministic=is_deterministic)

  return dataset_split_to_containers


def test_step_swag(
    model_to_eval, iterator, num_steps, eval_batch_size,
    sigmoid, device, image_h=512, image_w=512
):
  """Need to BatchNorm over the training set for every stochastic sample.
  Therefore, we:
    1. Sample from the posterior,
    2. (here) Compute a full epoch, and return predictions and labels,
    3. Update metrics altogether at the end.
"""

  def step_fn(inputs):
    images = inputs['features']
    labels = inputs['labels']
    images = torch.from_numpy(images._numpy()).view(eval_batch_size, 3,  # pylint: disable=protected-access
                                                    image_h,
                                                    image_w).to(device)
    labels = torch.from_numpy(
      labels._numpy()).to(device).float().unsqueeze(
      -1)  # pylint: disable=protected-access
    with torch.no_grad():
      logits = model_to_eval(images)

    probs = sigmoid(logits)
    return labels, probs

  labels_list = []
  probs_list = []
  model_to_eval.eval()
  for _ in range(num_steps):
    labels, probs = step_fn(next(iterator))
    labels_list.append(labels)
    probs_list.append(probs)

  return {
    'labels': torch.cat(labels_list, dim=0),
    'probs': torch.cat(probs_list, dim=0)
  }

def evaluate_swag_on_datasets(
  train_iterator, train_batch_size, eval_datasets, steps, eval_model,
  eval_batch_size, num_samples, swag_is_active,
  scale, sample_with_cov, device, epoch, sigmoid, image_h=512, image_w=512
):
  labels = []
  probs = []
  # dataset_split_to_containers = {}

  for sample in range(num_samples):
    # Sample from approx posterior
    if swag_is_active:
      eval_model.sample(scale=scale, cov=sample_with_cov)
      # BN Update requires iteration over full train loop, hence we
      # put the MC sampling outside of the evaluation loops.
      swag_utils.bn_update(
        train_iterator, eval_model, num_train_steps=steps['train'],
        train_batch_size=train_batch_size, image_h=512, image_w=512,
        device=device)

    for dataset_key, eval_dataset in eval_datasets.items():
      dataset_iterator = iter(eval_dataset)
      logging.info(
        f'Evaluating SWAG on {dataset_key}, sample {sample} '
        f'at epoch: %s', epoch + 1)

      epoch_results = test_step_swag(
        model_to_eval=eval_model, iterator=dataset_iterator,
        num_steps=steps[dataset_key], eval_batch_size=eval_batch_size,
        sigmoid=sigmoid, device=device, image_h=image_h, image_w=image_w)







def evaluate_model_and_compute_metrics(
    strategy, eval_datasets, steps, metrics, eval_estimator,
    uncertainty_estimator_fn, eval_batch_size, available_splits,
    estimator_args, is_deterministic=False, num_bins=15, use_tpu=True,
    backend="tf", eval_step_jax=None,
):
  """Main method for evaluation and computing metrics using arbitrary TF
  distribution strategies. Usable for evaluation during tuning."""
  # Compute predictions on all evaluation datasets
  eval_results = evaluate_model_on_datasets_tpu(
    strategy=strategy, datasets=eval_datasets, steps=steps,
    estimator=eval_estimator, estimator_args=estimator_args,
    uncertainty_estimator_fn=uncertainty_estimator_fn,
    eval_batch_size=eval_batch_size, is_deterministic=is_deterministic,
    backend=backend, eval_step_jax=eval_step_jax)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = compute_loss_and_accuracy_arrs_for_all_datasets(eval_results)

  # Compute all metrics for each dataset --
  # Robustness, Open Set Recognition, Retention AUC
  eval_results = compute_metrics_for_all_datasets(
    eval_results, ece_num_bins=num_bins, compute_retention_auc=True,
    verbose=False)

  # Log metrics
  total_results = log_epoch_metrics_new(
    metrics=metrics, eval_results=eval_results,
    use_tpu=use_tpu, dataset_splits=available_splits)

  return total_results


def evaluate_model_on_datasets(
  datasets, steps, estimator, estimator_args, uncertainty_estimator_fn,
  eval_batch_size
):
  # Need to collect these so we can form joint datasets:
  # e.g., joint_test = in_domain_test UNION ood_test
  dataset_split_to_containers = {}

  for dataset_split, dataset in datasets.items():
    # Containers for numpy storage of
    # image names, predictions, ground truth, uncertainty estimates
    names = list()
    y_true = list()
    y_pred = list()
    y_pred_entropy = list()
    y_pred_variance = list()
    y_aleatoric_uncert = list()
    y_epistemic_uncert = list()

    # Begin iteration for this dataset split
    start_time = time.time()
    dataset_iterator = iter(dataset)
    dataset_steps = steps[dataset_split]
    logging.info(f'Evaluating split {dataset_split}.')
    for step in range(dataset_steps):
      if step % 100 == 0:
        logging.info('Evaluated %d/%d batches.', step, dataset_steps)

      inputs = next(dataset_iterator)  # pytype: disable=attribute-error
      images = inputs['features']
      labels = inputs['labels']

      # Compute prediction, total, aleatoric, and epistemic
      # uncertainty estimates
      pred_and_uncert = uncertainty_estimator_fn(
        images, estimator, training_setting=False, **estimator_args)

      # Add this batch of predictions to the containers
      names.append(inputs['name'])
      y_true.append(labels.numpy())
      y_pred.append(pred_and_uncert['prediction'])
      y_pred_entropy.append(pred_and_uncert['predictive_entropy'])
      y_pred_variance.append(pred_and_uncert['predictive_variance'])
      y_aleatoric_uncert.append(pred_and_uncert['aleatoric_uncertainty'])
      y_epistemic_uncert.append(pred_and_uncert['epistemic_uncertainty'])

    # Update metadata
    time_elapsed = time.time() - start_time
    dataset_split_to_containers[dataset_split] = {}
    dataset_split_dict = dataset_split_to_containers[dataset_split]
    dataset_split_dict['total_ms_elapsed'] = time_elapsed * 1e6
    dataset_split_dict['dataset_size'] = dataset_steps * eval_batch_size

    dataset_split_dict['names'] = np.concatenate(names).flatten()
    dataset_split_dict['y_true'] = np.concatenate(y_true).flatten()
    dataset_split_dict['y_pred'] = np.concatenate(y_pred).flatten()
    # Use vectorized NumPy containers
    dataset_split_dict['y_pred_entropy'] = (
      np.concatenate(y_pred_entropy).flatten())
    dataset_split_dict['y_pred_variance'] = (
      np.concatenate(y_pred_variance).flatten())
    dataset_split_dict['y_aleatoric_uncert'] = (
      np.concatenate(y_aleatoric_uncert).flatten())
    dataset_split_dict['y_epistemic_uncert'] = (
      np.concatenate(y_epistemic_uncert).flatten())

  # Add Joint Dicts
  dataset_split_to_containers = add_joint_dicts(dataset_split_to_containers)

  return dataset_split_to_containers


def eval_model_and_store_results(
  datasets, steps, estimator, estimator_args, uncertainty_estimator_fn,
  eval_batch_size, model_type, eval_seed, output_dir,
  train_seed: Union[int, List[int]], k=None
):
  dataset_split_to_containers = evaluate_model_on_datasets(
    datasets, steps, estimator, estimator_args, uncertainty_estimator_fn,
    eval_batch_size)
  store_eval_results_and_metadata(
    dataset_split_to_containers, model_type, eval_seed,
    output_dir, train_seed, k=k)


def store_eval_results_and_metadata(
  dataset_split_to_containers, model_type, eval_seed, output_dir,
  train_seed: Union[int, List[int]], k=None
):
  run_datetime = datetime.datetime.now()

  for dataset_split, results_dict in dataset_split_to_containers.items():
    # Store results and metadata to file
    eval_results_dir = create_eval_results_dir(
      output_dir=output_dir, dataset_key=dataset_split, model_type=model_type,
      date_time=run_datetime, eval_seed=eval_seed, train_seed=train_seed, k=k)
    store_eval_results(eval_results_dir, results_dict)
    ms_per_example = results_dict['ms_per_example']
    store_eval_metadata(eval_results_dir, ms_per_example, k=k)


  # # Construct results df
  # results_df = pd.DataFrame(
  #   data={
  #     'name': names,
  #     'dataset_split': dataset_splits,
  #     'y_true': y_true,
  #     'y_pred': y_pred,
  #     'y_total_uncert': y_total_uncert,
  #     'y_aleatoric_uncert': y_aleatoric_uncert,
  #     'y_epistemic_uncert': y_epistemic_uncert
  # })
  # results_df['model_type'] = model_type
  # results_df['train_seed'] = train_seed
  # results_df['eval_seed'] = eval_seed
  # results_df['run_datetime'] = run_datetime
  # results_df['run_datetime'] = pd.to_datetime(results_df['run_datetime'])
  #
  # # Construct metadata df (run times)
  # metadata_df = pd.DataFrame(
  #   data={
  #     'dataset_split': dataset_split_for_metadata,
  #     'ms_per_example': ms_per_example
  # })
  # metadata_df['model_type'] = model_type
  # metadata_df['train_seed'] = train_seed
  # metadata_df['eval_seed'] = eval_seed
  # metadata_df['run_datetime'] = run_datetime
  # metadata_df['run_datetime'] = pd.to_datetime(metadata_df['run_datetime'])

  # return metadata_df, results_df


def compute_metrics_for_all_datasets(
    eval_results, ece_num_bins=15, compute_retention_auc=False,
    verbose=False
):
  for dataset_key, results_dict in eval_results.items():
    if verbose:
      logging.info(
        f'Computing metrics for dataset split {dataset_key}.')

    compute_open_set_recognition = 'joint' in dataset_key
    eval_results[dataset_key] = compute_dataset_eval_metrics(
      dataset_key, results_dict, ece_num_bins=ece_num_bins,
      compute_open_set_recognition=compute_open_set_recognition,
      compute_retention_auc=compute_retention_auc)

  return eval_results


def compute_loss_and_accuracy_arrs_for_all_datasets(eval_results):
  """
  Eval results is mapping from dataset_key to results
  :param eval_results:
  :return:
  """
  for dataset_key, results_dict in eval_results.items():
    eval_results[dataset_key] = compute_loss_and_accuracy_arrs(results_dict)

  return eval_results


def compute_loss_and_accuracy_arrs(results):
  """
  Results for a particular dataset, containing preds, uncertainty, etc.
  :param results:
  :return:
  """
  results = compute_log_loss_arr(results)
  y_pred, y_true = results['y_pred'], results['y_true']
  results['accuracy_arr'] = y_true == (y_pred > 0.5)
  return results


def compute_log_loss_arr(results, eps=1e-15):
  """Based on sklearn.preprocessing.log_loss, no aggregation."""
  y_pred, y_true = results['y_pred'], results['y_true']
  assert y_pred.shape == y_true.shape
  lb = LabelBinarizer()
  transformed_labels = lb.fit_transform(y_true)
  if transformed_labels.shape[1] == 1:
    transformed_labels = np.append(1 - transformed_labels,
                                   transformed_labels, axis=1)
  # Clipping
  y_pred = np.clip(y_pred, eps, 1 - eps)

  # If y_pred is of single dimension, assume y_true to be binary
  # and then check.
  if y_pred.ndim == 1:
    y_pred = y_pred[:, np.newaxis]
  if y_pred.shape[1] == 1:
    y_pred = np.append(1 - y_pred, y_pred, axis=1)

  y_pred /= y_pred.sum(axis=1)[:, np.newaxis]
  loss = -(transformed_labels * np.log(y_pred)).sum(axis=1)

  results['nll_arr'] = loss
  return results


def compute_dataset_eval_metrics(
    dataset_key, results, ece_num_bins=15,
    compute_open_set_recognition=False, compute_retention_auc=False
):
  """
  Args:
    dataset_key: str, name of dataset (prepends each metric in returned Dict)
    results: Dict, should include y_pred, y_true, y_pred_entropy
    and optionally is_ood (if we wish to do open set recognition, i.e.,
    have a joint eval set consisting of both in-domain and OOD examples).
    :param ece_num_bins:
  :return:
  """
  y_pred, y_true, y_pred_entropy = (
    results['y_pred'], results['y_true'], results['y_pred_entropy'])

  eval_metrics = dict()

  # Standard predictive metrics
  eval_metrics[f'{dataset_key}/negative_log_likelihood'] = log_loss(
    y_pred=y_pred, y_true=y_true)
  eval_metrics[f'{dataset_key}/auroc'] = roc_auc_score(
    y_true=y_true, y_score=y_pred)
  precision, recall, _ = precision_recall_curve(
    y_true=y_true, probas_pred=y_pred)
  eval_metrics[f'{dataset_key}/auprc'] = auc(recall, precision)
  eval_metrics[f'{dataset_key}/accuracy'] = (
    accuracy_score(y_true=y_true, y_pred=(y_pred > 0.5)))

  # Uncertainty metrics
  ece = rm.metrics.ExpectedCalibrationError(num_bins=ece_num_bins)
  ece.add_batch(y_pred, label=y_true)
  eval_metrics[f'{dataset_key}/ece'] = ece.result()['ece']

  if compute_open_set_recognition:
    is_ood = results['is_ood']
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = roc_auc_score(
      y_true=is_ood, y_score=y_pred_entropy)
    precision, recall, _ = precision_recall_curve(
      y_true=is_ood, probas_pred=y_pred_entropy)
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = auc(recall, precision)
  else:
    # This is added for convenience when logging (so the entry exists
    # in tabular format)
    eval_metrics[f'{dataset_key}/ood_detection_auroc'] = None
    eval_metrics[f'{dataset_key}/ood_detection_auprc'] = None

  if compute_retention_auc:
    assert 'accuracy_arr' in results
    assert 'nll_arr' in results
    eval_metrics[f'{dataset_key}/retention_accuracy_auc'] = np.mean(
      compute_retention_curve_on_accuracies(
        accuracies=results['accuracy_arr'], uncertainty=y_pred_entropy))
    eval_metrics[f'{dataset_key}/retention_nll_auc'] = np.mean(
      compute_retention_curve_on_losses(
        losses=results['nll_arr'], uncertainty=y_pred_entropy))

  return eval_metrics


def compute_roc_curve(y_uncertainty: np.ndarray, is_ood: np.ndarray):
  fpr, tpr, _ = roc_curve(y_true=is_ood, y_score=y_uncertainty)
  roc_auc = auc(x=fpr, y=tpr)
  return fpr, tpr, roc_auc


def compute_retention_curve_on_losses(losses, uncertainty):
  """Based on utils by Andrey Malinin, Yandex Research.
  https://github.com/yandex-research/shifts/blob/main/weather/assessment.py
  """
  n_objects = losses.shape[0]
  uncertainty_order = uncertainty.argsort()
  losses = losses[uncertainty_order]
  error_rates = np.zeros(n_objects + 1)
  error_rates[:-1] = np.cumsum(losses)[::-1] / n_objects
  return error_rates


def compute_retention_curve_on_accuracies(accuracies, uncertainty):
  """Based on utils by Andrey Malinin, Yandex Research.
  https://github.com/yandex-research/shifts/blob/main/weather/assessment.py
  """
  n_objects = accuracies.shape[0]
  uncertainty_order = uncertainty.argsort()
  accuracies = accuracies[uncertainty_order]
  cumul_oracle_collaborative_acc = np.zeros(n_objects + 1)

  for i in range(n_objects):
    j = n_objects - i
    cumul_oracle_collaborative_acc[i] = accuracies[:i].sum() + j

  cumul_oracle_collaborative_acc[-1] = accuracies.sum()

  acc_rates = cumul_oracle_collaborative_acc[::-1] / n_objects
  return acc_rates


def compute_ood_calibration_curve(y_pred: np.ndarray):
  """
  Determine the number of predictions with
  :param y_pred:
  :param n_buckets:
  :return:
  """
  assert y_pred.ndim == 1
  assert (np.max(y_pred) <= 1 and np.min(y_pred) >= 0), (
    'Expected input are probabilities.')

  # We want the probabiliity of the predicted class, so for all predictions
  # under 0.5 y^*, we transform as y^* = 1 - y^*
  sorted_y_pred = np.sort(y_pred)
  min_pred_above_halfway = None
  for i, pred in enumerate(sorted_y_pred):
    if pred >= 0.5:
      min_pred_above_halfway = i
      break

  sorted_y_pred[:min_pred_above_halfway] = (
      1 - sorted_y_pred[:min_pred_above_halfway])

  # Sort again
  sorted_y_pred = np.sort(sorted_y_pred)
  num_preds = len(sorted_y_pred)
  # num_buckets = min(num_buckets, num_preds)

  # Keep track of the number of predictions with
  # greater than or equal confidence

  num_preds_with_geq_confidence = []
  pointer = 0

  for pred_confidence in sorted_y_pred:
    while pointer < num_preds and sorted_y_pred[pointer] < pred_confidence:
      pointer += 1

    num_preds_with_geq_confidence.append(num_preds - pointer)

  sorted_y_pred = sorted_y_pred.tolist()
  sorted_y_pred.append(1)
  num_preds_with_geq_confidence.append(0)
  return np.array(sorted_y_pred), np.array(num_preds_with_geq_confidence)
