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

"""Utils for OOD evaluation.

Referneces:

[1]: Lee, Kimin, et al. "A simple unified framework for detecting
  out-of-distribution samples and adversarial attacks." Advances in neural
  information processing systems 31 (2018).
  https://arxiv.org/abs/1807.03888

"""

import jax
import numpy as np
import scipy
import sklearn.metrics


SUPPORTED_OOD_METRICS = ('msp', 'entropy', 'maha', 'rmaha')


# TODO(dusenberrymw): Move it to robustness metrics.
def compute_ood_metrics(targets,
                        predictions,
                        tpr_thres=0.95,
                        targets_threshold=None):
  """Computes Area Under the ROC and PR curves and FPRN.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall
  FPRN - False positive rate at which true positive rate is N.

  Args:
    targets: np.ndarray of targets, either 0 or 1, or continuous values.
    predictions: np.ndarray of predictions, any value.
    tpr_thres: float, threshold for true positive rate.
    targets_threshold: float, if target values are continuous values, this
      threshold binarizes them.

  Returns:
    A dictionary with AUC-ROC, AUC-PR, and FPRN scores.
  """

  if targets_threshold is not None:
    targets = np.array(targets)
    targets = np.where(targets < targets_threshold,
                       np.zeros_like(targets, dtype=np.int32),
                       np.ones_like(targets, dtype=np.int32))

  fpr, tpr, _ = sklearn.metrics.roc_curve(targets, predictions)
  fprn = fpr[np.argmax(tpr >= tpr_thres)]

  return {
      'auc-roc': sklearn.metrics.roc_auc_score(targets, predictions),
      'auc-pr': sklearn.metrics.average_precision_score(targets, predictions),
      'fprn': fprn,
  }


class OODMetric:
  """OOD metric class that stores scores and OOD labels."""

  def __init__(self, metric_name):
    if metric_name not in SUPPORTED_OOD_METRICS:
      raise NotImplementedError(
          ('Only msp, maha, and rmaha are supported for OOD evaluation!',
           'Got metric_name=%s!') % metric_name)
    self.metric_name = metric_name
    self.scores = []
    self.labels = []

  def update(self, scores, labels):
    self.scores += list(scores)
    self.labels += list(labels)

  def get_scores_and_labels(self):
    return self.scores, self.labels

  def get_metric_name(self):
    return self.metric_name

  def compute_ood_scores(self, scores):
    """Compute OOD scores.

    Compute OOD scores that indicate uncertainty.

    Args:
      scores: A dict that contains scores for computing OOD scores. A full dict
        can contain probs, Mahalanobis distance, and Relative Mahalanobis
        distance. The scores should be of the size [batch_size, num_classes]

    Returns:
      OOD scores: OOD scores that indicate uncertainty. Should be of the size
        [batch_size, ]

    Raises:
      KeyError: An error occurred when the corresponding scores needed for
        computing OOD scores are not found in the scores dict.
    """
    if self.metric_name == 'msp':
      if 'probs' in scores:
        ood_scores = 1 - np.max(scores['probs'], axis=-1)
      else:
        raise KeyError(
            ('The variable probs is needed for computing MSP OOD score. ',
             'But it is not found in the dict.'))
    elif self.metric_name == 'entropy':
      if 'entropy' in scores:
        ood_scores = scores['entropy']
      else:
        raise KeyError(
            'The variable entropy is needed for computing Entropy OOD score.',
            'But it is not found in the dict.')
    elif self.metric_name == 'maha':
      if 'dists' in scores:
        ood_scores = np.min(scores['dists'], axis=-1)
      else:
        raise KeyError(
            ('The variable dists is needed for computing Mahalanobis distance ',
             'OOD score. But it is not found in the dict.'))
    elif self.metric_name == 'rmaha':
      if 'dists' in scores and 'dists_background' in scores:
        ood_scores = np.min(
            scores['dists'], axis=-1) - scores['dists_background'].reshape(-1)
      else:
        raise KeyError((
            'The variable dists and dists_background are needed for computing ',
            'Mahalanobis distance OOD score. But it is not found in the dict.'))
    return ood_scores

  def compute_metrics(self, tpr_thres=0.95, targets_threshold=None):
    return compute_ood_metrics(
        self.labels,
        self.scores,
        tpr_thres=tpr_thres,
        targets_threshold=targets_threshold)


def compute_mean_and_cov(embeds, labels):
  """Computes class-specific means and shared covariance matrix of given embedding.

  The computation follows Eq (1) in [1].

  Args:
    embeds: An np.array of size [n_train_sample, n_dim], where n_train_sample is
      the sample size of training set, n_dim is the dimension of the embedding.
    labels: An np.array of size [n_train_sample, ]

  Returns:
    mean_list: A list of len n_class, and the i-th element is an np.array of
    size [n_dim, ] corresponding to the mean of the fitted Guassian distribution
    for the i-th class.
    cov: The shared covariance mmatrix of the size [n_dim, n_dim].
  """
  n_dim = embeds.shape[1]
  n_class = int(np.max(labels)) + 1
  mean_list = []
  cov = np.zeros((n_dim, n_dim))

  for class_id in range(n_class):
    data = embeds[labels == class_id]
    data_mean = np.mean(data, axis=0)
    cov += np.dot((data - data_mean).T, (data - data_mean))
    mean_list.append(data_mean)
  cov = cov / len(labels)
  return mean_list, cov


def compute_mahalanobis_distance(embeds, mean_list, cov, epsilon=1e-20):
  """Computes Mahalanobis distance between the input to the fitted Guassians.

  The computation follows Eq.(2) in [1].

  Args:
    embeds: An np.array of size [n_test_sample, n_dim], where n_test_sample is
      the sample size of the test set, n_dim is the size of the embeddings.
    mean_list: A list of len n_class, and the i-th element is an np.array of
      size [n_dim, ] corresponding to the mean of the fitted Guassian
      distribution for the i-th class.
    cov: The shared covariance mmatrix of the size [n_dim, n_dim].
    epsilon: The small value added to the diagonal of the covariance matrix to
      avoid singularity.

  Returns:
    out: An np.array of size [n_test_sample, n_class] where the [i, j] element
    corresponds to the Mahalanobis distance between i-th sample to the j-th
    class Guassian.
  """
  n_sample = embeds.shape[0]
  n_class = len(mean_list)

  v = cov + np.eye(cov.shape[0], dtype=int) * epsilon  # avoid singularity
  vi = np.linalg.inv(v)
  means = np.array(mean_list)

  out = np.zeros((n_sample, n_class))
  for i in range(n_sample):
    x = embeds[i]
    out[i, :] = np.diag(np.dot(np.dot((x - means), vi), (x - means).T))
  return out


def load_ood_datasets(
    dataset,
    ood_dataset,
    ood_split,
    pp_eval,
    ood_methods,
    train_split,
    data_dir,
    get_data_fn,
):
  """Load datasets for OOD evaluation.

  The datasets should include in-distribution test dataset, OOD test dataset,
  and in-distribution training dataset if Mahalanobis distance based method is
  applied.

  Args:
    dataset: The name of in-distribution dataset.
    ood_dataset: The name of OOD dataset.
    ood_split: The split of the OOD dataset.
    pp_eval: The pre-processing method applied to the input data.
    ood_methods: The OOD methods used for evaluation. Can be choose from 'msp',
      'maha', 'rmaha'.
    train_split: The split of the training in-distribution dataset.
    data_dir: The data directory.
    get_data_fn: A function for generates a tuple of (data iterator, num_steps)
      given a dataset name or builder, split, preprocessing function, and
      optional data_dir.

  Returns:
    ood_ds: A dictionary with dataset label as the key and dataset iterator as
    the value.
    ood_ds_names: A list of dataset labels.
  """
  ood_ds = {}
  if isinstance(ood_split, str):
    ood_ds.update({
        'ind': get_data_fn(dataset, ood_split, pp_eval, data_dir),
        'ood': get_data_fn(ood_dataset, ood_split, pp_eval, data_dir),
    })
    ood_ds_names = list(ood_ds.keys())
  else:
    raise NotImplementedError(
        'Only string type of ood_split is supported for OOD evaluation! Got ood_split=%s!'
        % str(ood_split))

  if 'maha' in ood_methods or 'rmaha' in ood_methods:
    # Adding training set for fitting class conditional Gaussian for
    # Mahalanoabis distance based method
    if isinstance(train_split, str):
      ood_ds.update(
          {'train_maha': get_data_fn(dataset, train_split, pp_eval, data_dir)})
      ood_ds_names.insert(0, 'train_maha')
    else:
      raise NotImplementedError(
          'Only string type of train_split is supported for OOD evaluation! Got train_split=%s!'
          % str(train_split))
  return ood_ds, ood_ds_names


def eval_ood_metrics(ood_ds, ood_ds_names, ood_methods, evaluation_fn,
                     opt_repl):
  """Evaluate the model for OOD detection and record metrics."""
  # MSP stands for maximum softmax probability, max(softmax(logits)).
  # MSP can be used as confidence score.
  # Maha stands for Mahalanobis distance between the test input and
  # fitted class conditional Gaussian distributions based on the
  # embeddings. Mahalanobis distance can be used as uncertainty score
  # or in other words, negative Mahalanobis distance can be used as
  # confidence score.
  # RMaha stnads for Relative Mahalanobis distance (Ren et al. 2021)
  # https://arxiv.org/abs/2106.09022
  ood_metrics = [OODMetric(name) for name in ood_methods]

  output = {}
  # Mean and cov of class conditional Guassian in Mahalanobis distance.
  # Mean_background and cov_background for the unified Guassian model
  # regardless of class labels for computing Relative Mahalanobis distance
  mean_list, cov = None, None
  mean_list_background, cov_background = None, None
  for val_name in ood_ds_names:
    # The dataset train_maha must come before ind and ood
    # because the train_maha will be used to esimate the class conditional
    # mean and shared covariance.
    val_iter, val_steps = ood_ds[val_name]
    ncorrect, loss, nseen = 0, 0, 0
    pre_logits_list, labels_list = [], []
    for _, batch in zip(range(val_steps), val_iter):
      batch_scores = {}
      batch_ncorrect, batch_losses, batch_n, batch_metric_args = evaluation_fn(
          opt_repl.target, batch['image'], batch['labels'], batch['mask'])
      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      loss += np.sum(np.array(batch_losses[0]))
      nseen += np.sum(np.array(batch_n[0]))

      # Here we parse batch_metric_args to compute OOD metrics.
      logits, labels, pre_logits, masks = batch_metric_args
      masks_bool = np.array(masks[0], dtype=bool)
      if not np.any(masks_bool):
        continue  # No valid examples in this batch.
      if val_name == 'train_maha':
        # For Mahalanobis distance, we need to first fit class conditional
        # Gaussian using training data.
        pre_logits_list.append(np.array(pre_logits[0])[masks_bool])
        labels_list.append(np.array(labels[0])[masks_bool])
      else:
        # Computes Mahalanobis distance.
        if mean_list is not None and cov is not None:
          dists = compute_mahalanobis_distance(
              np.array(pre_logits[0])[masks_bool], mean_list, cov)
          batch_scores['dists'] = dists

        if mean_list_background is not None and cov_background is not None:
          dists_background = compute_mahalanobis_distance(
              np.array(pre_logits[0])[masks_bool], mean_list_background,
              cov_background)
          batch_scores['dists_background'] = dists_background

        # Computes Maximum softmax probability (MSP)
        probs = jax.nn.softmax(logits[0], axis=-1)[masks_bool]
        batch_scores['probs'] = probs

        # Compute Entropy
        batch_scores['entropy'] = np.array(
            [scipy.stats.entropy(prob) for prob in probs])

        # Update metric state for each metric in ood_metrics
        for metric in ood_metrics:
          ood_scores = metric.compute_ood_scores(batch_scores)
          ood_labels = np.zeros_like(
              ood_scores) if val_name == 'ind' else np.ones_like(ood_scores)
          metric.update(ood_scores, ood_labels)

    if val_name == 'train_maha':
      pre_logits_train = np.vstack(np.vstack(pre_logits_list))
      labels_train = np.argmax(np.vstack(np.vstack(labels_list)), axis=-1)
      mean_list, cov = compute_mean_and_cov(pre_logits_train, labels_train)
      mean_list_background, cov_background = compute_mean_and_cov(
          pre_logits_train, np.zeros_like(labels_train))
    elif val_name == 'ind':
      output[f'{val_name}_prec@1'] = ncorrect / nseen
      output[f'{val_name}_loss'] = loss / nseen

  for metric in ood_metrics:
    metric_name = metric.get_metric_name()
    metric_values = metric.compute_metrics()
    output[f'ood_{metric_name}_auroc'] = metric_values['auc-roc']
    output[f'ood_{metric_name}_auprc'] = metric_values['auc-pr']
    output[f'ood_{metric_name}_fprn'] = metric_values['fprn']

  return output
