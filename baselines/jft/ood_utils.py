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

import numpy as np
import sklearn.metrics



# TODO(dusenberrymw): Move it to robustness metrics.
def ood_metrics(targets, predictions, tpr_thres=0.95, targets_threshold=None):
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
      "auc-roc": sklearn.metrics.roc_auc_score(targets, predictions),
      "auc-pr": sklearn.metrics.average_precision_score(targets, predictions),
      "fprn": fprn,
  }


def compute_mean_and_cov(embeds, labels):
  """Computes class-specific means and shared covariance matrix of given embedding.

  The computation follows Eq (1) in [1].

  Args:
    embeds: an np.array of size [n_train_sample, n_dim], where n_train_sample is
      the sample size of training set, n_dim is the dimension of the embedding.
    labels: an np.array of size [n_train_sample, ]

  Returns:
    mean_list: a list of len n_class, and the i-th element is an np.array of
    size [n_dim, ] corresponding to the mean of the fitted Guassian distribution
    for the i-th class.
    cov: the shared covariance mmatrix of the size [n_dim, n_dim].
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
    embeds: an np.array of size [n_test_sample, n_dim], where n_test_sample is
      the sample size of the test set, n_dim is the size of the embeddings.
    mean_list: a list of len n_class, and the i-th element is an np.array of
      size [n_dim, ] corresponding to the mean of the fitted Guassian
      distribution for the i-th class.
    cov: the shared covariance mmatrix of the size [n_dim, n_dim].
    epsilon: the small value added to the diagonal of the covariance matrix to
      avoid singularity.

  Returns:
    out: an np.array of size [n_test_sample, n_class] where the [i, j] element
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
