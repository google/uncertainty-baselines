# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Calculate ood metrics for segmentation tasks."""
from typing import Any, Optional, Dict

import jax
import jax.numpy as jnp
import numpy as np
import sklearn.metrics


SUPPORTED_OOD_METHODS = ('msp', 'entropy', 'mlogit')


def compute_ood_metrics(targets,
                        predictions,
                        tpr_thres=0.95,
                        targets_threshold=None):
  """Computes Area Under the ROC and PR curves and FPRN.

  ROC - Receiver Operating Characteristic
  PR  - Precision and Recall
  FPRN - False positive rate at which true positive rate is N.

  From uncertainty_baselines/ood_utils
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
      'auroc':
          float(sklearn.metrics.roc_auc_score(targets, predictions)),
      'auprc':
          float(sklearn.metrics.average_precision_score(targets, predictions)),
      'fprn':
          float(fprn),
  }


def preprocess_outlier_np(outlier):
  outlier_ = outlier.copy()
  outlier_[outlier_ == 255] = 0
  return outlier_


def preprocess_outlier(outlier):
  outlier_ = outlier.copy()
  outlier_ = outlier_.at[outlier_ == 255].set(0)
  return outlier_


def get_ood_score(
    logits: jnp.ndarray,
    method_name: str = 'msp',
    num_top_k: int = 5,
    ) -> Dict[str, Any]:
  """Get OOD score."""

  if method_name == 'msp':
    probs = jax.nn.softmax(logits, -1)
    max_probs = jnp.max(probs, -1)
    ood_score = 1 - max_probs
  elif method_name == 'entropy':
    probs = jax.nn.softmax(logits, -1)
    entropy = -jnp.sum(probs * jnp.log(probs), axis=-1)
    ood_score = entropy
  elif method_name == 'mlogit':
    max_logits = jnp.max(logits, -1)
    ood_score = 1 - max_logits
  elif method_name == 'sum_topklogit':
    ood_score = jax.lax.top_k(logits, num_top_k)[0].sum(-1)
  elif method_name == '1-sum_topklogit':
    ood_score = 1 - jax.lax.top_k(logits, num_top_k)[0].sum(-1)

  else:
    raise NotImplementedError(
        f'Missing method {method_name} to calculate OOD score.')
  return ood_score


def get_ood_metrics(
    logits: jnp.ndarray,
    ood_mask: jnp.ndarray,
    method_name: str = 'msp',
    weights: Optional[jnp.ndarray] = None,
    num_top_k: int = 5,
    ) -> Dict[str, Any]:
  """Get OOD metrics."""

  if method_name not in SUPPORTED_OOD_METHODS:
    raise NotImplementedError(
        'Only %s are supported for OOD evaluation! Got metric_name=%s!' %
        (','.join(SUPPORTED_OOD_METHODS), method_name))

  y_true = np.asarray(ood_mask)
  y_true = preprocess_outlier_np(y_true)

  probs = jax.nn.softmax(logits, -1)
  max_probs = jnp.max(probs, -1)
  max_logits = jnp.max(logits, -1)
  entropy = -jnp.sum(probs * jnp.log(probs), axis=-1)

  if method_name == 'msp':
    ood_score = np.asarray(1 - max_probs)
  elif method_name == 'entropy':
    ood_score = np.asarray(entropy)
  elif method_name == 'mlogit':
    ood_score = np.asarray(1 - max_logits)
  elif method_name == 'sum_topklogit':
    logits_top_k = np.partition(logits, -num_top_k)[..., -num_top_k:]
    ood_score = logits_top_k.sum(-1)
  elif method_name == '1-sum_topklogit':
    logits_top_k = np.partition(logits, -num_top_k)[..., -num_top_k:]
    ood_score = 1 - logits_top_k.sum(-1)

  else:
    raise NotImplementedError(f'Did not implement{method_name}')

  # the weights per entry are 1 if it should be included during computation
  # and 0 otherwise.
  y_true_masked = y_true[weights == 1]
  ood_score_masked = ood_score[weights == 1]

  metrics = compute_ood_metrics(y_true_masked.flatten(), ood_score_masked.flatten())

  return metrics
