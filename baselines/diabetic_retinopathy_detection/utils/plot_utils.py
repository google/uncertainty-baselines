import os
import time

import numpy as np
import pandas as pd
import robustness_metrics as rm
import tensorflow as tf
import torch
from absl import flags
from absl import logging
from sklearn.metrics import (log_loss, roc_auc_score, accuracy_score,
                             roc_curve, precision_recall_curve, auc)
import datetime
import utils  # local file import


import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp


def plot_predictive_entropy_histogram(
    y_uncertainty: np.ndarray, is_ood: np.ndarray,
    uncertainty_type: str, in_dist_str='In-Domain', ood_str='OOD',
    title='', plt_path='.'):
  """
  Should pass all uncertainty scores for in-domain and OOD points.

  :param y_uncertainty:
  :param is_ood:
  :param in_dist_str:
  :param ood_str:
  :param title:
  :param plt_path:
  :return:
  """
  assert uncertainty_type in {'entropy', 'variance'}
  short_to_full_uncertainty_type = {
    'entropy': 'Predictive Entropy',
    'variance': 'Predictive Variance'
  }
  full_uncertainty_type = short_to_full_uncertainty_type[uncertainty_type]
  sns.displot({
      full_uncertainty_type: y_uncertainty,
      'Dataset': [ood_str if ood_label else in_dist_str for ood_label in is_ood]
    }, x=full_uncertainty_type, hue="Dataset")
  plt.savefig(osp.join(plt_path, f'predictive_{uncertainty_type}.svg'))


def plot_roc_curve(fpr, tpr, roc_auc, title='', plt_path='.'):
  """
  Based on scikit-learn examples.
  https://scikit-learn.org/stable/auto_examples/model_selection/
  plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
  """
  plt.figure()
  lw = 2
  plt.plot(fpr, tpr, color='darkorange',
           lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
  plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
  plt.xlim([0.0, 1.0])
  plt.ylim([0.0, 1.05])
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'{title} ROC Curve')
  plt.legend(loc="lower right")
  plt.savefig(osp.join(plt_path, 'roc_curve.svg'))
