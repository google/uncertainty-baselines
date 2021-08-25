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
from itertools import cycle

import matplotlib.pyplot as plt
import seaborn as sns
import os.path as osp
from typing import Tuple


# Format: model_type, is_ensemble
MODEL_TYPE_TO_FULL_NAME = {
  ('deterministic', False): 'MAP',
  ('deterministic', True): 'Deep Ensemble',
  ('dropout', False): 'MC Dropout',
  ('dropout', True): 'MC Dropout Ensemble',
  ('rank1', False): 'Rank-1 BNN',
  ('rank1', True): 'Rank-1 Ensemble',
  ('vi', False): 'MFVI',
  ('vi', True): 'MFVI Ensemble',
  ('radial', False): 'Radial BNN',
  ('radial', True): 'Radial Ensemble',
  ('fsvi', False): 'Function Space VI',
  ('fsvi', True): 'Function Space VI Ensemble'
}


def get_colors_and_linestyle(model_name):
  cmap = plt.get_cmap("tab20c")
  blue = plt.get_cmap("Set1")(1)
  red = plt.get_cmap("Set1")(0)
  cmap_2 = plt.get_cmap("tab10")
  cmap_3 = plt.get_cmap("Paired")
  cmap_4 = plt.get_cmap("tab20b")
  gap = 50

  color_list = {
    "MAP": cmap(9),
    "Deep Ensemble": cmap(8),
    "MFVI": red,
    "MFVI Ensemble": cmap(5),
    "FSVI": blue,
    "FSVI Ensemble": cmap(1),
    "Radial BNN": plt.get_cmap("Reds")(2 * gap),
    "Radial Ensemble": plt.get_cmap("Reds")(3 * gap),
    "MC Dropout": cmap_2(5),
    "MC Dropout Ensemble": cmap_4(9),
    "Rank-1 BNN": cmap_4(17),
    "Rank-1 Ensemble": cmap_4(16),
  }
  linestyle_list = {
    "MAP": "-",
    "Deep Ensemble": "--",
    "MFVI": "-",
    "MFVI Ensemble": "--",
    "FSVI": "-",
    "FSVI Ensemble": "--",
    "Radial BNN": "-",
    "Radial Ensemble": "--",
    "MC Dropout": "-",
    "MC Dropout Ensemble": "--",
    "Rank-1 BNN": "-",
    "Rank-1 Ensemble": "--"
  }
  return color_list[model_name], linestyle_list[model_name]


RETENTION_ARR_TO_FULL_NAME = {
  'retention_accuracy_arr': 'Accuracy',
  'retention_nll_arr': 'NLL',
  'retention_auroc_arr': 'AUROC',
  'retention_auprc_arr': 'AUPRC'
}

def set_matplotlib_constants():
  plt.rcParams["figure.figsize"] = (6, 4)
  plt.rcParams["axes.titlesize"] = 12
  plt.rcParams["font.size"] = 12
  plt.rcParams["lines.linewidth"] = 2.0
  plt.rcParams["lines.markersize"] = 8
  plt.rcParams["grid.linestyle"] = "--"
  plt.rcParams["grid.linewidth"] = 1.0
  plt.rcParams["legend.fontsize"] = 10
  plt.rcParams["legend.facecolor"] = "white"
  plt.rcParams["axes.labelsize"] = 16
  plt.rcParams["xtick.labelsize"] = 12
  plt.rcParams["ytick.labelsize"] = 12
  plt.rcParams["xtick.direction"] = "in"
  plt.rcParams["ytick.direction"] = "in"
  plt.rcParams['xtick.major.pad'] = 8
  plt.rcParams['ytick.major.pad'] = 8
  plt.rcParams['axes.grid'] = True
  plt.rcParams["font.family"] = "Times New Roman"
  # plt.rcParams['text.usetex'] = True
  # plt.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']

def get_model_name(model_key: Tuple):
  """TODO: generalize"""
  model_type, k, _, _, _ = model_key
  if k > 1:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, True)]} (K = {k})'
  else:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, False)]}'


def plot_retention_curves(
    distribution_shift_name, dataset_to_model_results, plot_dir: str,
    no_oracle=False):
  """Creates a deferred prediction plot for each metric in the results_df.

  For a particular model, aggregates metric values (e.g., accuracy) when retain
  proportion is identical.

  If retain proportion, train seed, and eval seed are identical for a
  particular model and metric, we consider only the most recent result.
  Args:
   results_df: `pd.DataFrame`, expects columns: {metric, retain_proportion,
     value, model_type, train_seed, eval_seed, run_datetime}.
   plot_dir: `str`, directory to which plots will be written
  """
  set_matplotlib_constants()
  # colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
  # line_styles = ['-', '--', ':', 'dashdot']
  retention_types = ['retention_accuracy_arr',
                     'retention_nll_arr',
                     'retention_auroc_arr',
                     'retention_auprc_arr']

  datasets = list(sorted(list(dataset_to_model_results.keys())))

  for dataset in datasets:
    dataset_results = dataset_to_model_results[dataset]
    for tuning_domain in ['indomain', 'joint']:
      for retention_type in retention_types:
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)
        ax.plot([0, 1], [0, 1], linestyle=":", color="black")

        retention_name = RETENTION_ARR_TO_FULL_NAME[retention_type]
        plot_name = (f'retention-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{retention_name}')

        model_names = []
        for i, (
            (mt, k, is_d, key_tuning_domain, n_mc),
                model_dict) in enumerate(dataset_results.items()):
          if tuning_domain != key_tuning_domain:
            continue

          model_name = get_model_name(
            (mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          # Subsample the array to ~500 points
          retention_arr = np.array(model_dict[retention_type])

          if no_oracle:
            prop_expert = np.arange(
              retention_arr.shape[1]) / retention_arr.shape[1]
            prop_model = 1 - prop_expert
            retention_arr = (retention_arr - prop_expert) / prop_model

          subsample_factor = max(2, int(retention_arr.shape[1] / 500))
          retention_arr = retention_arr[::subsample_factor]

          retain_percs = np.arange(
            retention_arr.shape[1]) / retention_arr.shape[1]
          n_seeds = retention_arr.shape[0]
          mean = np.mean(retention_arr, axis=0)
          std_err = np.std(retention_arr, axis=0) / np.sqrt(n_seeds)

          color, linestyle = get_colors_and_linestyle(
            MODEL_TYPE_TO_FULL_NAME[(mt, k > 1)])

          # Visualize mean with standard error
          ax.plot(
              retain_percs,
              mean,
              label=model_name,
              # color=colors[i % len(colors)],
              # linestyle=line_styles[i % len(line_styles)]
              color=color,
              linestyle=linestyle
            )
          ax.fill_between(
              retain_percs,
              mean - std_err,
              mean + std_err,
              # color=colors[i % len(colors)],
              color=color,
              alpha=0.25)
          ax.set(xlabel='Proportion of Cases Referred to Expert',
                 ylabel=retention_name)
          ax.legend()
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(
            plot_dir, f'{plot_name}.pdf')
          fig.savefig(
            metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(
            f'Saved plot for distribution shift {distribution_shift_name},'
            f'dataset {dataset}, tuning domain {tuning_domain}, '
            f'metric {retention_type}, models {model_names} to '
            f'{metric_plot_path}')


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


# def plot_roc_curve(fpr, tpr, roc_auc, title='', plt_path='.'):
#   """
#   Based on scikit-learn examples.
#   https://scikit-learn.org/stable/auto_examples/model_selection/
#   plot_roc.html#sphx-glr-auto-examples-model-selection-plot-roc-py
#   """
#   plt.figure()
#   lw = 2
#   plt.plot(fpr, tpr, color='darkorange',
#            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
#   plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#   plt.xlim([0.0, 1.0])
#   plt.ylim([0.0, 1.05])
#   plt.xlabel('False Positive Rate')
#   plt.ylabel('True Positive Rate')
#   plt.title(f'{title} ROC Curve')
#   plt.legend(loc="lower right")
#   plt.savefig(osp.join(plt_path, 'roc_curve.svg'))

  def plot_auroc_entropy_all(preds_ood_dict: dict,
                             preds_test_dict: dict,
                             model_list: list,
                             data_training: str,
                             data_ood: str,
                             save_folder: str
                             ):
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.20, bottom=0.20)
    ax.plot([0, 1], [0, 1], linestyle=":", color="black")

    thresholds = np.arange(0, 1.02, 0.02)
    for model in tqdm(model_list):
      tpr_values = np.zeros(
        shape=(thresholds.shape[0], preds_ood_dict[model].shape[0]))
      for i in range(preds_ood_dict[model].shape[0]):
        predicted_labels_ood = preds_ood_dict[model].mean(1)[i, :]
        predicted_labels_test = preds_test_dict[model].mean(1)[i, :]

        ood_size = predicted_labels_ood.shape[0]
        test_size = predicted_labels_test.shape[0]
        anomaly_targets = jnp.concatenate(
          (np.zeros(test_size), np.ones(ood_size)))

        entropy_test = -(
            predicted_labels_test * jnp.log(predicted_labels_test + eps)
        ).sum(1)
        entropy_ood = -(
            predicted_labels_ood * jnp.log(predicted_labels_ood + eps)
        ).sum(1)
        scores = jnp.concatenate((entropy_test, entropy_ood))
        fpr, tpr, _ = sklearn.metrics.roc_curve(anomaly_targets, scores)

        for j in range(thresholds.shape[0]):
          fpr_idx = np.abs(fpr - thresholds[j]).argmin()
          tpr_values[j, i] = tpr[fpr_idx]

      tpr_value_mean = tpr_values.mean(1)
      tpr_value_std = tpr_values.std(1)
      tpr_value_ste = scipy.stats.sem(tpr_values, axis=1)
      ax.plot(thresholds,
              tpr_value_mean,
              color=color_list[model],
              label=label_list[model],
              linestyle=linestyle_list[model])
      ax.fill_between(
        thresholds,
        tpr_value_mean - tpr_value_ste,
        tpr_value_mean + tpr_value_ste,
        color=color_list[model],
        alpha=alpha,
      )
    # ax.legend(facecolor="white")
    ax.set_xlabel(f"False Positive Rate")
    ax.set_ylabel(f"True Positive Rate")
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlim([-0.03, 1.03])
    # ax.set_title(f"Trained on {train_name}, Evaluated on {ood_name}")
    os.makedirs(f"{save_folder}", exist_ok=True)
    fig.savefig(
      f"{save_folder}/{data_training.lower()}"
      + "_"
      + f"{data_ood.lower()}_auroc_entropy.pdf",
    )
    return fig, ax