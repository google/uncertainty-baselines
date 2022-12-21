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

"""Plot utils."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
from collections import defaultdict
import os
import os.path as osp
from typing import Any, Callable, Dict, List, NamedTuple, Tuple, Union
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import seaborn as sns
from sklearn.metrics import roc_curve
import tensorflow as tf

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
    ('fsvi', False): 'FSVI',
    ('fsvi', True): 'FSVI Ensemble'
}


def get_colors_and_linestyle(model_name):
  cmap = plt.get_cmap('tab20c')
  blue = plt.get_cmap('Set1')(1)

  color_list = {
      'MAP': cmap(8),
      'Deep Ensemble': cmap(8),
      'MFVI': 'mediumorchid',
      'MFVI Ensemble': 'mediumorchid',
      'FSVI': blue,
      'FSVI Ensemble': blue,
      'Radial BNN': 'indianred',
      'Radial Ensemble': 'indianred',
      'MC Dropout': 'darkgoldenrod',
      'MC Dropout Ensemble': 'darkgoldenrod',
      'Rank-1 BNN': 'dimgrey',
      'Rank-1 Ensemble': 'dimgrey',
  }
  linestyle_list = {
      'MAP': '-',
      'Deep Ensemble': ':',
      'MFVI': '-',
      'MFVI Ensemble': ':',
      'FSVI': '-',
      'FSVI Ensemble': ':',
      'Radial BNN': '-',
      'Radial Ensemble': ':',
      'MC Dropout': '-',
      'MC Dropout Ensemble': ':',
      'Rank-1 BNN': '-',
      'Rank-1 Ensemble': ':'
  }
  return color_list[model_name], linestyle_list[model_name]


RETENTION_ARR_TO_FULL_NAME = {
    'retention_accuracy_arr': 'Accuracy',
    'retention_nll_arr': 'NLL',
    'retention_auroc_arr': 'AUROC',
    'retention_auprc_arr': 'AUPRC'
}

ROC_TYPE_TO_FULL_NAME = {
    'drd': 'Diabetic Retinopathy Detection',
    'ood_detection': 'OOD Detection'
}


def set_matplotlib_constants():
  plt.rcParams['figure.figsize'] = (6, 4)
  plt.rcParams['axes.titlesize'] = 12
  plt.rcParams['font.size'] = 12
  plt.rcParams['lines.linewidth'] = 2.0
  plt.rcParams['lines.markersize'] = 8
  plt.rcParams['grid.linestyle'] = '--'
  plt.rcParams['grid.linewidth'] = 1.0
  plt.rcParams['legend.fontsize'] = 10
  plt.rcParams['legend.facecolor'] = 'white'
  plt.rcParams['axes.labelsize'] = 24
  plt.rcParams['xtick.labelsize'] = 16
  plt.rcParams['ytick.labelsize'] = 16
  plt.rcParams['xtick.direction'] = 'in'
  plt.rcParams['ytick.direction'] = 'in'
  plt.rcParams['xtick.major.pad'] = 8
  plt.rcParams['ytick.major.pad'] = 8
  plt.rcParams['axes.grid'] = True
  plt.rcParams['font.family'] = 'Times New Roman'


def get_model_name(model_key: Tuple):
  # TODO(nband): generalize.
  model_type, k, _, _, _ = model_key
  if k > 1:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, True)]} (K = {k})'
  else:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, False)]}'


def plot_roc_curves(distribution_shift_name, dataset_to_model_results,
                    plot_dir: str):
  """Plot ROC curves for a given distributional shift task and

  corresponding results.

  Args:
    distribution_shift_name: str, distribution shift used to compute results.
    dataset_to_model_results: Dict, results for each evaluation dataset.
    plot_dir: str, where to store plots.
  """
  set_matplotlib_constants()
  datasets = list(sorted(list(dataset_to_model_results.keys())))

  roc_types = {
      'drd': {
          'y_true': 'y_true',
          'y_pred': 'y_pred'
      },
      'ood_detection': {
          'y_true': 'is_ood',
          'y_pred': 'y_pred_entropy'
      },
  }

  thresholds = np.arange(0, 1.02, 0.02)

  for dataset in datasets:
    dataset_results = dataset_to_model_results[dataset]
    for tuning_domain in ['indomain', 'joint']:
      for roc_type, roc_dict in roc_types.items():
        # Need the joint datasets, which have an `is_ood` field
        if roc_type == 'ood_detection' and 'joint' not in dataset:
          continue

        y_true_key = roc_dict['y_true']
        y_pred_key = roc_dict['y_pred']

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)

        # The actual DRD predictions are quite far from the diagonal,
        # whereas OOD detection is close. Set frame accordingly.
        if roc_type == 'ood_detection':
          ax.plot([0, 1], [0, 1], linestyle=':', color='black')
          ax.set_ylim([-0.05, 1.05])
          ax.set_xlim([-0.03, 1.03])
        elif roc_type == 'drd':
          ax.plot(
              0.2,
              0.85,
              marker='o',
              color='limegreen',
              markersize=6,
              label='NHS Recommendation',
              linestyle='None')
          ax.set_ylim([0.45, 1.05])
          ax.set_xlim([-0.03, 0.93])

        roc_name = ROC_TYPE_TO_FULL_NAME[roc_type]

        plot_name = (f'roc-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{roc_type}')

        model_names = []
        for ((mt, k, is_d, key_tuning_domain, n_mc),
             model_dict) in dataset_results.items():
          if tuning_domain != key_tuning_domain:
            continue

          model_name = get_model_name((mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          print(model_name)
          print(model_dict.keys())
          print(dataset)
          print(tuning_domain)

          y_true = np.array(model_dict[y_true_key])
          y_pred = np.array(model_dict[y_pred_key])

          tpr_values = np.zeros(shape=(thresholds.shape[0], y_true.shape[0]))

          for seed_idx in range(y_true.shape[0]):
            y_true_seed = y_true[seed_idx, :]
            y_pred_seed = y_pred[seed_idx, :]
            fpr, tpr, _ = roc_curve(y_true=y_true_seed, y_score=y_pred_seed)

            for j in range(thresholds.shape[0]):
              fpr_idx = np.abs(fpr - thresholds[j]).argmin()
              tpr_values[j, seed_idx] = tpr[fpr_idx]

          tpr_value_mean = tpr_values.mean(1)
          tpr_value_ste = sem(tpr_values, axis=1)

          color, linestyle = get_colors_and_linestyle(
              MODEL_TYPE_TO_FULL_NAME[(mt, k > 1)])

          # Visualize mean with standard error
          ax.plot(
              thresholds,
              tpr_value_mean,
              color=color,
              label=model_name,
              linestyle=linestyle)
          ax.fill_between(
              thresholds,
              tpr_value_mean - tpr_value_ste,
              tpr_value_mean + tpr_value_ste,
              color=color,
              alpha=0.25,
          )

          # ax.legend(facecolor="white")
          ax.set_xlabel('False Positive Rate')
          ax.set_ylabel('True Positive Rate')
          ax.plot([0, 1], [0, 1], ls='--', c='.3', lw=0.75)
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(plot_dir, f'{plot_name}.pdf')
          fig.savefig(metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(
              f'Saved ROC plot for distribution shift {distribution_shift_name},'
              f'dataset {dataset}, tuning domain {tuning_domain}, '
              f'roc_type {roc_name}, models {model_names} to '
              f'{metric_plot_path}')

        print(plot_name)
        # plt.show()


def plot_retention_curves(distribution_shift_name,
                          dataset_to_model_results,
                          plot_dir: str,
                          no_oracle=True,
                          cutoff_perc=0.99):
  """Plot retention curves for a given distributional shift task and

  corresponding results.

  Args:
    distribution_shift_name: str, distribution shift used to compute results.
    dataset_to_model_results: Dict, results for each evaluation dataset.
    plot_dir: str, where to store plots.
    no_oracle: bool, if True, converts retention array that is computed with an
      Oracle Collaborative metric to remove the contribution of the oracle.
    cutoff_perc: float, specifies at which proportion the curves should be
      cutoff.
  """
  set_matplotlib_constants()
  retention_types = [
      'retention_accuracy_arr', 'retention_nll_arr', 'retention_auroc_arr',
      'retention_auprc_arr'
  ]

  datasets = list(sorted(list(dataset_to_model_results.keys())))

  for dataset in datasets:
    dataset_results = dataset_to_model_results[dataset]
    for tuning_domain in ['indomain', 'joint']:
      for retention_type in retention_types:
        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)

        retention_name = RETENTION_ARR_TO_FULL_NAME[retention_type]
        oracle_str = 'no_oracle' if no_oracle else 'oracle'
        plot_name = (f'retention-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{retention_name}-{oracle_str}')

        model_names = []
        for ((mt, k, is_d, key_tuning_domain, n_mc),
             model_dict) in dataset_results.items():
          if tuning_domain != key_tuning_domain:
            continue
          model_name = get_model_name((mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          # Subsample the array to ~500 points
          retention_arr = np.array(model_dict[retention_type])

          if no_oracle:
            prop_expert = np.arange(
                retention_arr.shape[1]) / retention_arr.shape[1]
            prop_model = 1 - prop_expert
            retention_arr = (retention_arr - prop_expert) / prop_model

          if retention_arr.shape[1] > 500:
            subsample_factor = max(2, int(retention_arr.shape[1] / 500))
            retention_arr = retention_arr[:, ::subsample_factor]

          retain_percs = np.arange(
              retention_arr.shape[1]) / retention_arr.shape[1]
          n_seeds = retention_arr.shape[0]
          mean = np.mean(retention_arr, axis=0)
          std_err = np.std(retention_arr, axis=0) / np.sqrt(n_seeds)

          if cutoff_perc is not None and 'accuracy' in retention_type:
            retain_percs = retain_percs[:-100]
            mean = mean[:-100]
            std_err = std_err[:-100]

          if 'retention_nll_arr' == retention_type:
            cutoff_index = int(retain_percs.shape[0] * 0.95)
            retain_percs = retain_percs[:cutoff_index]
            mean = mean[:cutoff_index]
            std_err = mean[:cutoff_index]

          color, linestyle = get_colors_and_linestyle(
              MODEL_TYPE_TO_FULL_NAME[(mt, k > 1)])

          # Visualize mean with standard error
          ax.plot(
              retain_percs,
              mean,
              label=model_name,
              color=color,
              linestyle=linestyle)
          ax.fill_between(
              retain_percs,
              mean - std_err,
              mean + std_err,
              color=color,
              alpha=0.25)
          ax.set(
              xlabel='Proportion of Cases Referred to Expert',
              ylabel=retention_name)
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(plot_dir, f'{plot_name}.pdf')
          fig.savefig(metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(f'Saved retention plot for distribution shift '
                       f'{distribution_shift_name},'
                       f'dataset {dataset}, '
                       f'tuning domain {tuning_domain}, '
                       f'metric {retention_type}, '
                       f'models {model_names}, '
                       f'oracle setting {oracle_str} to {metric_plot_path}.')

        print(plot_name)
        # plt.show()


N_SAMPLES_PER_CLASS = {
    'aptos': [1805, 370, 999, 193, 295],
    'eyepacs_train': [25810, 2443, 5292, 873, 708],
    'eyepacs_validation': [8130, 720, 1579, 237, 240],
    'eyepacs_test': [31403, 3042, 6282, 977, 966],
}


def plot_predictive_entropy_histogram(y_uncertainty: np.ndarray,
                                      is_ood: np.ndarray,
                                      uncertainty_type: str,
                                      in_dist_str='In-Domain',
                                      ood_str='OOD',
                                      title='',
                                      plt_path='.'):
  """Should pass all uncertainty scores for in-domain and OOD points.

  :param y_uncertainty:
  :param is_ood:
  :param in_dist_str:
  :param ood_str:
  :param title:
  :param plt_path:
  :return:
  """
  del title
  assert uncertainty_type in {'entropy', 'variance'}
  short_to_full_uncertainty_type = {
      'entropy': 'Predictive Entropy',
      'variance': 'Predictive Variance'
  }
  full_uncertainty_type = short_to_full_uncertainty_type[uncertainty_type]
  sns.displot(
      {
          full_uncertainty_type:
              y_uncertainty,
          'Dataset':
              [ood_str if ood_label else in_dist_str for ood_label in is_ood]
      },
      x=full_uncertainty_type,
      hue='Dataset')
  plt.savefig(osp.join(plt_path, f'predictive_{uncertainty_type}.svg'))


def plot_pie_chart(x: Dict[str, float], ax=None, title=''):
  if not ax:
    plt.figure()
    ax = plt.gca()
  labels = sorted(x.keys())
  vals = [x[l] for l in labels]
  ax.pie(
      vals,
      labels=None,
      normalize=True,
      shadow=False,
      autopct='%1.1f%%',
      startangle=-90,
      pctdistance=0.7,
      labeldistance=1.1)
  ax.set_title(title)
  plt.legend(labels=labels)
  plt.tight_layout()


def plot_pie_charts_all_dataset():
  for name, array in N_SAMPLES_PER_CLASS.items():
    plot_pie_chart({i: v for i, v in enumerate(array)}, title=name)


def compute_data_for_ece_plot(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              n_windows=100) -> Dict[str, np.ndarray]:
  probs = binary_converter(y_pred).flatten()
  labels = one_hot_encode(y_true, 2).flatten()

  probs_labels = np.stack([probs, labels]).T
  probs_labels = probs_labels[np.argsort(probs)]

  window_len = int(len(probs_labels) / n_windows)
  confidences = []
  accuracies = []
  # distances = []
  for i in range(len(probs_labels) - window_len):
    # distances.append(
    #   (probs_labels[i + window_len, 0] -
    #   probs_labels[i, 0]) / float(window_len))
    mean_confidences = mean_default_zero(probs_labels[i:i + window_len, 0])
    confidences.append(mean_confidences)
    class_accuracies = mean_default_zero(probs_labels[i:i + window_len, 1])
    accuracies.append(class_accuracies)
  return {
      'accuracies': np.array(accuracies),
      'confidences': np.array(confidences),
  }


def plot_ece(confidences, accuracies, label='', ax=None, **kwargs):
  """All inputs are 1D arrays
  """
  if not ax:
    plt.figure()
    ax = plt.gca()
  # ax.figure.set_size_inches(6, 4)
  ax.plot(confidences, accuracies, label=label, **kwargs)
  xbins = [i / 10. for i in range(11)]
  ax.plot(xbins, xbins, linestyle=':', color='black')
  ax.set_xlabel('Model Confidence')
  ax.set_ylabel('Model Accuracy')
  # ax.set_title(
  #   f"Reliability Diagram Trained on {train_name}, Evaluated on {ood_name}")
  # ax.set_title(f"Reliability Diagram")
  ax.legend(loc=4, facecolor='white')
  return ax


class ModelKey(NamedTuple):
  model_type: str
  k: int  # number of members in the ensemble
  is_deterministic: bool
  tuning_domain: str
  num_mc_samples: int  # format f'mc{num_mc_samples}/'


# RESULT_DICT contains keys [‘y_pred’, ‘y_true’, ‘y_aleatoric_uncert’,
#   ‘y_epistemic_uncert’, ‘y_total_uncert’, ‘is_ood']
RESULT_DICT = Dict[str, List[np.ndarray]]  # pylint: disable=invalid-name
MODEL_RESULT_DICT = Dict[Tuple, RESULT_DICT]  # pylint: disable=invalid-name
DATA_RESULT_DICT = Dict[str, MODEL_RESULT_DICT]  # pylint: disable=invalid-name


def grid_plot_wrapper(
    fn,
    n_plots,
    ncols,
    nrows=None,
    get_args: Union[List[List], Callable[[int], List]] = None,
    get_kwargs: Union[List[Dict], Callable[[int], Dict]] = None,
    axes: List[plt.axes] = None,
):
  if not get_args:
    get_args = lambda i: []
  if not get_kwargs:
    get_kwargs = lambda i: {}
  if isinstance(get_args, list):
    get_args = lambda i: get_args[i]
  if isinstance(get_kwargs, list):
    get_kwargs = lambda i: get_kwargs[i]
  if not nrows:
    nrows = int(np.ceil(n_plots / ncols))
  if axes is None:
    _, axes = plt.subplots(nrows=nrows, ncols=ncols)
  flatten_axes = [a for axs in axes for a in axs]  # pylint: disable=g-complex-comprehension
  for i, ax in enumerate(flatten_axes):
    fn(*get_args(i), ax=ax, **get_kwargs(i))
    if i + 1 == n_plots:
      break
  plt.tight_layout()
  return axes


def parse_model_key(model_key: Tuple):
  model_type, k, is_deterministic, tuning_domain, num_mc_samples = model_key
  return ModelKey(
      model_type=model_type,
      k=k,
      is_deterministic=is_deterministic,
      tuning_domain=tuning_domain,
      num_mc_samples=int(num_mc_samples))


def plot_ece_all_methods(models_results_dict: MODEL_RESULT_DICT):
  return grid_plot_wrapper(
      fn=plot_ece_one_method,
      n_plots=len(models_results_dict),
      ncols=3,
      get_args=[[parse_model_key(k), v] for k, v in models_results_dict.items()
               ],
  )


def aggregate_result_dict(result_dict: RESULT_DICT,
                          agg_fns: List[Callable],
                          keys: List = None):
  if keys:
    result_dict = {k: v for k, v in result_dict.items() if k in keys}
  agg_dict = {
      k: [fn(np.array(arrays)) for fn in agg_fns
         ] for k, arrays in result_dict.items()
  }
  return agg_dict


def plot_ece_one_method(model_key: ModelKey,
                        result_dict: RESULT_DICT,
                        ax=None,
                        n_windows=100,
                        plt_kwargs=None):
  if not ax:
    plt.figure()
    ax = plt.gca()
  # compute mean and std ece curves for all seeds
  ece_data = [
      compute_data_for_ece_plot(
          y_true=y_true, y_pred=y_pred, n_windows=n_windows)
      for y_true, y_pred in zip(result_dict['y_true'], result_dict['y_pred'])
  ]
  arrays = {
      key: np.stack([d[key] for d in ece_data]) for key in ece_data[0].keys()
  }
  mean = {k: np.mean(array, axis=0) for k, array in arrays.items()}
  # std = {k: np.std(array, axis=0) for k, array in arrays.items()}
  # plot them on an axes
  if plt_kwargs is None:
    plt_kwargs = {}
  plot_ece(ax=ax, label=model_key.model_type, **mean, **plt_kwargs)
  return ax


def mean_default_zero(inputs):
  """Be able to take the mean of an empty array without hitting NANs."""
  # pylint disable necessary for numpy and pandas
  if len(inputs) == 0:  # pylint: disable=g-explicit-length-test
    return 0
  else:
    return np.mean(inputs)


def one_hot_encode(labels, num_classes=None):
  """One hot encoder for turning a vector of labels into a OHE matrix."""
  if num_classes is None:
    num_classes = len(np.unique(labels))
  return np.eye(num_classes)[labels]


def binary_converter(probs):
  """Converts a binary probability vector into a matrix."""
  return np.array([[1 - p, p] for p in probs])


def verify_probability_shapes(probs):
  """Verify shapes of probs vectors and possibly stack 1D probs into 2D."""
  if probs.ndim == 2:
    num_classes = probs.shape[1]
    if num_classes == 1:
      probs = probs[:, 0]
      probs = binary_converter(probs)
      num_classes = 2
  elif probs.ndim == 1:
    # Cover binary case
    probs = binary_converter(probs)
    num_classes = 2
  else:
    raise ValueError('Probs must have 1 or 2 dimensions.')
  return probs, num_classes


def transpose_dict(d: Dict[Any, Dict[Any, Any]]):
  new_d = defaultdict(dict)
  for k1, v1 in d.items():
    for k2, v2 in v1.items():
      assert k1 not in new_d[k2]
      new_d[k2][k1] = v2
  return new_d


def plot_predictive_entropy_histogram_one_method(model_key: ModelKey,
                                                 domain_result_dict: Dict,
                                                 ax=None):
  if not ax:
    plt.figure()
    ax = plt.gca()
  model_type = model_key.model_type

  # draw histogram for ID
  mean_y_total_uncert = np.mean(
      domain_result_dict['in_domain_test']['y_total_uncert'], axis=0)
  plot_predictive_entropy_line_histogram(
      mean_y_total_uncert,
      ax=ax,
      color='red',
      title=f'Predictive Entropy {model_type}')

  # draw histogram for OOD
  mean_y_total_uncert = np.mean(
      domain_result_dict['ood_test']['y_total_uncert'], axis=0)
  plot_predictive_entropy_line_histogram(
      mean_y_total_uncert,
      ax=ax,
      color='blue',
      title=f'Predictive Entropy {model_type}')
  return ax


def line_hist(array, thresholds, ax=None, **kwargs):
  if not ax:
    plt.figure()
    ax = plt.gca()
  counts = []
  for i in range(len(thresholds) - 1):
    count = np.sum(
        np.logical_and(array >= thresholds[i], array < thresholds[i + 1]))
    counts.append(count)
  ax.plot(thresholds[:-1], np.array(counts), **kwargs)
  return ax


def hide_top_and_right_axis(ax):
  ax.spines['right'].set_visible(False)
  ax.spines['top'].set_visible(False)


def use_serif_font():
  plt.rc('font', family='serif')


def plot_predictive_entropy_line_histogram(y_total_uncert,
                                           start=-0.01,
                                           end=2.5,
                                           step_size=0.1,
                                           ax=None,
                                           color='blue',
                                           title='',
                                           **kwargs):
  if not ax:
    plt.figure()
    ax = plt.gca()
  thresholds = np.arange(start, end, step_size)
  line_hist(y_total_uncert, thresholds, ax=ax, color=color, **kwargs)
  ax.set_xlabel('Entropy (nats)')
  ax.set_ylabel('# of Examples')
  ax.set_title(title)
  # ax.legend(loc='upper left', facecolor='white', fontsize='small')
  return ax


def plot_total_versus_aleatoric_uncertainty_all_methods(
    data_result_dict: DATA_RESULT_DICT):

  def one_iter(domain: str, method_key: str):
    result_dict = data_result_dict[domain][method_key]
    plot_total_versus_aleatoric_uncertainty(
        y_true=result_dict['y_true'],
        y_pred=result_dict['y_pred'],
        y_total_uncert=result_dict['y_total_uncert'],
    )

  combinations = [[domain, model_key]  # pylint: disable=g-complex-comprehension
                  for domain, model_dict in data_result_dict.items()
                  for model_key, _ in model_dict.items()]

  return grid_plot_wrapper(
      fn=one_iter,
      n_plots=len(combinations),
      ncols=3,
      get_args=combinations,
  )


def plot_total_versus_aleatoric_uncertainty(y_true,
                                            y_pred,
                                            y_total_uncert,
                                            threshold=0.5,
                                            alpha=0.3):
  label_pred = np.array(y_pred > threshold, dtype=int)
  correct_index = np.nonzero(y_true == label_pred)[0]
  wrong_index = np.nonzero(y_true != label_pred)[0]

  alea_total_array = np.stack([y_pred, y_total_uncert]).T

  _, axes = plt.subplots(1, 2, sharey=True)
  # axes[0].scatter(
  #   alea_total_array[correct_index, 0],
  #   alea_total_array[correct_index, 1], alpha=alpha)
  # axes[1].scatter(
  #   alea_total_array[wrong_index, 0],
  #   alea_total_array[wrong_index, 1], alpha=alpha)
  sns.kdeplot(
      x=alea_total_array[correct_index, 0],
      y=alea_total_array[correct_index, 1],
      fill=True,
      ax=axes[0],
      color='green',
      alpha=alpha)
  sns.kdeplot(
      x=alea_total_array[wrong_index, 0],
      y=alea_total_array[wrong_index, 1],
      fill=True,
      ax=axes[1],
      color='red',
      alpha=alpha)


def read_eval_folder(path, allow_pickle=True):
  """e.g.

  path = (
    "gs://drd-fsvi-severity-results/2021-08-23-23-06-42/"
    "ood_validation/eval_results_80")
  """
  filenames = tf.io.gfile.listdir(path)
  d = {}
  for fn in filenames:
    p = os.path.join(path, fn)
    with tf.io.gfile.GFile(p, 'rb') as f:
      d[fn[:-4]] = np.load(f, allow_pickle=allow_pickle)
  return d


def plot_predictive_entropy_histogram_all_methods(
    data_result_dict: DATA_RESULT_DICT):
  transposed = transpose_dict(data_result_dict)
  return grid_plot_wrapper(
      fn=plot_predictive_entropy_histogram_one_method,
      ncols=3,
      n_plots=len(transposed),
      get_kwargs=[],
  )
