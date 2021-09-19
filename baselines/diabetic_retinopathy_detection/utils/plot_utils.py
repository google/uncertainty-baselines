import os
from collections import defaultdict
from typing import Tuple, Dict, List, Any, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
from absl import logging
from scipy.stats import sem
from sklearn.metrics import roc_curve

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
  cmap = plt.get_cmap("tab20c")
  blue = plt.get_cmap("Set1")(1)

  color_list = {
    "MAP": cmap(8),
    "Deep Ensemble": cmap(8),
    "MFVI": 'mediumorchid',
    "MFVI Ensemble": 'mediumorchid',
    "FSVI": blue,
    "FSVI Ensemble": blue,
    "Radial BNN": 'indianred',
    "Radial Ensemble": 'indianred',
    "MC Dropout": 'darkgoldenrod',
    "MC Dropout Ensemble": 'darkgoldenrod',
    "Rank-1 BNN": 'dimgrey',
    "Rank-1 Ensemble": 'dimgrey',
  }
  linestyle_list = {
    "MAP": "-",
    "Deep Ensemble": ":",
    "MFVI": "-",
    "MFVI Ensemble": ":",
    "FSVI": "-",
    "FSVI Ensemble": ":",
    "Radial BNN": "-",
    "Radial Ensemble": ":",
    "MC Dropout": "-",
    "MC Dropout Ensemble": ":",
    "Rank-1 BNN": "-",
    "Rank-1 Ensemble": ":"
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
  plt.rcParams["figure.figsize"] = (6, 4)
  plt.rcParams["axes.titlesize"] = 12
  plt.rcParams["font.size"] = 12
  plt.rcParams["lines.linewidth"] = 2.0
  plt.rcParams["lines.markersize"] = 8
  plt.rcParams["grid.linestyle"] = "--"
  plt.rcParams["grid.linewidth"] = 1.0
  plt.rcParams["legend.fontsize"] = 10
  plt.rcParams["legend.facecolor"] = "white"
  plt.rcParams["axes.labelsize"] = 24
  plt.rcParams["xtick.labelsize"] = 16
  plt.rcParams["ytick.labelsize"] = 16
  plt.rcParams["xtick.direction"] = "in"
  plt.rcParams["ytick.direction"] = "in"
  plt.rcParams['xtick.major.pad'] = 8
  plt.rcParams['ytick.major.pad'] = 8
  plt.rcParams['axes.grid'] = True
  plt.rcParams["font.family"] = "Times New Roman"


def get_model_name(model_key: Tuple):
  # TODO @nband: generalize.
  model_type, k, _, _, _ = model_key
  if k > 1:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, True)]} (K = {k})'
  else:
    return f'{MODEL_TYPE_TO_FULL_NAME[(model_type, False)]}'


def plot_roc_curves(
    distribution_shift_name, dataset_to_model_results, plot_dir: str
):
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
        if roc_type == 'ood_detection' and not 'joint' in dataset:
          continue

        y_true_key = roc_dict['y_true']
        y_pred_key = roc_dict['y_pred']

        fig, ax = plt.subplots()
        plt.subplots_adjust(left=0.20, bottom=0.20)

        # The actual DRD predictions are quite far from the diagonal,
        # whereas OOD detection is close. Set frame accordingly.
        if roc_type == 'ood_detection':
          ax.plot([0, 1], [0, 1], linestyle=":", color="black")
          ax.set_ylim([-0.05, 1.05])
          ax.set_xlim([-0.03, 1.03])
        elif roc_type == 'drd':
          ax.plot(0.2, 0.85, marker='o', color='limegreen',
                  markersize=6, label='NHS Recommendation',
                  linestyle="None")
          ax.set_ylim([0.45, 1.05])
          ax.set_xlim([-0.03, 0.93])

        roc_name = ROC_TYPE_TO_FULL_NAME[roc_type]

        plot_name = (f'roc-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{roc_type}')

        model_names = []
        for i, (
            (mt, k, is_d, key_tuning_domain, n_mc),
            model_dict) in enumerate(dataset_results.items()):
          if tuning_domain != key_tuning_domain:
            continue

          model_name = get_model_name(
            (mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          print(model_name)
          print(model_dict.keys())
          print(dataset)
          print(tuning_domain)

          y_true = np.array(model_dict[y_true_key])
          y_pred = np.array(model_dict[y_pred_key])

          tpr_values = np.zeros(
            shape=(thresholds.shape[0], y_true.shape[0]))

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
          ax.plot(thresholds,
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
          ax.set_xlabel(f"False Positive Rate")
          ax.set_ylabel(f"True Positive Rate")
          ax.plot([0, 1], [0, 1], ls="--", c=".3", lw=0.75)
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(
            plot_dir, f'{plot_name}.pdf')
          fig.savefig(
            metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(
            f'Saved ROC plot for distribution shift {distribution_shift_name},'
            f'dataset {dataset}, tuning domain {tuning_domain}, '
            f'roc_type {roc_name}, models {model_names} to '
            f'{metric_plot_path}')

        print(plot_name)
        # plt.show()

def plot_retention_curves(
    distribution_shift_name, dataset_to_model_results, plot_dir: str,
    no_oracle=True, cutoff_perc=0.99
):
  """Plot retention curves for a given distributional shift task and
  corresponding results.

  Args:
    distribution_shift_name: str, distribution shift used to compute results.
    dataset_to_model_results: Dict, results for each evaluation dataset.
    plot_dir: str, where to store plots.
    no_oracle: bool, if True, converts retention array that is computed with
      an Oracle Collaborative metric to remove the contribution of the oracle.
    cutoff_perc: float, specifies at which proportion the curves should be
      cutoff.
  """
  set_matplotlib_constants()
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

        retention_name = RETENTION_ARR_TO_FULL_NAME[retention_type]
        oracle_str = 'no_oracle' if no_oracle else 'oracle'
        plot_name = (f'retention-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{retention_name}-{oracle_str}')

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
            linestyle=linestyle
          )
          ax.fill_between(
            retain_percs,
            mean - std_err,
            mean + std_err,
            color=color,
            alpha=0.25)
          ax.set(xlabel='Proportion of Cases Referred to Expert',
                 ylabel=retention_name)
          fig.tight_layout()

        if isinstance(plot_dir, str):
          os.makedirs(plot_dir, exist_ok=True)
          metric_plot_path = os.path.join(
            plot_dir, f'{plot_name}.pdf')
          fig.savefig(
            metric_plot_path, transparent=True, dpi=300, format='pdf')
          logging.info(
            f'Saved retention plot for distribution shift '
            f'{distribution_shift_name},'
            f'dataset {dataset}, '
            f'tuning domain {tuning_domain}, '
            f'metric {retention_type}, '
            f'models {model_names}, '
            f'oracle setting {oracle_str} to {metric_plot_path}.')

        print(plot_name)
        # plt.show()


# TODO @nband: add predictive entropy histogram plots.
# def plot_predictive_entropy_histogram(
#     y_uncertainty: np.ndarray, is_ood: np.ndarray,
#     uncertainty_type: str, in_dist_str='In-Domain', ood_str='OOD',
#     title='', plt_path='.'):
#   """
#   Should pass all uncertainty scores for in-domain and OOD points.
#
#   :param y_uncertainty:
#   :param is_ood:
#   :param in_dist_str:
#   :param ood_str:
#   :param title:
#   :param plt_path:
#   :return:
#   """
#   assert uncertainty_type in {'entropy', 'variance'}
#   short_to_full_uncertainty_type = {
#     'entropy': 'Predictive Entropy',
#     'variance': 'Predictive Variance'
#   }
#   full_uncertainty_type = short_to_full_uncertainty_type[uncertainty_type]
#   sns.displot({
#       full_uncertainty_type: y_uncertainty,
#       'Dataset': [ood_str if ood_label else in_dist_str for ood_label in is_ood]
#     }, x=full_uncertainty_type, hue="Dataset")
#   plt.savefig(osp.join(plt_path, f'predictive_{uncertainty_type}.svg'))

# def plot_predictive_entropy_histogram_all_methods(data_result_dict: DATA_RESULT_DICT):
#   transposed = transpose_dict(data_result_dict)
#   return grid_plot_wrapper(
#     fn=plot_predictive_entropy_histogram_one_method,
#     ncols=3,
#     n_plots=len(transposed),
#     get_kwargs=[],
#   )

# RESULT_DICT contains keys [‘y_pred’, ‘y_true’, ‘y_aleatoric_uncert’,
#   ‘y_epistemic_uncert’, ‘y_total_uncert’, ‘is_ood']
RESULT_DICT = Dict[str, List[np.ndarray]]
MODEL_RESULT_DICT = Dict[Tuple, RESULT_DICT]
DATA_RESULT_DICT = Dict[str, MODEL_RESULT_DICT]


def grid_plot_wrapper(
    fn, n_plots, ncols, nrows=None,
    get_args: Union[List[List], Callable[[int], List]] = None,
    get_kwargs: Union[List[Dict], Callable[[int], Dict]] = None
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
  fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
  flatten_axes = [a for axs in axes for a in axs]
  for i, ax in enumerate(flatten_axes):
    fn(*get_args(i), ax=ax, **get_kwargs(i))
  return fig


def transpose_dict(d: Dict[Any, Dict[Any, Any]]):
  new_d = defaultdict(dict)
  for k1, v1 in d.items():
    for k2, v2 in v1.items():
      assert k1 not in new_d[k2]
      new_d[k2][k1] = v2
  return new_d
