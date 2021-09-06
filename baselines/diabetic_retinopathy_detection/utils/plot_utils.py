import os
import os.path as osp
from collections import defaultdict
from typing import Tuple, Dict, List, Any, Union, Callable

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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


def plot_roc_curves(
    distribution_shift_name, dataset_to_model_results, plot_dir: str):
  """TODO
  :param distribution_shift_name:
  :param dataset_to_model_results: 
  :param plot_dir: 
  :return: 
  """"""
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
      'y_score': 'y_pred_entropy'
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

        roc_name = ROC_TYPE_TO_FULL_NAME[roc_type]

        plot_name = (f'retention-{distribution_shift_name}-{dataset}'
                     f'-{tuning_domain}-{roc_name}')

        model_names = []
        for i, (
            (mt, k, is_d, key_tuning_domain, n_mc),
                model_dict) in enumerate(dataset_results.items()):
          if tuning_domain != key_tuning_domain:
            continue

          model_name = get_model_name(
            (mt, k, is_d, key_tuning_domain, n_mc))
          model_names.append(model_name)

          y_true = np.array(model_dict[y_true_key])
          y_pred = np.array(model_dict[y_pred_key])

          tpr_values = np.zeros(
            shape=(thresholds.shape[0], y_true.shape[0]))

          # TODO: this should be the case, but our test data may be corrupted
          #   in some way from partial runs
          # assert y_true.shape[0] == y_pred.shape[0]

          for seed_idx in range(y_true.shape[0]):
            y_true_seed = y_true[seed_idx, :]
            y_pred_seed = y_pred[seed_idx, :]
            fpr, tpr, _ = roc_curve(y_true=y_true_seed, y_score=y_pred_seed)

            for j in range(thresholds.shape[0]):
              fpr_idx = np.abs(fpr - thresholds[j]).argmin()
              tpr_values[j, i] = tpr[fpr_idx]

          tpr_value_mean = tpr_values.mean(1)
          # tpr_value_std = tpr_values.std(1)
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
          ax.set_ylim([-0.05, 1.05])
          ax.set_xlim([-0.03, 1.03])
          ax.legend()
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


def plot_retention_curves(
    distribution_shift_name, dataset_to_model_results, plot_dir: str,
    no_oracle=True):
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
        # ax.plot([0, 1], [0, 1], linestyle=":", color="black")

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

          if retention_arr.shape[1] > 3000:
            subsample_factor = max(2, int(retention_arr.shape[1] / 3000))
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
            f'Saved retention plot for distribution shift {distribution_shift_name},'
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


RESULT_DICT = Dict[str, List[np.ndarray]]
# RESULT_DICT contains keys [‘y_pred’, ‘y_true’, ‘y_aleatoric_uncert’,
#   ‘y_epistemic_uncert’, ‘y_total_uncert’, ‘is_ood']
MODEL_RESULT_DICT = Dict[Tuple, RESULT_DICT]
DATA_RESULT_DICT = Dict[str, MODEL_RESULT_DICT]


def grid_plot_wrapper(fn,
                      n_plots,
                      ncols,
                      nrows=None,
                      get_args: Union[List[List], Callable[[int], List]]=None,
                      get_kwargs: Union[List[Dict], Callable[[int], Dict]]=None):
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

def plot_predictive_entropy_histogram_all_methods(data_result_dict: DATA_RESULT_DICT):
  transposed = transpose_dict(data_result_dict)
  return grid_plot_wrapper(
    fn=plot_predictive_entropy_histogram_one_method,
    ncols=3,
    n_plots=len(transposed),
    get_kwargs=[],
  )

# def plot_auroc_entropy_all(preds_ood_dict: dict,
#                            preds_test_dict: dict,
#                            model_list: list,
#                            data_training: str,
#                            data_ood: str,
#                            save_folder: str
#                            ):
#   fig, ax = plt.subplots()
#   plt.subplots_adjust(left=0.20, bottom=0.20)
#   ax.plot([0, 1], [0, 1], linestyle=":", color="black")
#
#   thresholds = np.arange(0, 1.02, 0.02)
#   for model in tqdm(model_list):
#     tpr_values = np.zeros(
#       shape=(thresholds.shape[0], preds_ood_dict[model].shape[0]))
#     for i in range(preds_ood_dict[model].shape[0]):
#       predicted_labels_ood = preds_ood_dict[model].mean(1)[i, :]
#       predicted_labels_test = preds_test_dict[model].mean(1)[i, :]
#
#       ood_size = predicted_labels_ood.shape[0]
#       test_size = predicted_labels_test.shape[0]
#       anomaly_targets = jnp.concatenate(
#         (np.zeros(test_size), np.ones(ood_size)))
#
#       entropy_test = -(
#           predicted_labels_test * jnp.log(predicted_labels_test + eps)
#       ).sum(1)
#       entropy_ood = -(
#           predicted_labels_ood * jnp.log(predicted_labels_ood + eps)
#       ).sum(1)
#       scores = jnp.concatenate((entropy_test, entropy_ood))
#       fpr, tpr, _ = sklearn.metrics.roc_curve(anomaly_targets, scores)
#
#       for j in range(thresholds.shape[0]):
#         fpr_idx = np.abs(fpr - thresholds[j]).argmin()
#         tpr_values[j, i] = tpr[fpr_idx]
#
#     tpr_value_mean = tpr_values.mean(1)
#     tpr_value_std = tpr_values.std(1)
#     tpr_value_ste = scipy.stats.sem(tpr_values, axis=1)
#     ax.plot(thresholds,
#             tpr_value_mean,
#             color=color_list[model],
#             label=label_list[model],
#             linestyle=linestyle_list[model])
#     ax.fill_between(
#       thresholds,
#       tpr_value_mean - tpr_value_ste,
#       tpr_value_mean + tpr_value_ste,
#       color=color_list[model],
#       alpha=alpha,
#     )
#   # ax.legend(facecolor="white")
#   ax.set_xlabel(f"False Positive Rate")
#   ax.set_ylabel(f"True Positive Rate")
#   ax.set_ylim([-0.05, 1.05])
#   ax.set_xlim([-0.03, 1.03])
#   # ax.set_title(f"Trained on {train_name}, Evaluated on {ood_name}")
#   os.makedirs(f"{save_folder}", exist_ok=True)
#   fig.savefig(
#     f"{save_folder}/{data_training.lower()}"
#     + "_"
#     + f"{data_ood.lower()}_auroc_entropy.pdf",
#   )
#   return fig, ax