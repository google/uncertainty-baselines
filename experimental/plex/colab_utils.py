# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Utility functions used to process xmanager experiments."""

import enum
import itertools
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import immutabledict
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


_HPARAM_PREFIX = 'config.'
_RANDOM_SEED_COL = _HPARAM_PREFIX + 'seed'
_DATASET_COL = _HPARAM_PREFIX + 'dataset'
_MODEL_COL = 'model'

# In-distribution metrics across train/validation/test splits.
_SPLIT_METRICS = ('loss', 'prec@1', 'ece', 'calib_auc')
# Metrics specific to the diabetic retinopathy dataset.
_RETINA_METRICS = ('accuracy', 'negative_log_likelihood', 'ece',
                   'retention_auroc_auc', 'retention_accuracy_auc')
# Compute cost metrics.
_COMPUTE_METRICS = ('exaflops', 'tpu_days', 'gflops', 'ms_step')
# Shifted dataset metrics.
_SHIFTED_METRICS = ('nll', 'ece', 'brier', 'accuracy')

# Pretraining datasets.
_UPSTREAM_DATASETS = ('jft/entity:1.0.0', 'imagenet21k')
# Fewshot datasets of interest.
_FEWSHOT_DATASETS = ('imagenet', 'pets', 'birds', 'col_hist', 'cifar100',
                     'caltech', 'cars', 'dtd', 'uc_merced')

_NUM_CLASSES_BY_DATASET = immutabledict.immutabledict({
    'cifar10': 10,
    'cifar100': 100,
    'imagenet2012': 1000,
    'imagenet21k': 21841,
    'jft/entity:1.0.0': 18291,
    'retina_country': 2,
    'retina_severity': 2,
    # TODO(zmariet, dusenberrymw): Update for the specific ImageNet-Vid and
    # YTBB datasets, which use a subset of the 1000 Imagenet classes.
    'imagenet_variants': 1000,
    'few-shot birds': 200,
    'few-shot caltech': 102,
    'few-shot cars': 196,
    'few-shot cifar100': 100,
    'few-shot col_hist': 8,
    'few-shot dtd': 47,
    'few-shot imagenet': 1000,
    'few-shot pets': 37,
    'few-shot uc_merced': 21,


})


class MetricCategory(enum.Enum):
  PREDICTION = enum.auto()  # Prediction metrics, e.g., prec@1.
  UNCERTAINTY = enum.auto()  # Uncertainty metrics, e.g., ECE.
  ADAPTATION = enum.auto()  # Adaptation metrics, e.g., fewshot metrics.


def random_seed_col() -> str:
  """Returns the name of the column containing the experimental random seed."""
  return _RANDOM_SEED_COL


def dataset_col() -> str:
  """Returns the name of the column containing the training dataset."""
  return _DATASET_COL


def model_col() -> str:
  """Returns the name of the column containing the model name."""
  return _MODEL_COL


def upstream_datasets() -> Tuple[str, ...]:
  """Returns the datasets used for upstream pretraining."""
  return _UPSTREAM_DATASETS


def default_fewshot_datasets() -> Tuple[str, ...]:
  """Returns the default fewshot datasets used for reporting results."""
  return _FEWSHOT_DATASETS


def compute_metrics() -> Tuple[str, ...]:
  """Returns the metrics corresponding to computational requirements."""
  return _COMPUTE_METRICS


def ood_related_metrics() -> List[str]:
  """Returns the list of OOD metrics we care about for the open set recognition."""
  metrics = []
  # Out of distribution detection metrics
  metrics.extend(
      f'ood_{dset}_msp_auroc'
      for dset in ['cifar10', 'cifar100', 'svhn_cropped', 'places365_small'])
  metrics.extend(
      f'ood_{dset}_entropy_auroc'
      for dset in ['cifar10', 'cifar100', 'svhn_cropped', 'places365_small'])
  metrics.extend(
      f'ood_{dset}_mlogit_auroc'
      for dset in ['cifar10', 'cifar100', 'svhn_cropped', 'places365_small'])
  metrics.extend(f'ood_{dset}_maha_auroc'
                 for dset in ['cifar10', 'cifar100', 'svhn_cropped'])
  metrics.extend(f'ood_{dset}_rmaha_auroc'
                 for dset in ['cifar10', 'cifar100', 'svhn_cropped'])
  return metrics


def default_selected_metrics() -> List[str]:
  """Returns the list of metrics we care about for the big paper."""
  metrics = list(_COMPUTE_METRICS)

  # In-distribution metrics
  metrics.extend(f'test_{m}' for m in _SPLIT_METRICS)

  # Out of distribution metrics
  metrics.extend(f'{dset}_{m}' for dset, m in itertools.product(
      ['cifar_10h', 'imagenet_real'], _SPLIT_METRICS))

  # Out of distribution detection metrics
  metrics.extend(
      f'ood_{dset}_msp_auroc'
      for dset in ['cifar10', 'cifar100', 'svhn_cropped', 'places365_small']
  )

  # Shifted metrics
  metrics.extend(
      f'{dset}/{m}/mean'
      for dset, m in itertools.product(['imagenet_c'], _SHIFTED_METRICS))
  metrics.extend(f'imagenet_c/{m}' for m in ['mce', 'relative_mce'])
  metrics.extend(f'{dset}/{m}' for dset, m in itertools.product(
      ['imagenet_a', 'imagenet_r', 'imagenet_v2'], _SHIFTED_METRICS))
  metrics.extend(f'{dset}/{m}' for dset, m in itertools.product(
      ['imagenet_vid_robust', 'ytbb_robust'],
      ['accuracy_drop', 'accuracy_pmk', 'anchor_accuracy']))

  # Fewshot metrics
  metrics.extend(
      f'z/{dset}_{k}shot'
      for dset, k in itertools.product(_FEWSHOT_DATASETS, [1, 5, 10, 25])
  )

  # Retina metrics
  metrics.extend(
      f'{prefix}/{metric}' for metric, prefix in itertools.product(
          _RETINA_METRICS, ['in_domain_test', 'ood_test'])
  )

  return metrics


def get_base_metric(metric: str) -> str:
  """Validates `metric` and returns the base computed metric.

  Args:
    metric: A metric name that may contain auxiliary information such as
      the dataset it was computed on. For example: "ood_cifar10_msp_auroc".

  Returns:
    The base metric; for example, "auroc".

  Raises:
    ValueError: If metric does not follow the expected syntax.
  """
  if metric in _COMPUTE_METRICS:
    return metric
  # NOTE: `(?<!...)` means not preceded by `...`, and `(?!...)` means not
  # followed by `...`. `(?P<metric>)` gives the name `metric` to the group.
  pattern = (r'.*[_/](?P<metric>loss|likelihood|ece|auc|auroc|prec@1|\d+shot'
             r'|nll|brier|relative\_mce|(?<!relative\_)mce|accuracy\_drop'
             r'|accuracy\_pmk|anchor\_accuracy'
             r'|(?<!anchor\_)accuracy(?!(:\_pmk|\_drop)))')
  match = re.search(pattern, metric)
  if match is not None:
    return match.group('metric')
  else:
    raise ValueError(f'Unrecognized metric {metric}!')


def get_metric_category(metric: str) -> MetricCategory:
  """Returns which category `metric` belongs to for scoring purposes."""
  base_metric = get_base_metric(metric)
  if base_metric in [
      'loss', 'accuracy', 'likelihood', 'nll', 'brier', 'mce', 'relative_mce',
      'accuracy_drop', 'accuracy_pmk', 'anchor_accuracy'
  ]:
    if 'shot' in metric:
      return MetricCategory.ADAPTATION
    else:
      return MetricCategory.PREDICTION
  elif base_metric in ['auc', 'auroc', 'ece']:
    if 'shot' in metric:
      return MetricCategory.ADAPTATION
    else:
      return MetricCategory.UNCERTAINTY
  elif base_metric == 'prec@1':
    if 'shot' in metric:
      return MetricCategory.ADAPTATION
    else:
      return MetricCategory.PREDICTION
  raise ValueError(f'Metric {metric} is not used for scoring.')


def is_higher_better(metric: str) -> bool:
  """Returns True if the metric is to be maximized (e.g., precision)."""
  maximized_metrics = [
      'prec@1', 'accuracy', 'auc', 'auroc', 'anchor_accuracy', 'accuracy_pmk'
  ]
  minimized_metrics = [
      'loss', 'likelihood', 'ece', 'nll', 'brier', 'mce', 'relative_mce',
      'accuracy_drop'
  ]
  base_metric = get_base_metric(metric)
  if base_metric in maximized_metrics or 'shot' in base_metric:
    return True
  elif base_metric in minimized_metrics:
    return False
  else:
    raise ValueError(f'Metric {metric} is unrecognized. It was parsed as a '
                     f'{base_metric} base metric.')


def get_unique_value(df: pd.DataFrame, col: str) -> Any:
  """Returns the unique value in a dataframe column.

  Args:
    df: A pd.DataFrame.
    col: The column of interest.

  Returns:
    The unique value.

  Raises:
    ValueError: if the column does not have an unique value.
  """
  unique_values = df[col].unique()
  if len(unique_values) != 1:
    raise ValueError(
        f'Expected unique value in column {col}; got values {unique_values}!'
    )
  return unique_values[0]


def row_wise_unique_non_nan(df: pd.DataFrame) -> pd.Series:
  """Checks there is exactly one non-NA in each row, and returns its value."""
  non_nan_counts = df.notna().sum(axis=1)
  if not (non_nan_counts <= 1).all():
    num_incorrect_rows = int(np.sum(non_nan_counts > 1))
    raise ValueError(f'{num_incorrect_rows} rows have multiple set values!')
  return df.fillna(axis=1, method='bfill').iloc[:, 0]


def is_hyperparameter(
    column: str, auxiliary_hparams: Iterable[str] = ('learning_rate',)) -> bool:
  """Returns True if the column corresponds to a hyperparameter."""
  return column.startswith(_HPARAM_PREFIX) or column in auxiliary_hparams


def get_sweeped_hyperparameters(
    df,
    marginalization_hparams: Iterable[str] = (_RANDOM_SEED_COL,)) -> List[Any]:
  """Identifies the columns that correspond to a hyperparameter tuning sweep."""
  hparams = [c for c in df.columns if is_hyperparameter(c)]
  return [
      h for h in hparams
      if len(df[h].unique()) > 1 and h not in marginalization_hparams
  ]


def get_best_hyperparameters(df: pd.DataFrame,
                             tuning_metric: str,
                             marginalization_hparams: Iterable[str] = (
                                 _RANDOM_SEED_COL,),
                             verbose: bool = True) -> Dict[str, Any]:
  """Returns the best choice of hyperparameters for a given model and dataset.

  Each row of a dataframe corresponds to a single experiment; each column is
  either a standard hyperparameter (e.g., learning rate), a hyperparameter to
  marginalize over (e.g., random seed), or a metric (e.g., test loss).

  For each choice of non-marginalization hyperparameters, the tuning metric
  (e.g., the validation loss) is averaged over all marginalization
  hyperparameters; the hyperparameters achieving the best average metric are
  returned.

  In the unlikely event that separate hyperparameter values produce an optimal
  tuning metric, the first hyperparameter choice is returned.

  Args:
    df: A pd.DataFrame corresponding to a single model and dataset; each row
      corresponds to a different choice of hyperparameters and / or parameters
      to aggregate over (e.g., the random seed).
    tuning_metric: The metric over which hyperparameters are tuned.
    marginalization_hparams: Columns of the dataframe that correspond to
      hyperparameters to aggregate over rather than tune (e.g., random seed).
    verbose: If True, logs to stdout which hyperparameters were swept over, and
      which values are optimal.

  Returns:
    A dictionary mapping hyperparameters to their optimal values.
  """
  dataset = get_unique_value(df, _DATASET_COL)
  model = get_unique_value(df, _MODEL_COL)
  hps = get_sweeped_hyperparameters(df, marginalization_hparams)

  if not hps:  # There is no hyperparameter tuning.
    if verbose:
      print(f'No hyperparameter tuning for {model} on {dataset}.')
    return {}

  aggregated_results = df.groupby(hps)[tuning_metric].agg('mean').reset_index()

  if is_higher_better(tuning_metric):
    best_value_idx = aggregated_results[tuning_metric].idxmax()
  else:
    best_value_idx = aggregated_results[tuning_metric].idxmin()

  best_hps = aggregated_results.loc[best_value_idx][hps].to_dict()
  if verbose:
    print(f'For {model} on {dataset}, found {len(hps)} hyperparameters: {hps}.')
    print(f'\tBest hyperparameters: {best_hps}.')
  return best_hps


def get_tuned_results(df: pd.DataFrame,
                      tuning_metric: str,
                      marginalization_hparams: Iterable[str] = (
                          _RANDOM_SEED_COL,),
                      verbose: bool = True) -> pd.DataFrame:
  """Returns dataframe rows corresponding to optimal hyperparameter choices.

  Args:
    df: pd.DataFrame corresponding to different evaluations of a single model on
      a single dataset. Each row corresponds to a single experiment. This
      dataframe must contain at minimum columns [MODEL_COL, DATASET_COL,
      tuning_metric].
    tuning_metric: The metric over which hyperparameters are tuned.
    marginalization_hparams: Columns of the dataframe that correspond to
      hyperparameters to aggregate over rather than tune (e.g., random seed).
    verbose: if True, logs to stdout which hyperparameters were swept over, and
      which values are optimal.

  Returns:
    A subset of `df` corresponding to the experimental runs with optimal
    hyperparameters.
  """
  df = df.copy()
  best_hps = get_best_hyperparameters(
      df,
      tuning_metric,
      marginalization_hparams=marginalization_hparams,
      verbose=verbose)
  for k, v in best_hps.items():
    df = df[df[k] == v]
  return df


def _fill_upstream_test_metrics(df: pd.DataFrame) -> pd.DataFrame:
  """Copies validation metrics to test metrics on upstream datasets.

  Upstream datasets (Imagenet21K and JFT) have no dedicated test set, since they
  are used to pretrain models rather than evaluate them. To simplify downstream
  plotting analysis, we consider upstream validation metrics to be "test"
  metrics since validation metrics are typically dropped.

  Args:
    df: pd.DataFrame where rows correspond to model classes, and columns to
      reported metrics.

  Returns:
    A copy of `df` where the rows corresponding to models trained on upstream
    datasets have their in-distribution test metrics filled with in-distribution
    validation metrics.
  """
  df = df.copy()
  for m in _SPLIT_METRICS:
    if f'val_{m}' in df.columns:
      idx = df[_DATASET_COL].isin(_UPSTREAM_DATASETS)
      df.loc[idx, f'test_{m}'] = df.loc[idx, f'val_{m}']
  return df


def process_tuned_results(
    df: pd.DataFrame,
    relevant_metrics: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
  """Cleans and reformats the dataframe of all results used for the big paper.

  Args:
    df: pd.DataFrame where each row represents the average performance of a
      (model, train_dataset) pair with optimal hyperparameters across a variety
      of metrics. This dataframe is expected to have columns `[_MODEL_COL,
      _DATASET_COL]`, as well as all chosen metrics to analyze
      (`relevant_metrics`).
    relevant_metrics: Optional list of all metrics to analyze; defaults to
      metrics produced by `default_selected_metrics()`. These metrics must all
      be columns of `df`.

  Returns:
    pd.DataFrame where each row corresponds to a model in `df`, and each column
    is a 2-level multiindex of stucture `(metric, dataset)`, where metrics
    correspond to `relevant_metrics` and datasets are the available training
    sets. Note that not all `(metric, dataset)` columns exist, since some
    metrics are only reported on a subset of available training sets (for
    example, we don't report Cifar100 OOD numbers when training on Cifar100).
  """
  df = _fill_upstream_test_metrics(df)

  if relevant_metrics is None:
    relevant_metrics = default_selected_metrics()
  relevant_metrics = [m for m in relevant_metrics if m in df.columns]

  df = df.groupby([_MODEL_COL,
                   _DATASET_COL])[relevant_metrics].mean().reset_index()

  df = df.pivot(index=_MODEL_COL, columns=_DATASET_COL, values=relevant_metrics)
  df = df.dropna(axis=1, how='all')
  df.columns = df.columns.set_names(['metric', 'dataset'])

  # Fewshot metrics are only reported on upstream datasets. For each fewshot
  # metric `m`, `df[m]` is a dataframe with two columns: one for each possible
  # upstream dataset. Each row of `df[m]` only has one column with a non NaN
  # value, so we collapse both columns into a single column.
  fewshot_metrics = (c for c in df.columns.levels[0] if c.startswith('z/'))
  for fewshot_metric in fewshot_metrics:
    fewshot_match = re.match(r'z/(.*)_(\d*)shot', fewshot_metric)
    if fewshot_match is not None:
      dset, k = fewshot_match.groups()
    else:
      raise ValueError(f'Unconsistent fewshot metric {fewshot_metric}.')
    df[f'{k}shot_prec@1',
       f'few-shot {dset}'] = row_wise_unique_non_nan(df[fewshot_metric])

    df = df.drop(columns=fewshot_metric, level=0)

  # For now, we only care about compute metrics on upstream datasets, and only
  # one of the two upstream compute columns has a non-NaN value for each model.
  # We process the compute metrics similarly to the fewshot metrics.
  compute_cols = [c for c in df.columns.levels[0] if c in _COMPUTE_METRICS]
  for metric in compute_cols:
    compute_vals = row_wise_unique_non_nan(df[metric][list(_UPSTREAM_DATASETS)])
    df = df.drop(metric, axis=1, level=0)
    df[metric, 'compute'] = compute_vals

  return df


def _uniform_entropy(num_classes: int) -> float:
  """Entropy of the uniform categorical distribution over `num_classes`."""
  # TODO(zmariet, jsnoek): Consider switching to log2.
  return np.log(num_classes)  # Typically written as -n * 1/n * log(1/n).


def _normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
  """Normalizes metrics to [0, 1] (except for multiclass NLL); higher is better.

  All metrics are normalized to [0, 1] except for NLL on multiclass datasets;
  those are normalized to [0, max_entropy / U], where U is the entropy of the
  categorical uniform distribution. U does not bound multiclass entropy;
  however, the multiclass uniform entropy is too large to be a meaningful bound.

  Args:
    df: pd.DataFrame indexed by model name; each column corresponds to a metric,
      and each row corresponds to a model choice.

  Returns:
    A copy of `df` where all scores have been normalized and higher values
    indicate better performance.
  """
  df = df.copy()
  for column in df.columns:
    metric, dataset = column
    metric_type = get_base_metric(metric)
    if metric_type == 'ece':
      df[column] = 1. - df[column]

    elif metric_type in ['loss', 'likelihood', 'nll']:
      num_classes = _NUM_CLASSES_BY_DATASET[dataset]
      df[column] = 1. - df[column] / _uniform_entropy(num_classes)
  return df


def _drop_unused_measurements(
    df: pd.DataFrame,
    drop_compute: bool,
    drop_1shot: bool,
    drop_incomplete_measurements: bool,
    datasets: Optional[Iterable[str]] = None) -> pd.DataFrame:
  """Drops rows and columns that will not be used for analysis.

  Args:
    df: pd.DataFrame where each row corresponds to a model in `df`, and each
      column is a 2-level multiindex of stucture `(metric, dataset)`; in typical
      usage, `df` was obtained by calling `process_tuned_results`.
    drop_compute: Whether to drop metrics for compute cost.
    drop_1shot: Whether to include fewshot@1 results, which tend to have high
      variance.
    drop_incomplete_measurements: If True, only models which report values
      across all metrics will be considered for scoring. Otherwise, only models
      that report no measurements at all are dropped.
    datasets: Optional datasets of interest. If None, all datasets are kept.

  Returns:
    A pd.DataFrame indexed by model name with columns corresponding to scores.
  """
  df = df.copy()

  if drop_compute:
    df = df.drop(columns='compute', level=1, errors='ignore')
  if drop_1shot:
    cols_to_drop = [c for c in df.columns.levels[0] if c.startswith('1shot_')]
    df = df.drop(columns=cols_to_drop, level=0, errors='ignore')
  if datasets:
    df = df.drop(
        columns=[c for c in df.columns.levels[1] if c not in datasets],
        level=1)

  if drop_incomplete_measurements:
    df = df.dropna(how='any')
  else:
    df = df.dropna(how='all')

  # Level-0 column indices remain even if level-1 has been removed.
  df.columns = df.columns.remove_unused_levels()
  return df


def compute_score(df: pd.DataFrame,
                  drop_incomplete_measurements: bool,
                  baseline_model: Optional[str] = None,
                  drop_1shot: bool = True,
                  datasets: Optional[Iterable[str]] = None) -> pd.DataFrame:
  """Computes aggregate prediction, uncertainty, and adaptation scores.

  Args:
    df: pd.DataFrame where each row corresponds to a model in `df`, and each
      column is a 2-level multiindex of stucture `(metric, dataset)`; in typical
      usage, `df` was obtained by calling `process_tuned_results`.
    drop_incomplete_measurements: If True, only models which report values
      across all metrics will be considered for scoring. Otherwise, only models
      that report no measurements at all are dropped.
    baseline_model: If provided, scores will be computed relatively to the
      metrics of this model.
    drop_1shot: Whether to include fewshot@1 results, which tend to have high
      variance.
    datasets: Optional datasets of interest. If None, all datasets are kept.

  Returns:
    A pd.DataFrame indexed by model name with columns corresponding to scores.
  """
  df = _drop_unused_measurements(
      df,
      drop_compute=True,
      drop_1shot=drop_1shot,
      datasets=datasets,
      drop_incomplete_measurements=drop_incomplete_measurements)
  df = _normalize_scores(df)

  if baseline_model:
    df /= df.loc[baseline_model]

  categories = {category.name: [] for category in MetricCategory}
  for metric in df.columns.levels[0]:
    categories[get_metric_category(metric).name].append(metric)

  scores = df.mean(axis=1, skipna=False).to_frame(name='score')
  for category, metrics in categories.items():
    # Don't compute scores for methods that don't report all metrics in the
    # current category, since this would make metrics such as ranking
    # meaningless.
    category_df = df[metrics].dropna(how='any')
    if category_df.empty:
      continue

    # Average score per category.
    scores[f'score_{category.lower()}'] = category_df.mean(axis=1)

    # Number of times each model was the best per category.
    best_per_task = category_df.idxmax()
    col_name = f'#_best_{category.lower()}'
    scores[col_name] = best_per_task.groupby(best_per_task).count()
    # Fill in column with 0s for tasks that are never the best.
    scores[col_name] = scores[col_name].fillna(
        {model: 0 for model in category_df.index})

    # Average rank per category.
    scores[f'mean_rank_{category.lower()}'] = category_df.rank(
        ascending=False).mean(axis=1)

  return scores.sort_values(by='score', ascending=False)


def rank_models(df: pd.DataFrame,
                drop_incomplete_measurements: bool,
                drop_1shot: bool = True,
                datasets: Optional[Iterable[str]] = None) -> pd.DataFrame:
  """Ranks models across all metrics of interest; lower rank is better."""
  df = _drop_unused_measurements(
      df,
      drop_compute=True,
      drop_1shot=drop_1shot,
      drop_incomplete_measurements=drop_incomplete_measurements,
      datasets=datasets)
  df = _normalize_scores(df)
  return df.rank(ascending=False)


def rank_models_by_category(
    df: pd.DataFrame,
    drop_incomplete_measurements: bool,
    drop_1shot: bool = True,
    datasets: Optional[Iterable[str]] = None) -> Dict[str, pd.DataFrame]:
  """Returns a dictionary mapping MetricCategory to per-metric model ranking."""
  df = _drop_unused_measurements(
      df,
      drop_compute=True,
      drop_1shot=drop_1shot,
      datasets=datasets,
      drop_incomplete_measurements=drop_incomplete_measurements)
  df = _normalize_scores(df)

  categories = {category.name: [] for category in MetricCategory}
  for metric in df.columns.levels[0]:
    categories[get_metric_category(metric).name].append(metric)

  return {
      category.lower(): df[metrics].rank(ascending=False)
      for category, metrics in categories.items()
  }


def make_radar_plot(df,
                    row,
                    color,
                    max_val,
                    ax,
                    xticklabels=None,
                    yscales=None,
                    fontfamily='serif',
                    fontsize=40):
  """Generate a radar plot given a dataframe of results.

  Args:
    df: A pandas dataframe with methods as rows and tasks as columns
    row: A string indicating the row name to plot.
    color: The color to use for this row in the plot.
    max_val: The maximum value over all columns. It's recommended to normalize
      the columns and then set this to 1.
    ax: A pyplot axis handle corresponding to axis to plot on.
    xticklabels: List of strings with labels for the x-ticks around the
      perimeter of the plot.
    yscales: List of tuples containing (min, max) indicating the ranges for
      each y-axis in the plot (corresponding to xticklabels).
    fontfamily: String indicating the matplotlib font family to use.
    fontsize: Integer indicating the font size.
  """

  categories = list(df)[0:]
  num_categories = len(categories)

  nticks = 5
  angles = [
      n / float(num_categories) * 2 * math.pi for n in range(num_categories)
  ]
  angles += angles[:1]

  ax.set_theta_offset(math.pi / 2)
  ax.set_theta_direction(-1)

  xticklabels = xticklabels if xticklabels is not None else categories
  plt.xticks(
      angles[:-1],
      xticklabels,
      color='black',
      size=fontsize,
      fontweight='normal',
      fontname=fontfamily)

  ax.set_rlabel_position(0)

  #  Rescale the values to fit in each y-axis.
  values = df.loc[row].values.tolist()[0:].copy()
  rescaled_values = []
  if yscales is not None:
    for i, k in enumerate(yscales):
      min_val, max_val = k
      rescaled_values.append((float(values[i]) - min_val) / (max_val - min_val))
  values = rescaled_values
  values += values[:1]

  max_val += 0.2  # Adds some padding for ticklabels
  ticks = np.linspace(0, max_val, nticks)
  scaled_ticks = np.linspace(yscales[0][0], yscales[0][1], nticks)
  ticklabels = ['%.2f' % i for i in scaled_ticks]
  ax.set_yticks(ticks, ticklabels)
  ax.set_ylim(0.0, max_val)
  ax.set_yticklabels([])

  ax.plot(angles, values, color=color, linewidth=1, linestyle='solid')
  ax.fill(angles, values, color=color, alpha=0.5)

  ax.spines['polar'].set_visible(False)
  gridlines = ax.yaxis.get_gridlines()
  gridlines[-1].set_visible(False)
  ax.patch.set_alpha(0.01)

  def add_new_yaxis(min_range, max_range, angle):
    # Add ticks along the other axes.
    ax2 = ax.figure.add_axes(
        ax.get_position(),
        projection='polar',
        label='twin',
        frameon=False,
        theta_direction=ax.get_theta_direction(),
        theta_offset=ax.get_theta_offset(),
        zorder=0,  # zorder seems to be buggy for polar plots
        alpha=0.1)
    ax2.xaxis.set_visible(False)
    scaled_ticks = np.linspace(min_range, max_range, nticks)
    ticklabels = ['%.2f' % i for i in scaled_ticks]
    ax2.set_yticks(ticks, ticklabels)
    ax2.set_yticklabels(
        ticklabels,
        fontdict={
            'fontsize': fontsize * .66,
            'color': 'grey'
        },
        zorder=1)
    ax2.set_yticklabels(
        ticklabels, fontdict={'fontsize': fontsize * .8}, zorder=1)
    ax2.tick_params(zorder=0.5)
    ax2.set_ylim(0.0, max_val)
    ax2.set_theta_zero_location('W', offset=-np.rad2deg(angle) + 22.5 - 90)

    # Remove the tick label at zero
    ax2.yaxis.get_major_ticks()[0].label1.set_visible(False)
    ax2.yaxis.get_major_ticks()[-1].set_zorder(0.1)
    ax2.spines['polar'].set_visible(False)
    ax2.xaxis.set_visible(False)
    ax2.yaxis.set_zorder(0.1)
    ax2.yaxis.grid(False)
    ax2.xaxis.grid(False)

  for i in range(0, num_categories):
    add_new_yaxis(yscales[i][0], yscales[i][1], angles[i])


def process_fewshot_for_moe_comparison(
    measurements_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
  """Format the fewshot results for 'MoE', '[Det]_4' and '[MoE]_4'.

  This function is not needed for Det, BE and Het since they had dedicated xm
  jobs for their fewshot evaluations.

  Args:
    measurements_dict: dictionary of pd.DataFrame's where one row corresponds
      to one seed for one given model.

  Returns:
    A pd.DataFrame indexed by model name with multindex columns corresponding to
    pairs of (fewshot metric, fewshot dataset).
  """

  def _parse_column(c):
    match = re.fullmatch(r'z/(.*)_(\d*)shot_(.*)$', c)
    is_valid = match is not None and 'best_l2' not in c
    if not is_valid:
      return None
    else:
      dataset, shot, metric_type = match.groups()
      column_name = (f'{shot}shot_{metric_type}', f'few-shot {dataset}')
      return column_name

  rows = []
  for model_name in ('MoE', '[Det]_4', '[MoE]_4'):
    # We average over the different seeds.
    fewshot_dict = measurements_dict[model_name].mean(axis=0).to_dict()
    # We format the names of the columns.
    column_names = {c: _parse_column(c) for c in fewshot_dict}
    fewshot_dict = {
        column_name: fewshot_dict[key]
        for key, column_name in column_names.items()
        if column_name is not None
    }
    fewshot_dict['model_name'] = model_name
    rows.append(fewshot_dict)

  df = pd.DataFrame(rows).set_index('model_name')
  df.columns = pd.MultiIndex.from_tuples(
      df.columns, names=['metric', 'dataset'])
  return df
