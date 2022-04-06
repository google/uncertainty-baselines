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

"""Utility functions used to process xmanager experiments."""

import itertools
import re
from typing import Any, Dict, Iterable, List, Optional

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

# Pretraining datasets.
_UPSTREAM_DATASETS = ('jft/entity:1.0.0', 'imagenet21k')
# Fewshot datasets of interest.
_FEWSHOT_DATASETS = ('imagenet', 'pets', 'birds', 'col_hist', 'cifar100',
                     'caltech', 'cars', 'dtd', 'uc_merced')


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


def _is_higher_better(metric: str) -> bool:
  """Returns True if the metric is to be maximized (e.g., precision)."""
  return 'prec@' in metric or 'auroc' in metric


def random_seed_col() -> str:
  """Returns the name of the column containing the experimental random seed."""
  return _RANDOM_SEED_COL


def dataset_col() -> str:
  """Returns the name of the column containing the training dataset."""
  return _DATASET_COL


def model_col() -> str:
  """Returns the name of the column containing the model name."""
  return _MODEL_COL


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
    raise ValueError(f'Rows {df[non_nan_counts>=1]} have multiple set values!')
  return df.fillna(axis=1, method='bfill')[df.columns[0]]


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

  if _is_higher_better(tuning_metric):
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

  df = df.groupby([_MODEL_COL,
                   _DATASET_COL])[relevant_metrics].mean().reset_index()

  df = df.pivot(index=_MODEL_COL, columns=_DATASET_COL, values=relevant_metrics)
  df = df.dropna(axis=1, how='all')
  df.columns.set_names(['metric', 'dataset'], inplace=True)

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

    df.drop(columns=fewshot_metric, level=0, inplace=True)

  # For now, we only care about compute metrics on upstream datasets, and only
  # one of the two upstream compute columns has a non-NaN value for each model.
  # We process the compute metrics similarly to the fewshot metrics.
  compute_metrics = [c for c in df.columns.levels[0] if c in _COMPUTE_METRICS]
  for metric in compute_metrics:
    compute_vals = row_wise_unique_non_nan(df[metric][list(_UPSTREAM_DATASETS)])
    df.drop(metric, axis=1, level=0, inplace=True)
    df[metric, 'compute'] = compute_vals

  return df
