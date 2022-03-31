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

from typing import Any, Dict, Iterable, List

import pandas as pd


_HPARAM_PREFIX = 'config.'
_RANDOM_SEED_COL = _HPARAM_PREFIX + 'seed'
_DATASET_COL = _HPARAM_PREFIX + 'dataset'
_MODEL_COL = 'model'


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


def is_hyperparameter(
    column: str,
    auxiliary_hparams: Iterable[str] = ('learning_rate', _MODEL_COL)
) -> bool:
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

  Returns: A subset of `df` corresponding to the experimental runs with optimal
    hyperparameters.
  """
  df = df.copy()
  best_hps = get_best_hyperparameters(
      df, tuning_metric, marginalization_hparams=marginalization_hparams,
      verbose=verbose)
  for k, v in best_hps.items():
    df = df[df[k] == v]
  return df
