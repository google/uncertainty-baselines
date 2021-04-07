import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from run_deferred_prediction import DEFERRED_PREDICTION_MODEL_TYPES

METRIC_NAME_TO_PLOTTING_STYLE = {
  'auprc': 'AUPRC',
  'auroc': 'AUROC',
  'accuracy': 'Accuracy',
  'ece': 'ECE',
  'negative_log_likelihood': 'Negative Log Likelihood'
}
MODEL_TYPE_TO_PLOTTING_STYLE = {
  'deterministic': 'Deterministic',
  'dropout': 'MC Dropout',
  'variational_inference': 'Mean-Field VI',
  'radial': 'Radial BNN',
  'ensemble': 'Deep Ensemble',
  'dropoutensemble': 'MC Dropout Ensemble'
}


def get_results_from_model_dir(model_dir: str):
  """Return results pd.DataFrame, or None, from a subdirectory containing
  results from a particular model type run on deferred prediction.

  Args:
    model_dir: `str`, subdirectory that contains a `results.tsv` file
      for the corresponding model type.
  Returns:
    pd.DataFrame or None
  """
  model_results_path = os.path.join(model_dir, 'results.tsv')
  if not tf.io.gfile.exists(model_results_path):
    return None

  print(f'Found results at {model_results_path}.')

  with tf.io.gfile.GFile(model_results_path, 'r') as f:
    return pd.read_csv(f, sep='\t')


def plot_deferred_prediction_results(
  results_dir: str, output_dir: str, model_type: Optional[str] = None):
  """
  Load deferred prediction results from the specified directory.

  Args:
    results_dir: `str`, directory from which deferred prediction results are
      loaded. If you aim to generate a plot for multiple models, this should
      point to a directory in which each subdirectory has a name corresponding
      to a model type (see the top of run_deferred_prediction.py for supported
      model types).
      Otherwise, if you aim to generate a plot for a particular model,
      `results_dir` should point directly to a model type's subdirectory, and
      the `model_type` argument should be provided.
    output_dir: `str`, directory to which plots will be written
    model_type: `str`, should be provided if generating a plot for only one
      particular model.
  """
  if model_type is None:
    dir_path, child_dir_suffixes, _ = next(tf.io.gfile.walk(results_dir))
    model_dirs = []
    results_dfs = []

    for child_dir_suffix in child_dir_suffixes:
      try:
        model_type = child_dir_suffix.split('/')[0]
      except:
        continue

      if model_type not in DEFERRED_PREDICTION_MODEL_TYPES:
        continue

      model_dir = os.path.join(dir_path, model_type)
      print(f'Found deferred prediction results directory '
            f'for model {model_type} at {model_dir}.')
      model_dirs.append(model_dir)

    for model_dir in model_dirs:
      results = get_results_from_model_dir(model_dir)
      if results is not None:
        results_dfs.append(results)

    results_df = pd.concat(results_dfs, axis=0)
  else:
    print(f'Plotting deferred prediction results for model type {model_type}.')
    model_results_path = os.path.join(results_dir, 'results.tsv')
    try:
      with tf.io.gfile.GFile(model_results_path, 'r') as f:
        results_df = pd.read_csv(f, sep='\t')
    except (FileNotFoundError, tf.errors.NotFoundError):
      raise FileNotFoundError(f'No results found at path {model_results_path}.')

  plot_results_df(results_df, output_dir)


def plot_results_df(results_df: pd.DataFrame, output_dir: str):
  """
  Creates a deferred prediction plot for each metric in the results_df.
  For a particular model, aggregates metric values (e.g., accuracy)
  when retain proportion is identical. If retain proportion, train seed,
  and eval seed are identical for a particular model and metric, we
  consider only the most recent result.

  Args:
   results_df: `pd.DataFrame`, expects columns: {metric, retain_proportion,
    value, model_type, train_seed, eval_seed, run_datetime}.
   output_dir: `str`, directory to which plots will be written
  """
  COLORS = plt.rcParams['axes.prop_cycle'].by_key()['color']
  MARKERS = ["o", "D", "s", "8", "^", "*"]

  metrics = set(results_df['metric'])

  for metric in metrics:
    fig, ax = plt.subplots()
    plotting_metric_name = METRIC_NAME_TO_PLOTTING_STYLE[metric]
    metric_df = results_df[results_df['metric'] == metric].copy()
    baselines = list(sorted(set(metric_df['model_type'])))

    for b, baseline in enumerate(baselines):
      baseline_metric_df = metric_df[metric_df['model_type'] == baseline].copy()

      # Sort by datetime (newest first)
      baseline_metric_df.sort_values(
        'run_datetime', inplace=True, ascending=False)

      # For a particular baseline model and metric, drop duplicates
      # (keeping the newest entry)
      # if the retain proportion, train seed, and eval seed are identical
      baseline_metric_df.drop_duplicates(
        subset=['retain_proportion', 'train_seed', 'eval_seed'],
        keep='first', inplace=True)

      # Group by retain proportion, and
      # compute the mean and standard deviation of the metric
      agg_baseline_metric_df = baseline_metric_df.groupby(
        'retain_proportion').value.agg(['mean', 'std']).reset_index()
      retained_data = agg_baseline_metric_df['retain_proportion']
      mean = agg_baseline_metric_df['mean']
      std = agg_baseline_metric_df['std']

      # Visualize mean with standard error
      ax.plot(
        retained_data, mean, label=MODEL_TYPE_TO_PLOTTING_STYLE[baseline],
        color=COLORS[b % len(COLORS)],
        marker=MARKERS[b % len(MARKERS)])
      ax.fill_between(
        retained_data, mean - std, mean + std, color=COLORS[b % len(COLORS)],
        alpha=0.25)
      ax.set(
        xlabel="Proportion of Data Retained", ylabel=plotting_metric_name)
      ax.legend()
      fig.tight_layout()

    if isinstance(output_dir, str):
      os.makedirs(output_dir, exist_ok=True)
      metric_plot_path = os.path.join(output_dir, f'{plotting_metric_name}.pdf')
      fig.savefig(
        metric_plot_path,
        transparent=True,
        dpi=300,
        format="pdf")
      print(
        f'Saved plot for metric {metric}, baselines {baselines} '
        f'to {metric_plot_path}.')
