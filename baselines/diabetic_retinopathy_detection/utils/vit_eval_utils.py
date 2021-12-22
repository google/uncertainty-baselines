from .eval_utils import (
  add_joint_dicts, compute_loss_and_accuracy_arrs_for_all_datasets,
  compute_metrics_for_all_datasets)
from .metric_utils import log_vit_validation_metrics


def evaluate_vit_predictions(
  dataset_split_to_containers,
  is_deterministic,
  num_bins=15,
  return_per_pred_results=False
):
  """Compute evaluation metrics given ViT predictions.

  Args:
    dataset_split_to_containers: Dict, for each dataset, contains `np.array`
      predictions, ground truth, and uncertainty estimates.
    is_deterministic: bool, is the model a single deterministic network.
      In this case, we cannot capture epistemic uncertainty.
    num_bins: int, number of bins to use with expected calibration error.
    return_per_pred_results: bool,
  Returns:
    Union[Tuple[Dict, Dict], Dict]
      If return_per_pred_results, return two Dicts. Else, return only the second.
      first Dict:
        for each dataset, per-prediction results (e.g., each prediction,
        ground-truth, loss, retention arrays).
      second Dict:
        for each dataset, contains `np.array` predictions, ground truth,
        and uncertainty estimates.
  """
  eval_results = add_joint_dicts(
    dataset_split_to_containers, is_deterministic=is_deterministic)

  # For each eval dataset, add NLL and accuracy for each example
  eval_results = compute_loss_and_accuracy_arrs_for_all_datasets(eval_results)

  # Compute all metrics for each dataset --
  # Robustness, Open Set Recognition, Retention AUC
  metrics_results = compute_metrics_for_all_datasets(
    eval_results, use_precomputed_arrs=False, ece_num_bins=num_bins,
    compute_retention_auc=True,
    verbose=False)

  # Log metrics
  log_vit_validation_metrics(metrics_results)

  if return_per_pred_results:
    return eval_results, metrics_results
  else:
    return metrics_results
