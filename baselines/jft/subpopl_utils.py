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

"""Utils for computing metrics under subpopulation shift."""

from typing import Any, Callable, Dict

import numpy as np
import tensorflow as tf
import input_utils  # local file import from baselines.jft


def eval_subpopl_metrics(
    subpopl_val_ds_splits: Dict[str, tf.data.Dataset],
    evaluation_fn: Callable[..., Any],
    opt_target_repl: Any,
    n_prefetch: int = 1,
):
  """Evaluate the model under subpopulation shift.

  Args:
    subpopl_val_ds_splits: A dictionary mapping from subpopulation name to a
      `tf.data.Dataset` with the corresponding data.
    evaluation_fn: Function to evaluate the model with the parameters provided
      in `opt_target_repl`.
    opt_target_repl: The target of the replicated optmizer (`opt_repl.target`).
    n_prefetch: Number of points to pre-fectch in the dataset iterators.

  Returns:
    A dictionary of measurements for subpopulation shift metrics.
  """
  subpopl_measurements = {}
  # Iterate over subpopulations.
  for val_subpopl_name, val_ds in subpopl_val_ds_splits.items():
    val_iter = input_utils.start_input_pipeline(val_ds, n_prefetch=n_prefetch)
    ncorrect, nseen = 0, 0
    for batch in val_iter:
      batch_ncorrect, _, batch_n, _ = (
          evaluation_fn(opt_target_repl, batch['image'], batch['labels'],
                        batch['mask']))
      # All results are a replicated array shaped as follows:
      # (local_devices, per_device_batch_size, elem_shape...)
      # with each local device's entry being identical as they got psum'd.
      # So let's just take the first one to the host as numpy.
      ncorrect += np.sum(np.array(batch_ncorrect[0]))
      nseen += np.sum(np.array(batch_n[0]))

    subpopl_measurements.update({
        f'subpopl_{val_subpopl_name}_prec@1': ncorrect / nseen,
    })

  # Calculate aggregated metrics over subpopulations.
  agg_measurements = {}
  precs = [v for k, v in subpopl_measurements.items() if k.endswith('_prec@1')]
  agg_measurements['subpopl_avg_prec@1'] = np.mean(precs)
  agg_measurements['subpopl_med_prec@1'] = np.median(precs)
  agg_measurements['subpopl_var_prec@1'] = np.var(precs)
  agg_measurements['subpopl_p95_prec@1'] = np.percentile(precs, 95)
  agg_measurements['subpopl_p75_prec@1'] = np.percentile(precs, 75)
  agg_measurements['subpopl_p25_prec@1'] = np.percentile(precs, 25)
  agg_measurements['subpopl_p05_prec@1'] = np.percentile(precs, 5)

  return agg_measurements
