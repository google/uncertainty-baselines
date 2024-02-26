# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Utils for fewshot learning."""

import functools
import multiprocessing.pool

from absl import logging
from big_vision import input_pipeline
import big_vision.pp.builder as pp_builder
import jax.numpy as jnp
import numpy as np
import robustness_metrics as rm
from sklearn.linear_model import LogisticRegression
import tensorflow_datasets as tfds
import ood_utils  # local file import from baselines.jft


def evaluate(clfs, x_test, labels_test, x_ood_test, eval_ood_detection):
  """Evaluates metrics for ensemble of logistic regressors."""
  ens_size = len(clfs)
  repr_size = x_test.shape[-1] // ens_size
  probs = 0.0
  for i, clf in enumerate(clfs):
    probs += clf.predict_proba(x_test[:, i * repr_size : (i + 1) * repr_size])
  probs = probs / ens_size
  # Accuracy.
  int_preds = jnp.argmax(probs, axis=1)
  accuracy = jnp.mean(int_preds == labels_test)

  ece = rm.metrics.ExpectedCalibrationError(num_bins=15)
  ece.add_batch(probs, label=labels_test)
  ece_res = ece.result()["ece"]

  nll_metric = rm.metrics.NegativeLogLikelihood()
  nll_metric.add_batch(probs, label=labels_test)
  nll_res = nll_metric.result()["nll"]

  calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
  int_preds = np.argmax(probs, axis=-1)
  confidence = np.max(probs, axis=-1)

  calib_auc.add_batch(
      int_preds, label=labels_test, confidence=confidence.astype("float32"))
  calib_auc_res = calib_auc.result()["calibration_auc"]

  metric_results = {
      "test_prec@1": accuracy,
      "test_ece": ece_res,
      "test_nll": nll_res,
      "test_calib_auc": calib_auc_res,
  }

  ## MSP AUC.
  if eval_ood_detection:
    ood_probs = 0.0
    for i, clf in enumerate(clfs):
      ood_probs += clf.predict_proba(x_ood_test[:, i * repr_size:(i + 1) *
                                                repr_size])
    ood_probs = ood_probs / ens_size

    scores0 = 1 - np.max(probs, axis=-1)
    scores1 = 1 - np.max(ood_probs, axis=-1)
    ood_labels0 = np.zeros_like(scores0)
    ood_labels1 = np.ones_like(scores1)

    scores = jnp.concatenate([scores0, scores1])
    ood_labels = jnp.concatenate([ood_labels0, ood_labels1])
    msp_res = ood_utils.compute_ood_metrics(ood_labels, scores)["auroc"]
    metric_results["msp_auroc"] = msp_res

  return metric_results


def train_and_eval(l2_reg, x, y, ens_size,
                   x_test, labels_test, x_ood_test, eval_ood_detection):
  """Trains logistic regressors at l2_reg and evaluate them right after."""
  clfs = []
  repr_size = x.shape[-1] // ens_size
  for i in range(ens_size):
    clf = LogisticRegression(
        random_state=0, multi_class="multinomial", C=1 / l2_reg, max_iter=150)
    clf.fit(x[:, i * repr_size:(i + 1) * repr_size], y)
    clfs.append(clf)
  metric_results = evaluate(clfs, x_test, labels_test, x_ood_test,
                            eval_ood_detection)
  return l2_reg, metric_results


def select_best_l2_reg(results, shots_list, l2_regs, how="all"):
  """Selects best l2 regularization for each (dataset, num_shots) pair.

  Args:
    results: A dict of fewshot evalution results. Key is dataset name,
      value is also a dict, whose key is a tuple of (num_shots, l2_reg) with
      value being a dict of metric results. An example would be:

      results = {
          'birds': {
              (5, 1.0): {'test_prec@1': 0.5},
              (5, 2.0): {'test_prec@1': 0.6},
              (10, 1.0): {'test_prec@1': 0.55},
              (10, 2.0): {'test_prec@1': 0.65},
          },
          'cars': {
              (5, 1.0): {'test_prec@1': 0.6},
              (5, 2.0): {'test_prec@1': 0.7},
              (10, 1.0): {'test_prec@1': 0.65},
              (10, 2.0): {'test_prec@1': 0.76},
          },
          'cifar100': {
              (5, 1.0): {'test_prec@1': 0.8},
              (5, 2.0): {'test_prec@1': 0.7},
              (10, 1.0): {'test_prec@1': 0.85},
              (10, 2.0): {'test_prec@1': 0.76},
          }
      }

    shots_list: A list of integers, each being number of shots. An example
      would be shots_list = [5, 10].
    l2_regs: A list of floats, each being l2 regularization coefficient.
      An example would be shots_list = [1.0, 2.0].
    how: L2 regularization selection scheme. Currently only accepts `all` and
      `leave-self-out`.

  Returns:
    best_l2: A dict of best l2 regularization coefficients. Key is a tuple of
      (dataset_name, num_shots), value is the best l2 regularization for the
      corresponding dataset and fewshot setting.
  """

  if how not in ["all", "leave-self-out"]:
    raise ValueError(f"`{how}` is supplied as the L2 regularization selection "
                     "scheme, but currently only `all` and `leave-self-out` "
                     "are supported.")
  best_l2 = {}
  for num_shots in shots_list:
    reg_ranks = {}
    for dataset_name, res in results.items():
      # Gathers test accuracies under different l2 regularizations.
      reg_accus = [res[num_shots, l2]["test_prec@1"] for l2 in l2_regs]
      # Rank l2 regularizations based on corresponding test accuracies.
      # This is done for every dataset in results.
      # Note that we cannot directly use the ranks at this stage to select
      # the best l2 regularization because it peeks at the test data. Thus,
      # we have next step to reduce self-test-set awareness, for which
      # we currently have two strategies (see details below).
      reg_ranks[dataset_name] = np.argsort(np.argsort(reg_accus))

    for dataset_name in results.keys():
      if how == "all":
        # In this strategy, we average the ranks of l2 regularization across
        # all the datasets. So the best l2 regularization is chosen by all
        # the fewshot datasets available, not just one specific dataset.
        reg_ranks_to_use = np.mean(list(reg_ranks.values()), axis=0)
      elif how == "leave-self-out":
        # In this strategy, we average the ranks of l2 regularization across
        # all the other datasets (avoiding the influence from the current
        # dataset's test accuracy). This strategy reduces self-test-set
        # awareness the most.
        reg_ranks_to_use = np.mean([
            reg_rank for key, reg_rank in reg_ranks.items()
            if key != dataset_name
        ], axis=0)

      best_l2[(dataset_name, num_shots)] = l2_regs[np.argmax(reg_ranks_to_use)]

  return best_l2


class LogRegFewShotEvaluator:
  """Class for few-shot evaluation using logistic regression.

  This code is based on an older version of linear FewShotEvaluator
  at https://github.com/google-research/big_vision. Instead of a linear
  regression, we use here an approach based on a multiclass logistic regression,
  so that we can output a categorical probabilistic distribution, which enables
  reliability related metric evaluations (e.g., ECE or NLL).
  """

  def __init__(self,
               representation_fn,
               fewshot_config,
               batch_size=None,
               ens_size=1,
               l2_selection_scheme="leave-self-out"):
    self.shots = fewshot_config["shots"]
    self.l2_regs = fewshot_config["l2_regs"]
    batch_size = batch_size or fewshot_config.get("batch_size")  # bwd compat.
    self.batch_size = batch_size
    self.repr_fn = representation_fn
    self.pp_tr = fewshot_config["pp_train"]
    self.pp_te = fewshot_config["pp_eval"]
    self.walk_first = fewshot_config["walk_first"]
    self._datasets = {}  # This will be our cache for lazy loading.
    self.prefix_main = fewshot_config.get("prefix_main", "a/")
    self.prefix_lvl1 = fewshot_config.get("prefix_lvl1", "z/")
    self.prefix_lvl2 = fewshot_config.get("prefix_lvl2", "zz/")
    self.seed = fewshot_config.get("seed", 0)

    self.ood_datasets = fewshot_config.ood_datasets
    self.ens_size = ens_size
    self.l2_selection_scheme = l2_selection_scheme

  # Setup input pipeline.
  def _get_dataset(self, dataset, train_split, test_split):
    """Lazy-loads given dataset."""
    key = (dataset, train_split, test_split)
    try:
      return self._datasets[key]
    except KeyError:
      # TODO(kehanghan): Switch to `input_utils.get_data` instead of this
      # non-deterministic `input_pipeline.make_for_inference`.
      train_ds, batches_tr = input_pipeline.make_for_inference(  # pytype: disable=wrong-keyword-args
          dataset=dataset,
          split=train_split,
          batch_size=self.batch_size,
          # TODO(kehanghan): Switch to `clu.preprocess_spec` instead of this
          # non-deterministic `pp_builder.get_preprocess_fn`.
          # clu example usage: preprocess_spec.parse(spec=self.pp_tr,
          # available_ops=preprocess_utils.all_ops()).
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_tr))
      test_ds, batches_te = input_pipeline.make_for_inference(  # pytype: disable=wrong-keyword-args
          dataset=dataset,
          split=test_split,
          batch_size=self.batch_size,
          preprocess_fn=pp_builder.get_preprocess_fn(self.pp_te))
      num_classes = tfds.builder(dataset).info.features["label"].num_classes
      return self._datasets.setdefault(
          key, (train_ds, batches_tr, test_ds, batches_te, num_classes))

  def _get_repr(self, params, data, steps, **kwargs):
    """Compute representation for the whole dataset."""
    pre_logits_list = []
    labels_list = []
    for batch, _ in zip(input_pipeline.start_input_pipeline(data, 0),
                        range(steps)):
      pre_logits, labels, mask = self.repr_fn(
          params, batch["image"], batch["label"], batch["_mask"], **kwargs)
      # Shapes at this point are:
      # pre_logits: (hosts, devices, global_batch, features)
      # labels: (hosts, devices, global_batch)
      # mask: (hosts, devices, global_batch)
      mask = np.array(mask[0]).astype(bool)
      pre_logits_list.append(np.array(pre_logits[0])[mask])
      labels_list.append(np.array(labels[0])[mask])
    pre_logits = np.concatenate(pre_logits_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)

    logging.info(f"[fewshot]: pre_logits: {pre_logits.shape}, "  # pylint: disable=logging-fstring-interpolation
                 f"labels: {labels.shape}")
    return pre_logits, labels

  def compute_fewshot_metrics(self, params, dataset, train_split, test_split,
                              eval_ood_detection, ood_dataset, ood_train_split,
                              ood_test_split, **kw):
    """Compute few-shot metrics on one dataset."""
    train_ds, steps_tr, test_ds, steps_te, num_classes = self._get_dataset(
        dataset, train_split, test_split)
    logging.info("[fewshot][%s]: Precomputing train (%s)", dataset, train_split)
    repr_train, labels_train = self._get_repr(params, train_ds, steps_tr, **kw)
    logging.info("[fewshot][%s]: Precomputing test (%s)", dataset, test_split)
    repr_test, labels_test = self._get_repr(params, test_ds, steps_te, **kw)
    # For OOD detection.
    if eval_ood_detection:
      _, _, ood_test_ds, steps_ood_te, _ = self._get_dataset(ood_dataset,
                                                             ood_train_split,
                                                             ood_test_split)
      logging.info("[fewshot][%s]: Precomputing ood (%s)", dataset,
                   ood_test_split)
      repr_ood_test, _ = self._get_repr(params, ood_test_ds, steps_ood_te, **kw)

    logging.info("[fewshot][%s]: solving systems", dataset)

    # Collect where we have samples of which classes.
    class_indices = [np.where(labels_train == cls_i)[0]
                     for cls_i in range(num_classes)]

    results = {}
    for shots in self.shots:
      all_idx = [indices[:shots] for indices in class_indices]
      all_idx = np.concatenate(all_idx, axis=0)
      y = labels_train[all_idx]

      # Standardizes the input.
      mean = jnp.mean(repr_train[all_idx], axis=0, keepdims=True)
      std = jnp.std(repr_train[all_idx], axis=0, keepdims=True) + 1e-5
      x = (repr_train[all_idx] - mean) / std
      x_test = (repr_test - mean) / std
      if eval_ood_detection:
        x_ood_test = (repr_ood_test - mean) / std
      else:
        x_ood_test = None

      # Prepares train_and_eval for parallelization via multiprocessing.pool.
      partial_train_and_eval = functools.partial(
          train_and_eval,
          x=x,
          y=y,
          ens_size=self.ens_size,
          x_test=x_test,
          labels_test=labels_test,
          x_ood_test=x_ood_test,
          eval_ood_detection=eval_ood_detection)

      worker_count = len(self.l2_regs)
      with multiprocessing.pool.ThreadPool(worker_count) as pool:
        output = list(pool.map(partial_train_and_eval, self.l2_regs))

      for l2_reg, metric_results in output:
        results[shots, l2_reg] = metric_results

    return results

  def run_all(self, params, datasets, **kwargs):
    """Compute summary over all `datasets` that comes from config."""
    results = {}
    for name, dataset_args in datasets.items():
      eval_ood_detection = name in self.ood_datasets
      if eval_ood_detection:
        ood_dataset_args = self.ood_datasets[name]
      else:
        ood_dataset_args = (None, None, None)
      results[name] = self.compute_fewshot_metrics(params, *dataset_args,
                                                   eval_ood_detection,
                                                   *ood_dataset_args, **kwargs)

    best_l2 = select_best_l2_reg(
        results, self.shots, self.l2_regs, how=self.l2_selection_scheme)

    return results, best_l2

  def walk_results(self, fn, results, best_l2):
    """Call `fn` with a descriptive string and the result on all results."""
    # First, go through each individual result:
    for ds_shortname, result in results.items():
      for (shots, l2), metric_results in result.items():
        for name, value in metric_results.items():
          fn(f"{self.prefix_lvl2}{ds_shortname}_{shots}shot_l2={l2}_{name}",
             value)
    # Second, report each dataset/shot with the single "globally" best l2.
    for name_shots, l2 in best_l2.items():
      ds_shortname, shots = name_shots
      fn(f"{self.prefix_lvl1}best_l2_for_{ds_shortname}_{shots}shot", l2)
      metric_results = results[ds_shortname][shots, l2]
      for metric_name, value in metric_results.items():
        fn(f"{self.prefix_lvl1}{ds_shortname}_{shots}shot_{metric_name}", value)
    # And a highlight, if desired:
    if self.walk_first:
      ds_shortname, shots = self.walk_first
      l2 = best_l2[(ds_shortname, shots)]
      metric_results = results[ds_shortname][shots, l2]
      for metric_name, value in metric_results.items():
        fn(f"{self.prefix_main}{ds_shortname}_{shots}shot_{metric_name}", value)
