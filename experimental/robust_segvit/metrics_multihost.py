"""Calculate ood metrics across hosts."""
from typing import Any, Optional, Dict

import jax
import jax.numpy as jnp
import tensorflow as tf
from jax.experimental import multihost_utils
from ood_metrics import get_ood_score
from ood_metrics import get_score


def keras_auc_to_arrays(keras_auc_object):
  """Pull out arrays from keras roc object."""
  # The thresholds used are determinisitc, so we need not store them.
  tp = jnp.asarray(keras_auc_object.true_positives)
  fp = jnp.asarray(keras_auc_object.false_positives)
  tn = jnp.asarray(keras_auc_object.true_negatives)
  fn = jnp.asarray(keras_auc_object.false_negatives)
  return tp, fp, tn, fn


def arrays_to_keras_auc(tp, fp, tn, fn, keras_auc_object):
  """Assign confusion matrix arrays to a keras_auc_object."""
  keras_auc_object.true_positives.assign(tp)
  keras_auc_object.false_positives.assign(fp)
  keras_auc_object.true_negatives.assign(tn)
  keras_auc_object.false_negatives.assign(fn)
  return keras_auc_object


def combine_states(all_auc_states, num_thresholds=200):
  # jax can take in trees of arrays, tuple is considered a tree so we can
  # unpack it here.
  # each array here has dimensions #host x shape

  all_tp, all_fp, all_tn, all_fn = all_auc_states

  assert all_tp.shape == (jax.process_count(), num_thresholds)
  assert all_fp.shape == (jax.process_count(), num_thresholds)
  assert all_tn.shape == (jax.process_count(), num_thresholds)
  assert all_fn.shape == (jax.process_count(), num_thresholds)

  tp = jnp.sum(all_tp, 0)
  fp = jnp.sum(all_fp, 0)
  tn = jnp.sum(all_tn, 0)
  fn = jnp.sum(all_fn, 0)

  return tp, fp, tn, fn


class ComputeAUCMetric:
  """Calculate auc metrics across multiple hosts."""
  def __init__(self, curve, num_thresholds=200, from_logits=False):
    self.curve = curve
    self.num_thresholds = num_thresholds
    self.from_logits = from_logits
    self.auc = tf.keras.metrics.AUC(curve=self.curve,
                                    from_logits=self.from_logits,
                                    num_thresholds=self.num_thresholds)

  def calculate_and_update_scores(self, logits, label, sample_weight):
      self.auc.update_state(label, logits, sample_weight=sample_weight)

  def gather_metrics(self):
    auc_state = keras_auc_to_arrays(self.auc)

    # Gather the data across all hosts.
    all_auc_states = multihost_utils.process_allgather(auc_state)

    # Below we pick the first device.
    self.auc = arrays_to_keras_auc(*combine_states(all_auc_states,
                                                   num_thresholds=self.num_thresholds),
                                   self.auc)

    return self.auc.result().numpy()


class ComputeOODAUCMetric:
  """Calculate auc metrics across multiple hosts."""
  def __init__(self, curve, num_thresholds=200):
    self.curve = curve
    self.num_thresholds = num_thresholds
    self.from_logits = False
    self.auc = tf.keras.metrics.AUC(curve=self.curve,
                                    from_logits=self.from_logits,
                                    num_thresholds=self.num_thresholds)

  def calculate_and_update_scores(self, logits, label, sample_weight, *kwargs):
    ood_score = get_ood_score(logits, *kwargs)
    self.auc.update_state(label, ood_score, sample_weight=sample_weight)

  def gather_metrics(self):
    auc_state = keras_auc_to_arrays(self.auc)

    # Gather the data across all hosts.
    all_auc_states = multihost_utils.process_allgather(auc_state)

    # Below we pick the first device.
    self.auc = arrays_to_keras_auc(*combine_states(all_auc_states,
                                                   num_thresholds=self.num_thresholds),
                                   self.auc)

    return self.auc.result().numpy()


class ComputeScoreAUCMetric:
  """Calculate score based auc metrics across multiple hosts."""
  def __init__(self, curve, num_thresholds=200, summation_method='interpolation',thresholds=None):
    self.curve = curve
    self.num_thresholds = num_thresholds
    self.from_logits = False
    self.summation_method = summation_method
    self.thresholds = thresholds
    self.auc = tf.keras.metrics.AUC(curve=self.curve,
                                    from_logits=self.from_logits,
                                    num_thresholds=self.num_thresholds,
                                    summation_method=self.summation_method,
                                    thresholds=self.thresholds)

  def calculate_and_update_scores(self, logits, label, sample_weight, **kwargs):
    " label 1 for ood pixel and 0 is otherwise "
    conf = - 1 * get_score(logits=logits, **kwargs)

    # skip images where all the pixels are ood or there are no ood pixels
    all_pixel_ood = jnp.sum(label*sample_weight) == 1
    no_pixel_ood = jnp.sum(label*sample_weight) == 0

    if not(all_pixel_ood) and not(no_pixel_ood):
      self.auc.update_state(label, conf, sample_weight=sample_weight)

  def gather_metrics(self):
    auc_state = keras_auc_to_arrays(self.auc)

    # Gather the data across all hosts.
    all_auc_states = multihost_utils.process_allgather(auc_state)

    # Below we pick the first device.
    self.auc = arrays_to_keras_auc(*combine_states(all_auc_states,
                                                   num_thresholds=self.num_thresholds),
                                   self.auc)

    return self.auc.result().numpy()
