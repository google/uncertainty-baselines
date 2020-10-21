#import os.path

#from absl import app
#from absl import flags
#from absl import logging

import numpy as np
import tensorflow as tf
#import uncertainty_baselines as ub
import uncertainty_metrics as um

class BrierScore(tf.keras.metrics.Mean):

    def update_state(self, y_true, y_pred, sample_weight=None):
        brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
        super(BrierScore, self).update_state(brier_score)

# class summation(tf.keras.metrics.Mean):
#     def update_state(self, y_true, y_pred, sample_weight=None): 
#         #brier_score = um.brier_score(labels=y_true, probabilities=y_pred)
#         super(summation, self).update_state(y_pred)
     
def one_vs_all_loss_fn(dm_alpha: float = 1., from_logits: bool = True):
    """Requires from_logits=True to calculate correctly."""
    if not from_logits:
        raise ValueError('One-vs-all loss requires inputs to the '
                         'loss function to be logits, not probabilities.')

    def one_vs_all_loss(labels: tf.Tensor, logits: tf.Tensor):
        r"""Implements the one-vs-all loss function.

        As implemented in https://arxiv.org/abs/1709.08716, multiplies the output
        logits by dm_alpha (if using a distance-based formulation) before taking K
        independent sigmoid operations of each class logit, and then calculating the
        sum of the log-loss across classes. The loss function is calculated from the
        K sigmoided logits as follows -

        \mathcal{L} = \sum_{i=1}^{K} -\mathbb{I}(y = i) \log p(\hat{y}^{(i)} | x)
        -\mathbb{I} (y \neq i) \log (1 - p(\hat{y}^{(i)} | x))

        Args:
          labels: Integer Tensor of dense labels, shape [batch_size].
          logits: Tensor of shape [batch_size, num_classes].

        Returns:
          A scalar containing the mean over the batch for one-vs-all loss.
        """
        eps = 1e-6
        logits = logits * dm_alpha
        n_classes = tf.cast(logits.shape[1], tf.float32)

        one_vs_all_probs = tf.math.sigmoid(logits)
        labels = tf.cast(tf.squeeze(labels), tf.int32)
        row_ids = tf.range(tf.shape(one_vs_all_probs)[0], dtype=tf.int32)
        idx = tf.stack([row_ids, labels], axis=1)

        # Shape of class_probs is [batch_size,].
        class_probs = tf.gather_nd(one_vs_all_probs, idx)

        loss = (
            tf.reduce_mean(tf.math.log(class_probs + eps)) +
            n_classes * tf.reduce_mean(tf.math.log(1. - one_vs_all_probs + eps)) -
            tf.reduce_mean(tf.math.log(1. - class_probs + eps)))

        return -loss

    return one_vs_all_loss

