# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""Diversity Utilities."""

import functools
import tensorflow.compat.v2 as tf
import be_utils  # local file import
import diversity_metrics  # local file import


def fast_weights_similarity(trainable_variables, similarity_metric, dpp_kernel):
  """Computes similarity_metric on fast weights."""
  # Fast weights are specific to each ensemble member. These are the 'r' and
  # 's' vectors in the BatchEnsemble paper (https://arxiv.org/abs/2002.06715).
  if similarity_metric == 'cosine':
    similarity_fn = diversity_metrics.pairwise_cosine_similarity
  elif similarity_metric == 'dpp_logdet':
    similarity_fn = functools.partial(
        diversity_metrics.dpp_negative_logdet, kernel=dpp_kernel)
  else:
    raise ValueError('Could not recognize similarity_metric = {} : not in '
                     '[cosine, dpp_logdet]'.format(similarity_metric))
  fast_weights = [
      var for var in trainable_variables
      if not be_utils.is_batch_norm(var) and
      ('alpha' in var.name or 'gamma' in var.name)
  ]
  similarity_penalty = tf.reduce_mean(
      [tf.cast(similarity_fn(var), tf.float32) for var in fast_weights])
  return similarity_penalty


def outputs_similarity(ensemble_outputs_tensor, similarity_metric, dpp_kernel):
  """Computes similarity_metric on ensemble_outputs_tensor."""
  if similarity_metric == 'cosine':
    similarity_fn = functools.partial(
        diversity_metrics.pairwise_cosine_similarity, normalize=False)
  elif similarity_metric == 'dpp_logdet':
    similarity_fn = functools.partial(
        diversity_metrics.dpp_negative_logdet,
        kernel=dpp_kernel,
        normalize=False)
  else:
    raise ValueError('Could not recognize similarity_metric = {} : not in '
                     '[cosine, dpp_logdet]'.format(similarity_metric))
  similarity_penalty_list = tf.map_fn(
      similarity_fn, ensemble_outputs_tensor, dtype=tf.float32)
  similarity_penalty = tf.reduce_mean(similarity_penalty_list)
  return similarity_penalty


def scaled_similarity_loss(diversity_coeff,
                           diversity_schedule,
                           step,
                           similarity_metric,
                           dpp_kernel,
                           trainable_variables=None,
                           use_output_similarity=False,
                           ensemble_outputs_tensor=None):
  """Computes similarity_metric and scales according to diversity_schedule."""
  similarity_coeff = 0.
  if diversity_coeff > 0:
    if use_output_similarity:
      similarity_loss = outputs_similarity(ensemble_outputs_tensor,
                                           similarity_metric, dpp_kernel)
    else:
      similarity_loss = fast_weights_similarity(trainable_variables,
                                                similarity_metric, dpp_kernel)
    similarity_coeff = diversity_schedule(step)
  else:
    similarity_loss = 0.

  return similarity_coeff, similarity_loss


class ExponentialDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses exponential decay."""

  def __init__(self,
               initial_coeff,
               start_epoch,
               decay_epoch,
               steps_per_epoch,
               decay_rate,
               staircase=True,
               name=None):
    self.initial_coeff = initial_coeff
    self.start_epoch = start_epoch
    self.decay_epoch = decay_epoch
    self.steps_per_epoch = steps_per_epoch
    self.decay_rate = decay_rate
    self.staircase = staircase
    self.dtype = tf.float32

    self.coeff_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=self.initial_coeff,
        decay_steps=self.decay_epoch * self.steps_per_epoch,
        decay_rate=self.decay_rate,
        staircase=self.staircase)

  def __call__(self, step):
    starting_iteration = self.steps_per_epoch * self.start_epoch
    starting_iteration = tf.cast(starting_iteration, self.dtype)
    global_step = tf.cast(step, self.dtype)
    recomp_iteration = global_step - starting_iteration + 1.
    decayed_coeff = self.coeff_scheduler(recomp_iteration)
    # This is an autograph-friendly alternative to checking Tensorflow booleans
    # in eager mode.
    scale = tf.minimum(
        tf.maximum(tf.cast(recomp_iteration, self.dtype), 0.), 1.)
    return scale * decayed_coeff

  def get_config(self):
    return {
        'initial_coeff': self.initial_coeff,
        'start_epoch': self.start_epoch,
        'decay_epoch': self.decay_epoch,
        'steps_per_epoch': self.steps_per_epoch,
        'decay_rate': self.decay_rate,
        'staircase': self.staircase,
        'name': self.name
    }


class LinearAnnealing(tf.keras.optimizers.schedules.LearningRateSchedule):
  """A LearningRateSchedule that uses linear scaling."""

  def __init__(self,
               initial_coeff,
               annealing_epochs,
               steps_per_epoch,
               name=None):
    super(LinearAnnealing, self).__init__()
    self.initial_coeff = initial_coeff
    self.annealing_epochs = annealing_epochs
    self.steps_per_epoch = steps_per_epoch
    self.name = name

  def __call__(self, step):
    annealing_epochs = tf.cast(self.annealing_epochs, tf.float32)
    steps_per_epoch = tf.cast(self.steps_per_epoch, tf.float32)
    global_step_recomp = tf.cast(step, tf.float32)
    initial_coeff = tf.cast(self.initial_coeff, tf.float32)
    p = (global_step_recomp + 1) / (steps_per_epoch * annealing_epochs)
    return tf.math.multiply(initial_coeff, tf.math.minimum(1., p))

  def get_config(self):
    return {
        'initial_coeff': self.initial_coeff,
        'annealing_epochs': self.annealing_epochs,
        'steps_per_epoch': self.steps_per_epoch,
        'name': self.name
    }
