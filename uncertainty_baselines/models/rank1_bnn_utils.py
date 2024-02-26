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

"""Rank-1 BNN Utilities."""
import edward2 as ed
import numpy as np
import tensorflow as tf


def _make_sign_initializer(random_sign_init):
  if random_sign_init > 0:
    return ed.initializers.RandomSign(random_sign_init)
  else:
    return tf.keras.initializers.RandomNormal(mean=1.0,
                                              stddev=-random_sign_init)


def make_initializer(initializer, random_sign_init, dropout_rate):
  """Builds initializer with specific mean and/or stddevs."""
  if initializer == 'trainable_deterministic':
    return ed.initializers.TrainableDeterministic(
        loc_initializer=_make_sign_initializer(random_sign_init))
  elif initializer == 'trainable_half_cauchy':
    stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
    return ed.initializers.TrainableHalfCauchy(
        loc_initializer=_make_sign_initializer(random_sign_init),
        scale_initializer=tf.keras.initializers.Constant(stddev_init),
        scale_constraint='softplus')
  elif initializer == 'trainable_cauchy':
    stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
    return ed.initializers.TrainableCauchy(
        loc_initializer=_make_sign_initializer(random_sign_init),
        scale_initializer=tf.keras.initializers.Constant(stddev_init),
        scale_constraint='softplus')
  elif initializer == 'trainable_normal':
    stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
    return ed.initializers.TrainableNormal(
        mean_initializer=_make_sign_initializer(random_sign_init),
        stddev_initializer=tf.keras.initializers.TruncatedNormal(
            mean=stddev_init, stddev=0.1),
        stddev_constraint='softplus')
  elif initializer == 'trainable_log_normal':
    stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
    return ed.initializers.TrainableLogNormal(
        loc_initializer=_make_sign_initializer(random_sign_init),
        scale_initializer=tf.keras.initializers.TruncatedNormal(
            mean=stddev_init, stddev=0.1),
        scale_constraint='softplus')
  elif initializer == 'trainable_normal_fixed_stddev':
    return ed.initializers.TrainableNormalFixedStddev(
        stddev=tf.sqrt(dropout_rate / (1. - dropout_rate)),
        mean_initializer=_make_sign_initializer(random_sign_init))
  elif initializer == 'trainable_normal_shared_stddev':
    stddev_init = np.log(np.expm1(np.sqrt(dropout_rate / (1. - dropout_rate))))
    return ed.initializers.TrainableNormalSharedStddev(
        mean_initializer=_make_sign_initializer(random_sign_init),
        stddev_initializer=tf.keras.initializers.Constant(stddev_init),
        stddev_constraint='softplus')
  return initializer


def make_regularizer(regularizer, mean, stddev):
  """Builds regularizer with specific mean and/or stddevs."""
  if regularizer == 'normal_kl_divergence':
    return ed.regularizers.NormalKLDivergence(mean=mean, stddev=stddev)
  elif regularizer == 'log_normal_kl_divergence':
    return ed.regularizers.LogNormalKLDivergence(
        loc=tf.math.log(1.), scale=stddev)
  elif regularizer == 'normal_kl_divergence_with_tied_mean':
    return ed.regularizers.NormalKLDivergenceWithTiedMean(stddev=stddev)
  elif regularizer == 'cauchy_kl_divergence':
    return ed.regularizers.CauchyKLDivergence(loc=mean, scale=stddev)
  elif regularizer == 'normal_empirical_bayes_kl_divergence':
    return ed.regularizers.NormalEmpiricalBayesKLDivergence(mean=mean)
  elif regularizer == 'trainable_normal_kl_divergence_stddev':
    return ed.regularizers.TrainableNormalKLDivergenceStdDev(mean=mean)
  return regularizer
