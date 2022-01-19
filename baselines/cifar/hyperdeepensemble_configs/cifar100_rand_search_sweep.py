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

"""Hyperparameter sweep for initial random search of hyper-deep ensemble."""

TRAIN_PROPORTION = 0.95
TRAIN_SET_SIZE = 50000 * TRAIN_PROPORTION
DEFAULT_L2 = 1e-4
L2_RANGE = [1e-2 / TRAIN_SET_SIZE, 1e2 / TRAIN_SET_SIZE]
LS_RANGE = [1e-5, 2e-5]
NUM_RUNS = 100


def _get_domain(hyper):
  """Get hyperparemeter search domain."""

  # include default model
  hps = [
      hyper.fixed('input_conv_l2', DEFAULT_L2, length=1),
      hyper.fixed('group_1_conv_l2', DEFAULT_L2, length=1),
      hyper.fixed('group_2_conv_l2', DEFAULT_L2, length=1),
      hyper.fixed('group_3_conv_l2', DEFAULT_L2, length=1),
      hyper.fixed('dense_kernel_l2', DEFAULT_L2, length=1),
      hyper.fixed('dense_bias_l2', DEFAULT_L2, length=1),
      hyper.fixed('label_smoothing', 0, length=1),
      hyper.fixed('seed', 0, length=1),
      hyper.fixed('bn_l2', DEFAULT_L2, length=1),
      hyper.fixed('train_proportion', TRAIN_PROPORTION, length=1),
      hyper.fixed('l2', None, length=1),  # disable global l2
      hyper.fixed('dataset', 'cifar100', length=1),
  ]
  hyperparameters1 = hyper.zipit(hps, length=1)

  # sample random hyperparameters
  l2_interval = hyper.interval(L2_RANGE[0], L2_RANGE[1])
  ls_interval = hyper.interval(LS_RANGE[0], LS_RANGE[1])
  hps = [
      hyper.loguniform('input_conv_l2', l2_interval),
      hyper.loguniform('group_1_conv_l2', l2_interval),
      hyper.loguniform('group_2_conv_l2', l2_interval),
      hyper.loguniform('group_3_conv_l2', l2_interval),
      hyper.loguniform('dense_kernel_l2', l2_interval),
      hyper.loguniform('dense_bias_l2', l2_interval),
      hyper.loguniform('label_smoothing', ls_interval),
      hyper.uniform('seed', hyper.discrete(range(1, 10000))),
      hyper.fixed('bn_l2', DEFAULT_L2, length=NUM_RUNS - 1),
      hyper.fixed('train_proportion', TRAIN_PROPORTION, length=NUM_RUNS - 1),
      hyper.fixed('l2', None, length=NUM_RUNS - 1),  # disable global l2
      hyper.fixed('dataset', 'cifar100', length=NUM_RUNS - 1),
  ]
  hyperparameters2 = hyper.zipit(hps, length=NUM_RUNS - 1)

  return hyper.chainit([hyperparameters1, hyperparameters2])


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  return _get_domain(hyper)
