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

"""Hyperparameter sweep to retrain on train+val. the hyper-deep ensemble."""

DEFAULT_L2 = 1e-4
TRAIN_SET_SIZE = 0.95 * 50000

# NOTE: below, we normalize by TRAIN_SET_SIZE because the models from the random
# search used a custom convention for l2, normalized by the train dataset size.

SELECTED_HPS = [
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': DEFAULT_L2,
        'group_1_conv_l2': DEFAULT_L2,
        'group_2_conv_l2': DEFAULT_L2,
        'group_3_conv_l2': DEFAULT_L2,
        'dense_kernel_l2': DEFAULT_L2,
        'dense_bias_l2': DEFAULT_L2,
        'label_smoothing': 0.,
        'seed': 0,
    },
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': 0.012100236214494308 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 10.78423537517882 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 18.538256915276644 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 22.437062519429173 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 0.04348043221587187 / TRAIN_SET_SIZE,
        'dense_bias_l2': 0.4853927238831955 / TRAIN_SET_SIZE,
        'label_smoothing': 0.00001974636306895064,
        'seed': 9979,
    },
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': 1.8733092610235154 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 2.149858832455459 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 0.02283349328860761 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 0.06493892708718176 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 6.729067408174627 / TRAIN_SET_SIZE,
        'dense_bias_l2': 96.88491593762551 / TRAIN_SET_SIZE,
        'label_smoothing': 0.000014114745706823372,
        'seed': 3709,
    },
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': 0.015678701837246456 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 0.4258111048922535 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 7.349544939683454 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 47.917791858938074 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 0.09229319107759451 / TRAIN_SET_SIZE,
        'dense_bias_l2': 0.012796648147884173 / TRAIN_SET_SIZE,
        'label_smoothing': 0.000017578836421073824,
        'seed': 1497,
    },
]


def _get_domain(hyper):
  """Get hyperparemeter search domain."""

  hyperparameters = []
  for hps in SELECTED_HPS:
    hyperparameters_ = [
        hyper.fixed('l2', None, length=1),  # disable global l2
        hyper.fixed('train_proportion', 1.0, length=1),
        hyper.fixed('dataset', 'cifar100', length=1),
        hyper.fixed('train_epochs', 250, length=1),
    ]
    for name, value in hps.items():
      hyperparameters_.append(hyper.fixed(name, value, length=1))
    hyperparameters.append(hyper.product(hyperparameters_))

  return hyper.chainit(hyperparameters)


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  return _get_domain(hyper)
