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

"""Hyperparameter sweep to retrain on train+val. the hyper-deep ensemble."""

DEFAULT_L2 = 2e-4
TRAIN_SET_SIZE = 0.95 * 50000

# NOTE: below, we normalize by TRAIN_SET_SIZE because the models from the random
# search used a custom convention for l2, normalized by the train dataset size.

SELECTED_HPS = [
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': 8.944952710299972 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 2.1730327525906854 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 2.792603836883228 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 0.04785750379925924 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 1.814794347655776 / TRAIN_SET_SIZE,
        'dense_bias_l2': 0.986256209103742 / TRAIN_SET_SIZE,
        'label_smoothing': 0.0007326829665482376,
        'seed': 107,
    },
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
        'input_conv_l2': 0.020949474836316633 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 0.4758180190149804 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 1.155046932643923 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 0.16979667673847318 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 0.07783651929524404 / TRAIN_SET_SIZE,
        'dense_bias_l2': 0.2004094246704319 / TRAIN_SET_SIZE,
        'label_smoothing': 0.00033195742091665463,
        'seed': 8941,
    },
    {
        'bn_l2': DEFAULT_L2,
        'input_conv_l2': 1.1382448043444013 / TRAIN_SET_SIZE,
        'group_1_conv_l2': 8.564006426332158 / TRAIN_SET_SIZE,
        'group_2_conv_l2': 0.0757505267574009 / TRAIN_SET_SIZE,
        'group_3_conv_l2': 0.02290197146892039 / TRAIN_SET_SIZE,
        'dense_kernel_l2': 0.7979287734324894 / TRAIN_SET_SIZE,
        'dense_bias_l2': 0.6030044362874522 / TRAIN_SET_SIZE,
        'label_smoothing': 0.000010047881330848954,
        'seed': 2035,
    },
]


def _get_domain(hyper):
  """Get hyperparemeter search domain."""

  hyperparameters = []
  for hps in SELECTED_HPS:
    hyperparameters_ = [
        hyper.fixed('l2', None, length=1),  # disable global l2
        hyper.fixed('train_proportion', 1.0, length=1),
        hyper.fixed('dataset', 'cifar10', length=1),
        hyper.fixed('train_epochs', 250, length=1),
    ]
    for name, value in hps.items():
      hyperparameters_.append(hyper.fixed(name, value, length=1))
    hyperparameters.append(hyper.product(hyperparameters_))

  return hyper.chainit(hyperparameters)


def get_sweep(hyper):
  """Returns hyperparameter sweep."""
  return _get_domain(hyper)
