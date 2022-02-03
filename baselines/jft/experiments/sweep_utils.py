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

"""Utilities to sweep over settings (e.g., downstream datasets)."""

import ml_collections


def flatten(d, parent_key='', sep='.'):
  """Flattens a potentially nested ConfigDict into one dictionary."""
  items = []
  for k, v in d.items():
    new_key = parent_key + sep + k if parent_key else k
    if isinstance(v, ml_collections.ConfigDict):
      items.extend(flatten(v, new_key, sep=sep).items())
    else:
      items.append((new_key, v))
  return dict(items)


def fixed(hyper, config):
  config = flatten(config)
  return [hyper.fixed(f'config.{k}', v, length=1) for k, v in config.items()]


def cifar10(hyper, size=384, steps=10_000, warmup=500):
  """A fixed sweep for CIFAR-10 specific settings."""
  name = 'cifar10'
  n_cls = 10
  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({n_cls}, key="label", key_result="labels")'
  pp_common += '|keep(["image", "labels", "id"])'
  pp_train = f'decode|inception_crop({size})|flip_lr' + pp_common
  pp_eval = f'decode|resize({size})' + pp_common

  config = ml_collections.ConfigDict()
  config.dataset = name
  config.train_split = 'train[:98%]'
  config.pp_train = pp_train
  # TODO(trandustin): Add support for multiple validation splits.
  config.val_split = 'train[98%:]'
  config.test_split = 'test'
  config.pp_eval = pp_eval
  config.num_classes = n_cls
  config.lr = ml_collections.ConfigDict()
  config.lr.warmup_steps = warmup
  config.total_steps = steps

  # OOD evaluation
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_datasets = ['cifar100', 'svhn_cropped']
  config.ood_num_classes = [100, 10]
  config.ood_split = 'test'
  config.ood_methods = ['msp', 'entropy', 'maha', 'rmaha']
  pp_eval_ood = []
  for num_classes in config.ood_num_classes:
    if num_classes > config.num_classes:
      # Note that evaluation_fn ignores the entries with all zero labels for
      # evaluation. When num_classes > n_cls, we should use onehot{num_classes},
      # otherwise the labels that are greater than n_cls will be encoded with
      # all zeros and then be ignored.
      pp_eval_ood.append(
          pp_eval.replace(f'onehot({config.num_classes}',
                          f'onehot({num_classes}'))
    else:
      pp_eval_ood.append(pp_eval)
  config.pp_eval_ood = pp_eval_ood

  # CIFAR-10H eval
  config.eval_on_cifar_10h = True
  config.pp_eval_cifar_10h = f'decode|resize({size})|value_range(-1, 1)|keep(["image", "labels"])'
  return fixed(hyper, config)


def cifar100(hyper, size=384, steps=10_000, warmup=500):
  """A fixed sweep for CIFAR-100 specific settings."""
  name = 'cifar100'
  n_cls = 100
  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({n_cls}, key="label", key_result="labels")'
  pp_common += '|keep(["image", "labels"])'
  pp_train = f'decode|inception_crop({size})|flip_lr' + pp_common
  pp_eval = f'decode|resize({size})' + pp_common

  config = ml_collections.ConfigDict()
  config.dataset = name
  config.train_split = 'train[:98%]'
  config.pp_train = pp_train
  # TODO(trandustin): Add ability to support multiple validation splits.
  config.val_split = 'train[98%:]'
  config.test_split = 'test'
  config.pp_eval = pp_eval
  config.num_classes = n_cls
  config.lr = ml_collections.ConfigDict()
  config.lr.warmup_steps = warmup
  config.total_steps = steps

  # OOD evaluation
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_datasets = ['cifar10', 'svhn_cropped']
  config.ood_num_classes = [10, 10]
  config.ood_split = 'test'
  config.ood_methods = ['msp', 'entropy', 'maha', 'rmaha']
  pp_eval_ood = []
  for num_classes in config.ood_num_classes:
    if num_classes > config.num_classes:
      # Note that evaluation_fn ignores the entries with all zero labels for
      # evaluation. When num_classes > n_cls, we should use onehot{num_classes},
      # otherwise the labels that are greater than n_cls will be encoded with
      # all zeros and then be ignored.
      pp_eval_ood.append(
          pp_eval.replace(f'onehot({config.num_classes}',
                          f'onehot({num_classes}'))
    else:
      pp_eval_ood.append(pp_eval)
  config.pp_eval_ood = pp_eval_ood
  return fixed(hyper, config)


def imagenet(hyper, size=384, steps=20_000, warmup=500):
  """A fixed sweep for ImageNet specific settings."""
  name = 'imagenet2012'
  n_cls = 1000
  pp_common = '|value_range(-1, 1)'
  pp_common += '|onehot(1000, key="{lbl}", key_result="labels")'
  pp_common += '|keep(["image", "labels"])'
  pp_train = f'decode_jpeg_and_inception_crop({size})|flip_lr'
  pp_train += pp_common.format(lbl='label')
  pp_eval = f'decode|resize({size})' + pp_common.format(lbl='label')

  config = ml_collections.ConfigDict()
  config.dataset = name
  config.train_split = 'train[:99%]'
  config.pp_train = pp_train
  # TODO(trandustin): Add ability to support multiple validation splits.
  config.val_split = 'train[99%:]'
  config.test_split = 'validation'
  config.pp_eval = pp_eval
  config.num_classes = n_cls
  config.lr = ml_collections.ConfigDict()
  config.lr.warmup_steps = warmup
  config.total_steps = steps

  # OOD evaluation
  config.eval_on_imagenet_real = True
  config.pp_eval_imagenet_real = f'decode|resize({size})|value_range(-1, 1)|keep(["image", "labels"])'  # pylint: disable=line-too-long
  return fixed(hyper, config)
