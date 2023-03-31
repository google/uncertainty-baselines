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

"""Most common few-shot eval configuration."""

import ml_collections


def get_fewshot(batch_size=None, target_resolution=224, resize_resolution=256,
                runlocal=False):
  """Returns a standard-ish fewshot eval configuration."""
  config = ml_collections.ConfigDict()
  if batch_size:
    config.batch_size = batch_size
  config.representation_layer = 'pre_logits'
  config.log_steps = 25_000
  config.datasets = {  # pylint: disable=g-long-ternary
      'birds': ('caltech_birds2011', 'train', 'test'),
      'caltech': ('caltech101', 'train', 'test'),
      'cars': ('cars196:2.1.0', 'train', 'test'),
      'cifar100': ('cifar100', 'train', 'test'),
      'col_hist': ('colorectal_histology', 'train[:2000]', 'train[2000:]'),
      'dtd': ('dtd', 'train', 'test'),
      'imagenet': ('imagenet2012_subset/10pct', 'train', 'validation'),
      'pets': ('oxford_iiit_pet', 'train', 'test'),
      'uc_merced': ('uc_merced', 'train[:1000]', 'train[1000:]'),
  } if not runlocal else {
      'pets': ('oxford_iiit_pet', 'train', 'test'),
  }
  config.ood_datasets = {
      'cifar100': ('cifar10', 'train', 'test'),
      'imagenet': ('places365_small', 'train', 'validation'),
  }
  config.pp_train = f'decode|resize({resize_resolution})|central_crop({target_resolution})|value_range(-1,1)|keep("image", "label")'
  config.pp_eval = config.pp_train
  config.shots = [1, 5, 10, 25]
  config.l2_regs = [2.0 ** i for i in range(-10, 20)]
  # choose either "all" or "leave-self-out' for l2_selection_scheme.
  config.l2_selection_scheme = 'leave-self-out'
  config.walk_first = ('imagenet', 10) if not runlocal else ('pets', 10)

  return config
