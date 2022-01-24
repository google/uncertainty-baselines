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

# pylint: disable=line-too-long
r"""ResNet-50x2 finetuning on CIFAR-10.

"""
# pylint: enable=line-too-long

import ml_collections


def get_sweep(hyper):
  return hyper.product([])


def get_config():
  """Config for CIFAR-10 fine-tuning a BiT ResNet-50x2 pretrained on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.dataset = 'cifar10'
  config.val_split = 'train[98%:]'
  config.train_split = 'train[:98%]'
  config.num_classes = 10

  # OOD eval
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_datasets = ['cifar100', 'svhn_cropped']
  config.ood_split = 'test'
  config.ood_methods = ['msp', 'maha', 'rmaha']

  BATCH_SIZE = 512  # pylint: disable=invalid-name
  config.batch_size = BATCH_SIZE

  config.total_steps = 10_000

  INPUT_RES = 384  # pylint: disable=invalid-name
  pp_common = '|value_range(-1, 1)'
  # pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = f'decode|inception_crop({INPUT_RES})|flip_lr' + pp_common
  config.pp_eval = f'decode|resize({INPUT_RES})' + pp_common

  # CIFAR-10H eval
  config.eval_on_cifar_10h = True
  config.pp_eval_cifar_10h = f'decode|resize({INPUT_RES})|value_range(-1, 1)|keep(["image", "labels"])'

  # Imagenet ReaL eval
  config.eval_on_imagenet_real = False

  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 10
  config.log_eval_steps = 100
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  # pre-trained model ckpt file
  # !!!  The below section should be modified per experiment
  config.model_init = '/path/to/pretrained_model_ckpt.npz'
  # Model definition to be copied from the pre-training config
  config.model = ml_collections.ConfigDict()
  config.model.num_layers = 50
  config.model.width_factor = 2

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'

  # Disable unnecessary CNS TTLs.
  config.ttl = 0
  return config
