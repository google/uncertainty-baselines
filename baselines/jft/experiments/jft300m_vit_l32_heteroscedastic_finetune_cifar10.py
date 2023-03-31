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

# pylint: disable=line-too-long
r"""Finetune a ViT-L/32 heteroscedastic model on CIFAR-10.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.


def get_sweep(hyper):
  temp_grid = [0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
  return hyper.product([
      hyper.sweep('config.model.temperature', temp_grid),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
  ])


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.dataset = 'cifar10'
  config.train_split = 'train[:98%]'
  config.val_split = 'train[98%:]'
  config.test_split = 'test'
  config.num_classes = 10

  BATCH_SIZE = 512  # pylint: disable=invalid-name
  config.batch_size = BATCH_SIZE

  config.total_steps = 10_000

  INPUT_RES = 384  # pylint: disable=invalid-name
  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = f'decode|inception_crop({INPUT_RES})|flip_lr' + pp_common
  config.pp_eval = f'decode|resize({INPUT_RES})' + pp_common
  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  # OOD eval
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_datasets = ['cifar100', 'svhn_cropped']
  config.ood_num_classes = [100, 10]
  config.ood_split = 'test'
  config.ood_methods = ['msp', 'entropy', 'maha', 'rmaha', 'mlogit']
  pp_eval_ood = []
  for num_classes in config.ood_num_classes:
    if num_classes > config.num_classes:
      # Note that evaluation_fn ignores the entries with all zero labels for
      # evaluation. When num_classes > n_cls, we should use onehot{num_classes},
      # otherwise the labels that are greater than n_cls will be encoded with
      # all zeros and then be ignored.
      pp_eval_ood.append(
          config.pp_eval.replace(f'onehot({config.num_classes}',
                                 f'onehot({num_classes}'))
    else:
      pp_eval_ood.append(config.pp_eval)
  config.pp_eval_ood = pp_eval_ood

  # CIFAR-10H eval
  config.eval_on_cifar_10h = True
  config.pp_eval_cifar_10h = f'decode|resize({INPUT_RES})|value_range(-1, 1)|keep(["image", "labels"])'

  # Imagenet ReaL eval
  config.eval_on_imagenet_real = False

  config.log_training_steps = 500
  config.log_eval_steps = 2000
  config.checkpoint_steps = 2500
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  # pre-trained model ckpt file
  # !!!  The below section should be modified per experiment
  config.model_init = '/path/to/pretrained_model_ckpt.npz'
  # Model definition to be copied from the pre-training config
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.classifier = 'token'  # Or 'gap'
  config.model.fix_base_model = False

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

  config.reint_head = True

  # Heteroscedastic
  config.model.multiclass = True
  config.model.temperature = 3.0
  config.model.mc_samples = 1000
  config.model.num_factors = 0
  config.model.param_efficient = False
  config.model.return_locs = False  # True -> fine-tune a homoscedastic model

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.001
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'
  return config
