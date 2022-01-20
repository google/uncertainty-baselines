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
r"""ViT-L/32 finetuning on CIFAR-10 from upstream batchensemble.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.dataset = 'cifar10'
  config.val_split = 'train[98%:]'
  config.train_split = 'train[:98%]'
  config.num_classes = 10

  BATCH_SIZE = 512  # pylint: disable=invalid-name
  config.batch_size = BATCH_SIZE
  config.batch_size_eval = BATCH_SIZE

  config.total_steps = 10_000

  INPUT_RES = 384  # pylint: disable=invalid-name
  pp_common = '|value_range(-1, 1)'
  # pp_common += f'|onehot({config.num_classes})'
  # To use ancestor 'smearing', use this line instead:
  pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
  pp_common += '|keep(["image", "labels"])'
  config.pp_train = f'decode|inception_crop({INPUT_RES})|flip_lr' + pp_common
  config.pp_eval = f'decode|resize({INPUT_RES})' + pp_common

  # OOD eval
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
  config.model.patch_size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.classifier = 'token'  # Or 'gap'

  # BatchEnsemble parameters.
  config.model_name = 'PatchTransformerBE'
  config.model.transformer.be_layers = (21, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = 0.5
  config.fast_weight_lr_multiplier = 1.0

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 1e-3
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'

  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('config.model.transformer.random_sign_init', [-0.5, 0.5]),
      hyper.sweep('config.fast_weight_lr_multiplier', [0.5, 1.0, 2.0]),
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
  ])
