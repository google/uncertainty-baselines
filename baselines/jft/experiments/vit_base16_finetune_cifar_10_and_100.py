# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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
r"""ViT-B/16 finetuning on CIFAR 10 and CIFAR 100 using hypersweep.

"""
# pylint: enable=line-too-long

import ml_collections



def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.dataset = 'cifar10'
  config.val_split = 'train[98%:]'
  config.train_split = 'train[:98%]'
  config.num_classes = 10

  # OOD eval
  # ood_split is the data split for both the ood_dataset and the dataset.
  config.ood_dataset = 'cifar100'
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
  config.eval_on_cifar_10h = False
  config.pp_eval_cifar_10h = ''

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
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 768
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 3072
  config.model.transformer.num_heads = 12
  config.model.transformer.num_layers = 12
  config.model.classifier = 'token'  # Or 'gap'

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

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

  config.args = {}
  return config


def get_sweep(hyper):
  """Sweeps over datasets."""

  # pylint: disable=g-long-lambda
  c100 = lambda **kw: task(
      hyper,
      'cifar100',
      'train[:98%]',
      'train[98%:]',
      'cifar10',
      n_cls=100,
      **kw)
  c10 = lambda **kw: task(
      hyper,
      'cifar10',
      'train[:98%]',
      'train[98%:]',
      'cifar100',
      n_cls=10,
      **kw)
  # pylint: enable=g-long-lambda

  tasks = hyper.chainit([
      # Same sizes as in default BiT-HyperRule, for models that supports hi-res.
      c100(size=384, steps=10_000, warmup=500),
      c100(size=384, steps=10_000, warmup=500, train_data_aug=False),
      c10(size=384, steps=10_000, warmup=500),
      c10(size=384, steps=10_000, warmup=500, train_data_aug=False),
  ])

  return hyper.product([
      hyper.sweep('config.model_init', [MODEL_INIT_I21K, MODEL_INIT_JFT]),
      tasks,
      hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
  ])


def fixed(hyper, **kw):
  return hyper.zipit(
      [hyper.fixed(f'config.{k}', v, length=1) for k, v in kw.items()])


def task(hyper,
         name,
         train,
         val,
         ood_name,
         n_cls,
         steps,
         warmup,
         size,
         train_data_aug=True):
  """Vision task with val and test splits."""
  common = '|value_range(-1, 1)'
  common += f'|onehot({n_cls}, key="label", key_result="labels")'
  common += '|keep(["image", "labels"])'
  pp_eval = f'decode|resize({size})' + common

  if train_data_aug:
    pp_train = f'decode|inception_crop({size})|flip_lr' + common
  else:
    pp_train = f'decode|resize({size})' + common

  task_hyper = {
      'dataset': name,
      'train_split': train,
      'pp_train': pp_train,
      'val_split': val,
      'ood_dataset': ood_name,
      'pp_eval': pp_eval,
      'num_classes': n_cls,
      'lr.warmup_steps': warmup,
      'total_steps': steps,
  }

  if name == 'cifar10':
    # CIFAR-10H eval
    eval_on_cifar_10h = True
    pp_eval_cifar_10h = f'resize({size})' + '|value_range(-1, 1)' + (
        '|keep(["image", "labels"])')
    task_hyper.update({
        'eval_on_cifar_10h': eval_on_cifar_10h,
        'pp_eval_cifar_10h': pp_eval_cifar_10h
    })

  return fixed(hyper, **task_hyper)
