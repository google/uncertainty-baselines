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
r"""Active learn a pretrained ViT-L/32 on CIFAR-10.

Based on: vit_l32_finetune.py and sweep_utils.py

"""
# pylint: enable=line-too-long

import ml_collections
import sweep_utils  # local file import from baselines.jft.experiments


def get_config():
  """Config for finetuning."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # Active learning section
  config.model_type = 'deterministic'  # 'batchensemble'
  config.acquisition_method = ''  # set in sweep
  config.max_training_set_size = 0  # set in sweep
  config.initial_training_set_size = 0  # set in sweep
  config.acquisition_batch_size = 0  # set in sweep
  config.early_stopping_patience = 128

  # Dataset section
  config.dataset = ''  # set in sweep
  config.test_split = ''  # set in sweep
  config.val_split = ''  # set in sweep
  config.train_split = ''  # set in sweep
  config.num_classes = None  # set in swee

  config.batch_size = 256  # half of config's 512 - due to memory issues
  config.total_steps = None  # set in sweep

  config.pp_train = ''  # set in sweep
  config.pp_eval = ''  # set in sweep

  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 5000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  config.model_init = ''  # set in sweep
  config.model_type = 'deterministic'  # 'batchensemble'
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [32, 32]
  config.model.hidden_size = 1024
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.mlp_dim = 4096
  config.model.transformer.num_heads = 16
  config.model.transformer.num_layers = 24
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.classifier = 'token'
  # This is "no head" fine-tuning, which we use by default.
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.003  # set in sweep
  # Turned off for active learning:
  config.lr.warmup_steps = 0
  config.lr.decay_type = 'cosine'
  config.eval_on_cifar_10h = False
  # OOD evaluation. Not used but needed for sweep_utils.
  config.ood_datasets = []
  config.ood_num_classes = []
  config.ood_split = ''
  config.ood_methods = []
  config.pp_eval_ood = []
  config.eval_on_cifar_10h = False
  config.pp_eval_cifar_10h = ''
  config.eval_on_imagenet_real = False
  config.pp_eval_imagenet_real = ''

  return config


def get_sweep(hyper):
  """Sweeps over datasets."""
  # Adapted the sweep over checkpoints from vit_l32_finetune.py.

  sweep_lr = True  # whether to sweep over learning rates
  acquisition_methods = ['uniform']  # ['uniform', 'entropy', 'margin']
  #. ['uniform', 'entropy', 'margin', 'density']
  def sweep_checkpoints(use_jft, sweep_lr=sweep_lr):
    """whether to use JFT-300M or ImageNet-21K settings."""
    checkpoints = ['/path/to/pretrained_model_ckpt.npz']
    if use_jft:
      cifar10_sweep = sweep_utils.cifar10(hyper, steps=1000)
      cifar10_sweep.append(hyper.fixed('config.lr.base', 0.01, length=1))
      cifar10_sweep = hyper.product(cifar10_sweep)

      cifar100_sweep = sweep_utils.cifar100(hyper, steps=1000)
      cifar100_sweep.append(hyper.fixed('config.lr.base', 0.03, length=1))
      cifar100_sweep = hyper.product(cifar100_sweep)

      # Temporarity disable imagenet and places due to OOM.
      # pylint: disable=unused-variable
      imagenet_sweep = sweep_utils.imagenet(hyper, steps=1000)
      imagenet_sweep.append(hyper.fixed('config.lr.base', 0.03, length=1))
      imagenet_sweep = hyper.product(imagenet_sweep)

      places365_sweep = hyper.product([
          hyper.product(sweep_utils.places365_small(hyper, steps=1000)),
          hyper.sweep('config.lr.base', [0.03]),
      ])
      # pylint: enable=unused-variable
    else:
      cifar10_sweep = sweep_utils.cifar10(hyper, steps=1000)
      cifar10_sweep.append(hyper.fixed('config.lr.base', 0.003, length=1))
      cifar10_sweep = hyper.product(cifar10_sweep)

      cifar100_sweep = sweep_utils.cifar100(hyper, steps=1000)
      cifar100_sweep.append(hyper.fixed('config.lr.base', 0.01, length=1))
      cifar100_sweep = hyper.product(cifar100_sweep)

      imagenet_sweep = sweep_utils.imagenet(hyper, steps=1000)
      imagenet_sweep.append(hyper.fixed('config.lr.base', 0.01, length=1))
      imagenet_sweep = hyper.product(imagenet_sweep)

      places365_sweep = hyper.product([
          hyper.product(sweep_utils.places365_small(hyper, steps=1000)),
          hyper.sweep('config.lr.base', [0.01]),
      ])
    if sweep_lr:
      # Apply a learning rate sweep following Table 4 of Vision Transformer
      # paper.
      checkpoints = [checkpoints[0]]
      def set_data_sizes(a, b, c):
        return [
            hyper.fixed('config.initial_training_set_size', a, length=1),
            hyper.fixed('config.max_training_set_size', b, length=1),
            hyper.fixed('config.acquisition_batch_size', c, length=1),
        ]

      data_sizes = [
          hyper.product(set_data_sizes(a, b, c))
          for a, b, c in [(48990, 48991, 1)]
      ]
      cifar10_sweep = hyper.product([
          hyper.product(sweep_utils.cifar10(hyper, steps=1000)),
          hyper.sweep('config.lr.base',
                      [0.06, 0.03, 0.015, 0.01, 0.005, 0.001]),
          hyper.chainit(data_sizes),
      ])

      cifar100_sweep = hyper.product([
          hyper.product(sweep_utils.cifar100(hyper, steps=1000)),
          hyper.sweep('config.lr.base',
                      [0.06, 0.03, 0.015, 0.01, 0.005, 0.001]),
          hyper.chainit(data_sizes),
      ])

      imagenet_sweep = hyper.product([
          hyper.product(sweep_utils.imagenet(hyper)),
          hyper.sweep('config.lr.base', [0.06, 0.03, 0.01, 0.003]),
      ])

      places365_sweep = hyper.product([
          hyper.product(sweep_utils.places365_small(hyper, steps=1000)),
          hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]),
      ])
    return hyper.product([
        hyper.chainit([
            cifar10_sweep,
            cifar100_sweep,
            # places365_sweep,
            # imagenet_sweep,
        ]),
        hyper.sweep('config.model_init', checkpoints),
        hyper.sweep('config.acquisition_method', acquisition_methods),
    ])

  return hyper.chainit(
      [sweep_checkpoints(use_jft) for use_jft in [True, False]])
