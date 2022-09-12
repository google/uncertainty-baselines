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
r"""BE L/32 active learning.

"""
# pylint: enable=line-too-long

import ml_collections
from experiments import sweep_utils  # local file import from baselines.jft


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  config.seed = 0

  # Active learning section
  config.model_type = 'batchensemble'
  config.acquisition_method = ''  # set in sweep
  config.max_training_set_size = 0  # set in sweep
  config.initial_training_set_size = 0  # set in sweep
  config.acquisition_batch_size = 0  # set in sweep
  config.early_stopping_patience = 128
  config.finetune_head_only = False
  config.power_acquisition = False

  config.dataset = ''  # set in sweep
  config.val_split = ''  # set in sweep
  config.train_split = ''  # set in sweep
  config.test_split = ''  # set in sweep
  config.num_classes = None  # set in sweep

  config.batch_size = 512
  config.batch_size_eval = 512
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

  # OOD evaluation. They're all set in the sweep.
  config.ood_datasets = []
  config.ood_num_classes = []
  config.ood_split = ''
  config.ood_methods = []
  config.pp_eval_ood = []
  config.eval_on_cifar_10h = False
  config.pp_eval_cifar_10h = ''
  config.eval_on_imagenet_real = False
  config.pp_eval_imagenet_real = ''

  # Model section
  config.model_init = ''  # set in sweep

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

  # BatchEnsemble parameters.
  config.model.transformer.be_layers = (22, 23)
  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = -0.5
  # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
  # need checkpoints that only apply BE on the FF block.
  config.model.transformer.ensemble_attention = True
  config.fast_weight_lr_multiplier = 1.0

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay
  config.loss = 'softmax_xent'

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 1e-3  # Set in sweep.
  config.lr.warmup_steps = 500  # Set in sweep
  config.lr.decay_type = 'cosine'

  return config


def get_sweep(hyper):
  """Sweep over datasets and relevant hyperparameters."""
  data_size_mode = 'num_classes/2'
  acquisition_methods = [
      'uniform', 'margin'
  ]
  cp_options = ['']

  hparam_sweep = sweep_utils.get_hparam_best('BE')
  if data_size_mode == 'tuning':
    hparam_sweep = sweep_utils.get_hparam_sweep('BE')
    acquisition_methods = ['uniform']
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  model_specs = hyper.sweep('config.model_init', checkpoints)
  def set_data_sizes(a, b, c):
    return [
        hyper.fixed('config.initial_training_set_size', a, length=1),
        hyper.fixed('config.max_training_set_size', b, length=1),
        hyper.fixed('config.acquisition_batch_size', c, length=1),
    ]

  all_data_sizes = sweep_utils.get_data_sizes(data_size_mode)
  all_sweeps = []
  for dataset in ['cifar10', 'cifar100', 'imagenet', 'places365']:
    sweep_func = sweep_utils.get_dataset_sweep()[dataset]
    data_sizes = [
        hyper.product(set_data_sizes(a, b, c))
        for a, b, c in all_data_sizes[dataset]
    ]
    sweep_list = [
        hyper.product(sweep_func(hyper, steps=2000)),
        hyper.chainit(data_sizes)
    ]
    for hparam in [
        'config.lr.base', 'config.fast_weight_lr_multiplier',
        'config.model.transformer.random_sign_init'
    ]:
      sweep_list.append(hyper.sweep(hparam, hparam_sweep[hparam][dataset]))
    dataset_sweep = hyper.product(sweep_list)
    all_sweeps.append(dataset_sweep)

  return hyper.product([
      hyper.sweep('config.acquisition_method', acquisition_methods),
      hyper.chainit(all_sweeps),
      # use mean of performance for choosing hyperparameters.
      hyper.product([
          model_specs,
          hyper.sweep('config.seed', [189, 827, 905, 964, 952]),
      ])
  ])
