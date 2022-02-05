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
r"""ViT-B/16 finetuning on a Diabetic Retinopathy Detection dataset.

"""
# pylint: enable=line-too-long

import ml_collections
# TODO(dusenberrymw): Open-source remaining imports.


def get_sweep(hyper):
  return hyper.product([])


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # * Data load / output flags. *

  # The directory where the model weights and training/evaluation summaries
  #   are stored.
  config.output_dir = (
    '/tmp/diabetic_retinopathy_detection/vit-16-i21k/deterministic')

  # Dataset for model pretraining. Specifies the config to use.
  config.pretrain_dataset = 'imagenet21k'

  # Fine-tuning dataset.
  config.data_dir = 'gs://ub-data/retinopathy'

  # REQUIRED: distribution shift.
  # 'aptos': loads APTOS (India) OOD validation and test datasets.
  #   Kaggle/EyePACS in-domain datasets are unchanged.
  # 'severity': uses DiabeticRetinopathySeverityShift dataset, a subdivision
  #   of the Kaggle/EyePACS dataset to hold out clinical severity labels as OOD.
  config.distribution_shift = 'aptos'

  # If provided, resume training and/or conduct evaluation using this
  #   checkpoint. Will only be used if the output_dir does not already
  #   contain a checkpointed model. See `checkpoint_utils.py`.
  config.resume_checkpoint_path = None

  config.prefetch_to_device = 2
  config.trial = 0

  # * Logging and hyperparameter tuning. *

  config.use_wandb = False  # Use wandb for logging.
  config.wandb_dir = 'wandb'  # Directory where wandb logs go.
  config.project = 'ub-debug'  # Wandb project name.
  config.exp_name = None  # Give experiment a name.
  config.exp_group = None  # Give experiment a group name.

  # * Model Flags. *

  # TODO(nband): fix issue with sigmoid loss.
  config.num_classes = 2

  # REQUIRED: Specifies size of ViT backbone model. Options: {"B/16", "L/32"}.
  config.vit_model_size = 'B/16'

  # pre-trained model ckpt file
  # !!!  The below section should be modified per experiment
  config.model_init = (
      'gs://ub-data/ImageNet21k_ViT-B16_ImagetNet21k_ViT-B_16_28592399.npz')

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

  # * Preprocessing. *

  # Input resolution of each retina image. (Default: 512)
  config.pp_input_res = 512  # pylint: disable=invalid-name
  pp_common = f'|onehot({config.num_classes})'
  config.pp_train = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)
  config.pp_eval = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)

  # * Training Misc. *
  config.batch_size = 512  # using TPUv3-64
  config.seed = 0  # Random seed.
  config.shuffle_buffer_size = 15_000  # Per host, so small-ish is ok.

  # * Optimization. *
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'
  config.lr = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0  # Gradient clipping threshold.
  config.weight_decay = None  # No explicit weight decay.
  config.lr.base = 0.003
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'linear'

  # The dataset is imbalanced (e.g., in Country Shift, we have 19.6%, 18.8%,
  # 19.2% positive examples in train, val, test respectively).
  # None (default) will not perform any loss reweighting.
  # 'constant' will use the train proportions to reweight the binary cross
  #   entropy loss.
  # 'minibatch' will use the proportions of each minibatch to reweight the loss.
  config.class_reweight_mode = None

  # * Evaluation Misc. *
  config.only_eval = False  # Disables training, only evaluates the model.
  config.use_validation = True  # Whether to use a validation split.
  config.use_test = True  # Whether to use a test split.

  # * Step Counts. *
  config.total_steps = 10_000
  config.log_training_steps = 100
  config.log_eval_steps = 1000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.args = {}
  return config

def get_sweep(hyper):
  """Sweeps over datasets."""
  # Adapted the sweep over checkpoints from vit_l32_finetune.py.
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  use_jft = True  # whether to use JFT-300M or ImageNet-21K settings
  sweep_lr = True  # whether to sweep over learning rates
  acquisition_methods = ['uniform', 'entropy', 'margin', 'density']
  if use_jft:
    cifar10_sweep = sweep_utils.cifar10(hyper, val_split='test')
    cifar10_sweep.append(hyper.fixed('config.lr.base', 0.01, length=1))
    cifar10_sweep = hyper.product(cifar10_sweep)
  else:
    cifar10_sweep = sweep_utils.cifar10(hyper, val_split='test')
    cifar10_sweep.append(hyper.fixed('config.lr.base', 0.003, length=1))
    cifar10_sweep = hyper.product(cifar10_sweep)
  if sweep_lr:
    # Apply a learning rate sweep following Table 4 of Vision Transformer paper.
    checkpoints = [checkpoints[0]]
    cifar10_sweep = sweep_utils.cifar10(hyper, val_split='train[98%:]')
    cifar10_sweep.append(
        hyper.sweep('config.lr.base', [0.03, 0.01, 0.003, 0.001]))
    cifar10_sweep = hyper.product(cifar10_sweep)

  return hyper.product([
      cifar10_sweep,
      hyper.sweep('config.model_init', checkpoints),
      hyper.sweep('config.acquisition_method', acquisition_methods),
  ])
