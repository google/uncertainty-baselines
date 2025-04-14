# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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
r"""ViT-L/32 + BatchEnsemble finetuning on RETINA.

This config is used for models pretrained on either JFT-300M or ImageNet-21K.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Config."""
  config = ml_collections.ConfigDict()

  # Directories
  # model_init should be modified per experiment
  config.model_init = ('gs://ub-checkpoints/ImageNet21k_BE-L32/'
      'baselines-jft-0209_205214/1/checkpoint.npz')
  config.data_dir = 'gs://ub-data/retinopathy'
  # The directory where the model weights and training/evaluation summaries
  #   are stored.
  config.output_dir = (
      '/tmp/diabetic_retinopathy_detection/vit-32-i21k/batchensemble')

  # REQUIRED: distribution shift.
  # 'aptos': loads APTOS (India) OOD validation and test datasets.
  #   Kaggle/EyePACS in-domain datasets are unchanged.
  # 'severity': uses DiabeticRetinopathySeverityShift dataset, a subdivision
  #   of the Kaggle/EyePACS dataset to hold out clinical severity labels as OOD.
  config.distribution_shift = 'aptos'

  # If checkpoint path is provided, resume training and/or conduct evaluation
  #   with this checkpoint. See `checkpoint_utils.py`.
  config.resume = None

  config.prefetch_to_device = 2
  config.trial = 0

  # Logging and hyperparameter tuning

  config.use_wandb = False  # Use wandb for logging.
  config.wandb_dir = 'wandb'  # Directory where wandb logs go.
  config.project = 'ub-debug'  # Wandb project name.
  config.exp_name = None  # Give experiment a name.
  config.exp_group = None  # Give experiment a group name.

  # Model Flags

  # TODO(nband): fix issue with sigmoid loss.
  config.num_classes = 2

  # Model section
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

  # This is "no head" fine-tuning, which we use by default
  config.model.representation_size = None

  # BatchEnsemble parameters.

  use_jft = False

  if use_jft:
    # JFT-pretrained settings.
    # TODO(trandustin): Remove `ensemble_attention` hparam once we no longer
    # need checkpoints that only apply BE on the FF block.
    config.model.transformer.ensemble_attention = True
    config.model.transformer.be_layers = (22, 23)
  else:
    # I21K-pretrained settings.
    config.model.transformer.ensemble_attention = False
    config.model.transformer.be_layers = (21, 22, 23)

  config.model.transformer.ens_size = 3
  config.model.transformer.random_sign_init = -0.5  # set in sweep
  config.fast_weight_lr_multiplier = 1.0  # set in sweep

  # Preprocessing
  # Input resolution of each retina image. (Default: 512)
  config.pp_input_res = 512  # pylint: disable=invalid-name
  pp_common = f'|onehot({config.num_classes})'
  config.pp_train = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)
  config.pp_eval = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)

  # Training Misc
  config.batch_size = 128  # set in sweep
  config.seed = 0  # Random seed.
  config.shuffle_buffer_size = 15_000  # Per host, so small-ish is ok.

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'
  config.grad_clip_norm = 1.0
  config.weight_decay = 0.  # No explicit weight decay

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 1e-3  # Set in sweep.
  config.lr.decay_type = 'linear'

  # The dataset is imbalanced (e.g., in Country Shift, we have 19.6%, 18.8%,
  # 19.2% positive examples in train, val, test respectively).
  # None (default) will not perform any loss reweighting.
  # 'constant' will use the train proportions to reweight the binary cross
  #   entropy loss.
  # 'minibatch' will use the proportions of each minibatch to reweight the loss.
  config.class_reweight_mode = None

  # Evaluation Misc
  config.only_eval = False  # Disables training, only evaluates the model
  config.use_validation = True  # Whether to use a validation split
  config.use_test = True  # Whether to use a test split

  # Step Counts
  config.total_and_warmup_steps = (10_000, 500)
  config.log_training_steps = 100
  config.log_eval_steps = 1000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.args = {}
  return config


def get_sweep(hyper):
  """Sweeps over hyperparameters."""
  checkpoints = ['/path/to/pretrained_model_ckpt.npz']
  use_jft = True  # whether to use JFT or I21K
  if use_jft:
    ensemble_attention = True
    be_layers = (22, 23)
    ens_size = 3
  else:
    ensemble_attention = False
    be_layers = (21, 22, 23)
    ens_size = 3
  return hyper.product([
      hyper.sweep('config.distribution_shift', ['aptos', 'severity']),
      hyper.sweep('config.batch_size', [64, 128]),
      hyper.sweep('config.total_and_warmup_steps',
                  [(12_500, 5000), (20_000, 7500), (50_000, 10000)]),
      hyper.sweep('config.lr.decay_type', ['cosine', 'linear']),
      hyper.zipit([
          hyper.loguniform('config.lr.base',
                           hyper.interval(0.0001, 0.02)),
          hyper.loguniform('config.weight_decay',
                           hyper.interval(1e-6, 2e-4)),
      ], length=1),
      hyper.sweep('config.grad_clip_norm', [2.5]),
      hyper.sweep('config.fast_weight_lr_multiplier', [0.5, 1.0, 2.0]),
      hyper.sweep('config.model.transformer.random_sign_init', [-0.5, 0.5]),
      hyper.sweep('config.model_init', checkpoints),
      hyper.fixed(
          'config.model.transformer.be_layers', be_layers, length=1),
      hyper.fixed(
          'config.model.transformer.ensemble_attention',
          ensemble_attention,
          length=1),
      hyper.fixed('config.model.transformer.ens_size', ens_size, length=1),
  ])
