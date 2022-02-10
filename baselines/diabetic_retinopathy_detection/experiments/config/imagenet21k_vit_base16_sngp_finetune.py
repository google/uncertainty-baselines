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
r"""ViT-SNGP-B/16 finetuning on a Diabetic Retinopathy Detection dataset.

"""
# pylint: enable=line-too-long

import ml_collections


def get_config():
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()

  # Fine-tuning dataset
  config.data_dir = 'gs://ub-data/retinopathy'
  # config.in_domain_dataset = 'ub_diabetic_retinopathy_detection'
  # config.ood_dataset = 'aptos'
  # config.train_split = 'train'
  # config.val_split = 'validation'

  # TODO(nband): fix issue with sigmoid loss.
  config.num_classes = 2

  # BATCH_SIZE = 64  # pylint: disable=invalid-name
  # config.batch_size = BATCH_SIZE

  # config.total_steps = 10_000

  # Input resolution of each retina image. (Default: 512)
  config.pp_input_res = 512  # pylint: disable=invalid-name
  pp_common = f'|onehot({config.num_classes})'
  config.pp_train = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)
  config.pp_eval = (
      f'diabetic_retinopathy_preprocess({config.pp_input_res})' + pp_common)

  config.shuffle_buffer_size = 15_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  # NOTE: eval is very fast O(seconds) so it's fine to run it often.
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  # pre-trained model ckpt file
  # !!!  The below section should be modified per experiment
  config.model_init = 'gs://ub-checkpoints/ImageNet21k_ViT-B16-GP/ImageNet21k_ViT-B:16-GP_29240948.npz'

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

  # Re-initialize the trainable parameters in GP output layer (Also those in the
  # dense output layer if loading from deterministic checkpoint).
  config.model_reinit_params = ('head/output_layer/kernel',
                                'head/output_layer/bias', 'head/kernel',
                                'head/bias')

  # Gaussian process layer section
  # config.gp_layer = ml_collections.ConfigDict()

  # config.gp_layer.ridge_penalty = 1.
  # Disable momentum in order to use exact covariance update for finetuning.
  # config.gp_layer.covmat_momentum = -1.
  # config.gp_layer.mean_field_factor = 20.

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.loss = 'softmax_xent'  # or 'sigmoid_xent'

  # config.lr = ml_collections.ConfigDict()

  # Set as command line arguments for wandb compatibility.
  # config.grad_clip_norm = 1.
  # config.weight_decay = None  # No explicit weight decay
  # config.lr.base = 0.0005
  # config.lr.warmup_steps = 500
  # config.lr.decay_type = 'cosine'

  config.args = {}
  return config
