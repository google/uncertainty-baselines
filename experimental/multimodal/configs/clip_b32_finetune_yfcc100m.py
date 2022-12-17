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
r"""CLIP finetuning on YFCC100M.

"""
# pylint: enable=line-too-long

import ml_collections
from configs import clip_common  # local file import from experimental.multimodal


def get_config():
  """Config for finetuning CLIP on YFCC100M."""
  config = ml_collections.ConfigDict()

  TEXT_FEATURE = 'title'  # pylint: disable=invalid-name
  # SPLIT = f'with_{TEXT_FEATURE}'  # pylint: disable=invalid-name
  SPLIT = 'clip'  # pylint: disable=invalid-name
  # SPLIT = 'full'  # pylint: disable=invalid-name

  config.model_name = 'vit_b32'
  config.only_eval = False
  # Fine-tuning dataset
  config.dataset = 'yfcc100m'
  config.train_split = f'{SPLIT}[10000:]'
  config.val_split = f'{SPLIT}[:10000]'

  config.batch_size = 16384
  config.batch_size_eval = 16384
  config.val_cache = False

  config.total_steps = 15_000

  config.tokenizer_max_len = 77

  INPUT_RES = clip_common.IMAGE_RESOLUTION[config.model_name]  # pylint: disable=invalid-name
  train_image_pp = f'decode|inception_crop({INPUT_RES})|value_range(-1,1)'
  text_pp = '|shuffle_join(key="tags")'
  text_pp += f'|clip_tokenize({config.tokenizer_max_len}, key="{TEXT_FEATURE}", key_result="text")'
  final_pp = '|keep(["image", "text"])'
  config.pp_train = train_image_pp + text_pp + final_pp
  eval_image_pp = f'decode|resize({INPUT_RES})|value_range(-1,1)'
  config.pp_eval = eval_image_pp + text_pp + final_pp

  config.shuffle_buffer_size = 250_000  # Per host, so small-ish is ok.

  config.log_training_steps = 50
  config.log_eval_steps = 1000
  config.checkpoint_steps = 1000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  config.model_init = clip_common.CHECKPOINTS[config.model_name]
  config.convert_pytorch = True
  config.model = ml_collections.config_dict.create(
      **clip_common.CONFIGS[config.model_name])

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = 1e-5

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 3e-4
  config.lr.warmup_steps = max(int(0.03 * config.total_steps), 500)
  config.lr.decay_type = 'cosine'

  # zeroshot section
  def zeroshot_pp(nclasses):
    zeroshot_pp = f'decode|resize({INPUT_RES})|central_crop({INPUT_RES})|value_range(-1, 1)'
    zeroshot_pp += f'|onehot({nclasses}, key="label", key_result="text")'
    zeroshot_pp += '|keep(["image", "text"])'
    return zeroshot_pp

  config.zeroshot_eval_datasets = {
      'imagenet': {
          'dataset': 'imagenet2012',
          'split': 'validation',
          'classnames_key': 'imagenet',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(1000)
      },
  }

  return config


def get_sweep(hyper):
  return hyper.product([
      hyper.sweep('config.lr.base', [3e-4, 1e-4, 3e-5, 1e-5]),
      hyper.sweep('config.weight_decay', [1e-5, 3e-6]),
      hyper.sweep('config.total_steps', [15_000]),
  ])
