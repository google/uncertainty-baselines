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
r"""CLIP ImageNet zero-shot evaluation.

"""
# pylint: enable=line-too-long

import ml_collections

from configs import clip_common  # local file import from experimental.multimodal


def get_config():
  """Config for zero-shot evaluation of CLIP on ImageNet."""
  config = ml_collections.ConfigDict()

  config.model_name = 'vit_b16'
  config.only_eval = True

  # Fine-tuning dataset
  config.dataset = 'coco_captions'
  config.train_split = 'train'
  config.val_split = 'val'

  BATCH_SIZE = 512  # pylint: disable=invalid-name
  config.batch_size = BATCH_SIZE
  config.batch_size_eval = BATCH_SIZE
  config.val_cache = False

  config.total_steps = 20_000

  config.tokenizer_max_len = 77

  # PP modified from
  # third_party/py/big_vision/configs/proj/image_text/lit_coco.py
  INPUT_RES = clip_common.IMAGE_RESOLUTION[config.model_name]  # pylint: disable=invalid-name
  coco_pp = 'get_coco_captions("text")'
  image_pp = f'|decode|resize({INPUT_RES})|flip_lr|randaug(2,10)|value_range(-1,1)'
  text_pp = f'|clip_tokenize({config.tokenizer_max_len}, key="text", key_result="text")'
  final_pp = '|keep(["image", "text"])'
  config.pp_train = config.pp_eval = coco_pp + image_pp + text_pp + final_pp

  config.shuffle_buffer_size = 50_000  # Per host, so small-ish is ok.

  config.log_training_steps = 100
  config.log_eval_steps = 1000
  config.checkpoint_steps = 4000
  config.checkpoint_timeout = 1

  config.prefetch_to_device = 2
  config.trial = 0

  # Model section
  config.model_init = clip_common.CHECKPOINTS[config.model_name]
  config.convert_pytorch = True
  config.model = ml_collections.config_dict.create(
      **clip_common.CONFIGS[config.model_name])

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.06
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'

  # zeroshot section
  def zeroshot_pp(num_classes, resize_method='area'):
    zeroshot_pp = f'decode|resize_small({INPUT_RES}, method="{resize_method}")|central_crop({INPUT_RES})'
    # zeroshot_pp += '|value_range(-1, 1)'
    zeroshot_pp += f'|value_range(0, 1)|normalize({clip_common.CLIP_IMAGE_MEAN}, {clip_common.CLIP_IMAGE_STD})'
    zeroshot_pp += f'|onehot({num_classes}, key="label", key_result="text")'
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
      'cars196': {
          'dataset': 'cars196',
          'split': 'train',
          'classnames_key': 'cars196',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(196)
      },
      'caltech101': {
          'dataset': 'caltech101',
          'split': 'train',
          'classnames_key': 'caltech101',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(102)
      },
      'cifar10': {
          'dataset': 'cifar10',
          'split': 'train',
          'classnames_key': 'cifar10',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(10, 'bicubic')
      },
      'cifar100': {
          'dataset': 'cifar100',
          'split': 'train',
          'classnames_key': 'cifar100',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(100, 'bicubic')
      },
      'dtd': {
          'dataset': 'dtd',
          'split': 'test',
          'classnames_key': 'dtd',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(47)
      },
      'eurosat': {
          'dataset': 'eurosat',
          'split': 'train',
          'classnames_key': 'eurosat',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(10, 'bicubic')
      },
      'resisc45': {
          'dataset': 'resisc45',
          'split': 'train',
          'classnames_key': 'resisc45',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(45, 'bicubic')
      },
      'imagenet_prompt': {
          'dataset': 'imagenet2012',
          'split': 'validation',
          'classnames_key': 'imagenet',
          'prompts_key': 'imagenet',
          'pp_spec': zeroshot_pp(1000)
      },
      'cars196_prompt': {
          'dataset': 'cars196',
          'split': 'train',
          'classnames_key': 'cars196',
          'prompts_key': 'cars196',
          'pp_spec': zeroshot_pp(196)
      },
      'caltech101_prompt': {
          'dataset': 'caltech101',
          'split': 'train',
          'classnames_key': 'caltech101',
          'prompts_key': 'caltech101',
          'pp_spec': zeroshot_pp(102)
      },
      'cifar10_prompt': {
          'dataset': 'cifar10',
          'split': 'train',
          'classnames_key': 'cifar10',
          'prompts_key': 'cifar10',
          'pp_spec': zeroshot_pp(10, 'bicubic')
      },
      'cifar100_prompt': {
          'dataset': 'cifar100',
          'split': 'train',
          'classnames_key': 'cifar100',
          'prompts_key': 'cifar100',
          'pp_spec': zeroshot_pp(100, 'bicubic')
      },
      'dtd_prompt': {
          'dataset': 'dtd',
          'split': 'test',
          'classnames_key': 'dtd',
          'prompts_key': 'dtd',
          'pp_spec': zeroshot_pp(47)
      },
      'eurosat_prompt': {
          'dataset': 'eurosat',
          'split': 'train',
          'classnames_key': 'eurosat',
          'prompts_key': 'eurosat',
          'pp_spec': zeroshot_pp(10, 'bicubic')
      },
      'resisc45_prompt': {
          'dataset': 'resisc45',
          'split': 'train',
          'classnames_key': 'resisc45',
          'prompts_key': 'resisc45',
          'pp_spec': zeroshot_pp(45, 'bicubic')
      },
      'imagenet_best': {
          'dataset': 'imagenet2012',
          'split': 'validation',
          'classnames_key': 'imagenet',
          'prompts_key': 'imagenet_best',
          'pp_spec': zeroshot_pp(1000)
      },
  }

  return config


def get_sweep(hyper):
  return hyper.product([
  ])
