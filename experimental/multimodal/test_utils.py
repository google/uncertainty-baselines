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

"""Testing-related utility functions."""

import ml_collections


def get_config(dataset_name, model_type='vit', batch_size=3, total_steps=2):
  """Config for training CLIP."""
  config = ml_collections.ConfigDict()
  config.seed = 0

  config.batch_size = batch_size
  config.total_steps = total_steps

  num_examples = config.batch_size * config.total_steps

  if dataset_name == 'imagenet2012':
    config.dataset = 'imagenet2012'
    config.val_split = f'train[:{num_examples}]'
    config.train_split = f'train[{num_examples}:{num_examples*2}]'
  else:
    msg = f'Invalid dataset: `{dataset_name}`. Only `imagenet2012` is supported for now.'
    raise ValueError(msg)

  config.prefetch_to_device = 1
  config.shuffle_buffer_size = 20
  config.val_cache = False

  config.log_training_steps = config.total_steps
  config.log_eval_steps = config.total_steps
  config.checkpoint_steps = config.total_steps
  config.keep_checkpoint_steps = config.total_steps

  config.tokenizer_max_len = 77

  INPUT_RES = 32  # pylint: disable=invalid-name
  img_pp = f'decode|resize({INPUT_RES})|central_crop({INPUT_RES})|value_range(-1,1)'
  txt_pp = f'|clip_i1k_label_names|clip_tokenize({config.tokenizer_max_len}, key="label", key_result="text")'
  final_pp = '|keep(["image", "text"])'
  config.pp_train = config.pp_eval = img_pp + txt_pp + final_pp

  # Model section
  # TODO(jallingham): Rework API for clip_tokenizer and clip model so that the
  # vocab_size, vision_features, and text_features can be made small for tests.
  config.model = ml_collections.ConfigDict()
  config.model.embed_dim = 1
  config.model.vocab_size = 49408
  if model_type == 'vit':
    config.model.vision_num_layers = 1
    config.model.vision_patch_size = 32
  elif model_type == 'resnet':
    config.model.vision_num_layers = (1, 1, 1, 1)
  else:
    raise NotImplementedError('model_type has to be either vit or resnet.')
  config.model.vision_features = 64
  config.model.text_features = 64
  config.model.text_num_heads = 2
  config.model.text_num_layers = 1

  # Optimizer section
  config.optim_name = 'Momentum'
  config.optim = ml_collections.ConfigDict()
  config.grad_clip_norm = 1.0
  config.weight_decay = None  # No explicit weight decay

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.06
  config.lr.warmup_steps = 500
  config.lr.decay_type = 'cosine'

  # Few-shot eval section
  if model_type == 'vit':
    config.fewshot = ml_collections.ConfigDict()
    config.fewshot.log_steps = config.total_steps
    config.fewshot.datasets = {
        'pets': ('oxford_iiit_pet', f'train[:{num_examples}]',
                 f'test[:{num_examples}]'),
    }
    config.fewshot.pp_train = f'decode|resize({INPUT_RES})|central_crop({INPUT_RES})|value_range(-1,1)|drop("segmentation_mask")'
    config.fewshot.pp_eval = f'decode|resize({INPUT_RES})|central_crop({INPUT_RES})|value_range(-1,1)|drop("segmentation_mask")'
    config.fewshot.shots = [10]
    config.fewshot.l2_regs = [2.0**-6]
    config.fewshot.walk_first = ('pets', config.fewshot.shots[0])
  # TODO(jjren) make fewshotters compatible for resnet (take batch_states)

  # Zero-shot eval section
  def zeroshot_pp(num_classes):
    zeroshot_pp = f'decode|resize({INPUT_RES})|central_crop({INPUT_RES})|value_range(-1, 1)'
    zeroshot_pp += f'|onehot({num_classes}, key="label", key_result="text")'
    zeroshot_pp += '|keep(["image", "text"])'
    return zeroshot_pp

  config.zeroshot_eval_datasets = {
      'imagenet': {
          'dataset': 'imagenet2012',
          'split': f'train[:{num_examples}]',
          'classnames_key': 'imagenet',
          'prompts_key': 'none',
          'pp_spec': zeroshot_pp(1000)
      }
  }

  return config
