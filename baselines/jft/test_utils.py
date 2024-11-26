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

"""Testing-related utility functions."""

import ml_collections  # pylint: disable=g-bad-import-order


def get_config(
    dataset_name,
    classifier,
    representation_size,
    batch_size=3,
    total_steps=3,
    use_sngp=False,
    use_gp_layer=False):
  """Config for training a patch-transformer on JFT."""
  config = ml_collections.ConfigDict()
  config.seed = 0

  config.batch_size = batch_size
  config.total_steps = total_steps

  num_examples = config.batch_size * config.total_steps

  # TODO(dusenberrymw): JFT + mocking is broken.
  # config.dataset = 'jft/entity:1.0.0'
  # config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  # config.train_split = 'train'  # task_adapt used train+validation so +64167
  # config.num_classes = 18291
  # NOTE: TFDS mocking currently ignores split slices.
  if dataset_name == 'cifar10':
    config.dataset = 'cifar10'
    config.val_split = f'train[:{num_examples}]'
    config.train_split = f'train[{num_examples}:{num_examples*2}]'
    config.num_classes = 10
  elif dataset_name == 'imagenet2012':
    config.dataset = 'imagenet2012'
    config.val_split = f'train[:{num_examples}]'
    config.train_split = f'train[{num_examples}:{num_examples*2}]'
    config.num_classes = 1000
  elif dataset_name == 'imagenet21k':
    config.dataset = 'imagenet21k'
    config.val_split = f'full[:{num_examples}]'
    config.train_split = f'full[{num_examples}:{num_examples*2}]'
    config.num_classes = 21843

  config.prefetch_to_device = 1
  config.shuffle_buffer_size = 20
  config.val_cache = False

  config.log_training_steps = config.total_steps
  config.log_eval_steps = config.total_steps
  config.checkpoint_steps = config.total_steps
  config.keep_checkpoint_steps = config.total_steps

  pp_common = '|value_range(-1, 1)'
  if dataset_name in ['cifar10', 'imagenet2012']:
    label_key = '"label"'
  else:
    label_key = '"labels"'
  pp_common += (
      f'|onehot({config.num_classes}, key={label_key}, key_result="labels")')
  pp_common += '|keep(["image", "labels", "id"])'
  # id is needed for active learning
  # TODO(dusenberrymw): Mocking doesn't seem to encode into jpeg format.
  # config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_train = 'decode|inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.init_head_bias = 1e-3

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 4
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 3
  config.model.transformer.num_heads = 2
  config.model.transformer.num_layers = 1
  config.model.classifier = classifier
  config.model.representation_size = representation_size

  if use_sngp:
    # Reinitialize GP output layer.
    config.model_reinit_params = [
        'head/output_layer/kernel', 'head/output_layer/bias', 'head/kernel',
        'head/bias'
    ]
    config.use_gp_layer = use_gp_layer

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = None  # No explicit weight decay

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.1
  config.lr.warmup_steps = 1 if total_steps > 1 else 0
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-3

  # Few-shot eval section
  config.fewshot = ml_collections.ConfigDict()
  config.fewshot.representation_layer = 'pre_logits'
  config.fewshot.log_steps = config.total_steps
  config.fewshot.datasets = {
      'pets': ('oxford_iiit_pet', f'train[:{num_examples}]',
               f'test[:{num_examples}]'),
      'imagenet': ('imagenet2012_subset/1pct', f'train[:{num_examples}]',
                   f'validation[:{num_examples}]'),
  }
  config.fewshot.pp_train = 'decode|resize(256)|central_crop(224)|value_range(-1,1)|drop("segmentation_mask")'
  config.fewshot.pp_eval = 'decode|resize(256)|central_crop(224)|value_range(-1,1)|drop("segmentation_mask")'
  config.fewshot.shots = [10]
  config.fewshot.l2_regs = [2.0**-6]
  config.fewshot.walk_first = ('imagenet', config.fewshot.shots[0])

  return config
