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
r"""Train segmenter model on street_hazards.

Compare performance from deterministic upstream checkpoints.

"""
# pylint: enable=line-too-long

import ml_collections
import os
import datetime

_CITYSCAPES_FINE_TRAIN_SIZE = 2975
_CITYSCAPES_COARSE_TRAIN_SIZE = 19998

_ADE20K_TRAIN_SIZE = 20210
_PASCAL_VOC_TRAIN_SIZE = 10582
_PASCAL_CONTEXT_TRAIN_SIZE = 4998
_STREET_HAZARDS_TRAIN_SIZE = 5125

TRAIN_SIZES = {
    'cityscapes': _CITYSCAPES_FINE_TRAIN_SIZE,
    'ade20k': _ADE20K_TRAIN_SIZE,
    'ade20k_ind': _ADE20K_TRAIN_SIZE,
    'pascal_voc': _PASCAL_VOC_TRAIN_SIZE,
    'pascal_context': _PASCAL_CONTEXT_TRAIN_SIZE,
    'street_hazards': _STREET_HAZARDS_TRAIN_SIZE

}

# Model specs.
LOAD_PRETRAINED_BACKBONE = True
BACKBONE_ORIGIN = 'vision_transformer'
VIT_SIZE = 'L'
STRIDE = 16
RESNET_SIZE = None
CLASSIFIER = 'token'
target_size = (720, 720)
UPSTREAM_TASK = 'augreg+i21k+imagenet2012'


# Upstream
MODEL_PATHS = {
    # Imagenet 21k + finetune in imagenet2012 with perf 0.85 adap_res 384 with augreg
    ('vision_transformer', 'L', 16, None, 'token', 'i21k+imagenet2012'):
        'gs://vit_models/imagenet21k+imagenet2012/ViT-L_16.npz',
    ('vision_transformer', 'L', 16, None, 'token', 'augreg+i21k+imagenet2012'):
        'gs://vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
}


MODEL_PATH = MODEL_PATHS[(BACKBONE_ORIGIN, VIT_SIZE, STRIDE, RESNET_SIZE,
                          CLASSIFIER, UPSTREAM_TASK)]

if VIT_SIZE == 'B':
  mlp_dim = 3072
  num_heads = 12
  num_layers = 12
  hidden_size = 768
elif VIT_SIZE == 'L':
  mlp_dim = 4096
  num_heads = 16
  num_layers = 24
  hidden_size = 1024

TRAIN_SAMPLES = 32


def get_config(runlocal=''):
  """Returns the configuration for ADE20k_ind segmentation."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'street_hazards_deterministic'

  # Dataset.
  config.dataset_name = 'robust_segvit_segmentation'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = target_size
  config.dataset_configs.train_target_size = config.dataset_configs.get_ref(
      'target_size')
  config.dataset_configs.denoise = None
  config.dataset_configs.use_timestep = 0

  config.dataset_configs.train_split = 'train'
  config.dataset_configs.name = 'street_hazards'
  config.dataset_configs.dataset_name = ''  # ood name flag to write in eval.

  # Model.
  config.model_name = 'segvit'
  config.model = ml_collections.ConfigDict()

  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (STRIDE, STRIDE)

  config.model.backbone = ml_collections.ConfigDict()
  config.model.backbone.type = 'vit'
  config.model.backbone.mlp_dim = mlp_dim
  config.model.backbone.num_heads = num_heads
  config.model.backbone.num_layers = num_layers
  config.model.backbone.hidden_size = hidden_size
  config.model.backbone.dropout_rate = 0.1
  config.model.backbone.attention_dropout_rate = 0.0
  config.model.backbone.classifier = CLASSIFIER

  # Decoder
  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.type = 'linear'

  # Training.
  config.trainer_name = 'segvit_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.0
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  config.num_training_epochs = ml_collections.FieldReference(100)
  config.batch_size = 32
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # Learning rate.
  config.num_train_examples = TRAIN_SIZES.get(config.dataset_configs.name)
  config.steps_per_epoch = config.get_ref(
      'num_train_examples') // config.get_ref('batch_size')
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 1 * config.get_ref('steps_per_epoch')
  config.lr_configs.steps_per_cycle = config.get_ref(
      'num_training_epochs') * config.get_ref('steps_per_epoch')
  config.lr_configs.base_learning_rate = 1e-4

  # model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  # load pretrained backbone
  config.load_pretrained_backbone = LOAD_PRETRAINED_BACKBONE
  config.pretrained_backbone_configs = ml_collections.ConfigDict()
  config.pretrained_backbone_configs.checkpoint_format = BACKBONE_ORIGIN
  config.pretrained_backbone_configs.checkpoint_path = MODEL_PATH
  config.pretrained_backbone_configs.token_init = True
  config.pretrained_backbone_configs.classifier = 'token'
  config.pretrained_backbone_configs.backbone_type = 'vit'

  # Logging.
  config.write_summary = True
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = False  # Profile using xprof.
  config.checkpoint = True  # Do checkpointing.
  config.checkpoint_steps = 5 * config.get_ref('steps_per_epoch')

  config.debug_train = False  # Debug mode during training.
  config.debug_eval = False  # Debug mode during eval.
  config.log_eval_steps = 1 * config.get_ref('steps_per_epoch')

  # Evaluation.
  config.eval_configs = ml_collections.ConfigDict()
  config.eval_configs.mode = 'standard'
  config.eval_mode = False
  config.eval_covariate_shift = True
  config.eval_label_shift = True
  config.model.input_shape = target_size

  config.eval_robustness_configs = ml_collections.ConfigDict()
  config.eval_robustness_configs.auc_online = True
  config.eval_robustness_configs.method_name = 'msp'

  # wandb.ai configurations.
  config.use_wandb = False
  config.wandb_dir = 'wandb'
  config.wandb_project = 'rdl-debug'
  config.wandb_entity = 'ekellbuch'
  config.wandb_exp_name = None  # Give experiment a name.
  config.wandb_exp_name = (
          os.path.splitext(os.path.basename(__file__))[0] + '_' +
          datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S'))
  config.wandb_exp_group = None  # Give experiment a group name.

  if runlocal:
    config.count_flops = False
    config.dataset_configs.train_target_size = (128, 128)
    config.model.input_shape = config.dataset_configs.train_target_size
    config.batch_size = 8
    config.num_training_epochs = 5
    config.warmup_steps = 0
    config.dataset_configs.train_split = f'train[:{TRAIN_SAMPLES}]'
    config.dataset_configs.validation_split = f'validation[:{TRAIN_SAMPLES}]'
    config.num_train_examples = TRAIN_SAMPLES

  return config


def checkpoint(hyper, backbone_origin, vit_size, stride, resnet_size,
               classifier, upstream_task):
  """Defines checkpoints for sweep."""
  overwrites = []
  if resnet_size is not None:
    raise NotImplementedError('')
  else:
    overwrites.append(
        hyper.sweep('config.model.patches', [{
            'size': (stride, stride)
        }]))

  if vit_size == 'B':
    overwrites.append(hyper.sweep('config.model.backbone.mlp_dim', [3072]))
    overwrites.append(hyper.sweep('config.model.backbone.num_heads', [12]))
    overwrites.append(hyper.sweep('config.model.backbone.num_layers', [12]))
    overwrites.append(hyper.sweep('config.model.backbone.hidden_size', [768]))
  elif vit_size == 'L':
    overwrites.append(hyper.sweep('config.model.backbone.mlp_dim', [4096]))
    overwrites.append(hyper.sweep('config.model.backbone.num_heads', [16]))
    overwrites.append(hyper.sweep('config.model.backbone.num_layers', [24]))
    overwrites.append(hyper.sweep('config.model.backbone.hidden_size', [1024]))
  else:
    raise NotImplementedError('')

  overwrites.append(
      hyper.sweep('config.pretrained_backbone_configs.checkpoint_format',
                  [backbone_origin]))
  overwrites.append(
      hyper.sweep('config.pretrained_backbone_configs.checkpoint_path', [
          MODEL_PATHS[(backbone_origin, vit_size, stride, resnet_size,
                       classifier, upstream_task)]
      ]))

  return hyper.product(overwrites)


def get_sweep(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""

  learning_rate = hyper.sweep('config.lr_configs.base_learning_rate',
                              [1e-4, 3e-4, 3e-5, 1e-5])

  epochs = hyper.sweep('config.num_training_epochs', [100, 50, 200, 250])

  return hyper.product([learning_rate, epochs])
