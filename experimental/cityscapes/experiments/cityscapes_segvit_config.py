# pylint: disable=line-too-long
r"""Default configs for Cityscapes segmentation.

Launch a TPU job:
gxm third_party/py/scenic/google/xm/launch_xm.py -- \
  --binary //third_party/py/scenic/projects/segvit:main \
  --config='third_party/py/scenic/projects/segvit/configs/cityscapes_segvit_config.py' \
  --platform=df_8x8 --xm_resource_alloc=group:brain/grand-vision-xm-df \
  --priority=115 \
  --exp_name=cityscapes_segvit \
  --notes "R50-ViT-B/16 1024x2048 sweep"

Test run: xid/27318283
Performance: ~78% mIoU (WID 9 https://flatboard.corp.google.com/plot/hu4ooWrx4t0)

"""
# pylint: enable=line-too-long

import ml_collections

_CITYSCAPES_TRAIN_SIZE = 2975

# Model specs.
VIT_SIZE = 'B'
STRIDE = 16
RESNET_SIZE = 50
CLASSIFIER = 'token'

# JFT pretrained models derived from:
# https://colab.corp.google.com/drive/1GNO2D-BhZGX8UARyZCQ8xfhlCea42yx9#scrollTo=UXdJdTS6rfsx
MODEL_PATHS = {
    ('B', 32, 50, 'token'):
        '/cns/tp-d/home/dune/task_adapt/xzhai/tmp/hybrid/17221856/5/checkpoint.npz',
    ('B', 16, 50, 'token'):
        '/cns/tp-d/home/dune/task_adapt/xzhai/tmp/hybrid/17221856/6/checkpoint.npz',
    ('B', 32, None, 'token'):
        '/cns/tp-d/home/brain-ber/adosovitskiy/17084881/1/checkpoint.npz',
    ('B', 16, None, 'token'):
        '/cns/vz-d/home/brain-ber/adosovitskiy/17402132/1/checkpoint.npz',
    ('L', 32, 50, 'token'):
        '/cns/tp-d/home/brain-ber/adosovitskiy/17215117/1/checkpoint.npz',
    ('L', 16, 50, 'token'):
        '/cns/tp-d/home/brain-ber/adosovitskiy/17193867/2/checkpoint.npz',
    ('L', 32, None, 'token'):
        '/cns/lu-d/home/brain-ber/adosovitskiy/17085772/1/checkpoint.npz',
    ('L', 16, None, 'token'):
        '/cns/tp-d/home/brain-ber/adosovitskiy/17192124/1/checkpoint.npz',
}

MODEL_PATH = MODEL_PATHS[(VIT_SIZE, STRIDE, RESNET_SIZE, CLASSIFIER)]


def get_config():
  """Returns the configuration for Cityscapes segmentation."""
  config = ml_collections.ConfigDict()
  config.experiment_name = 'cityscapes_segvit'

  # dataset
  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = (1024, 2048)

  # model
  config.model_name = 'segmenter'
  config.model = ml_collections.ConfigDict()

  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = (STRIDE, STRIDE)

  config.model.backbone = ml_collections.ConfigDict()
  config.model.backbone.type = 'vit_plus'
  config.model.backbone.body = get_backbone_config(config)

  # decoder
  config.model.decoder = ml_collections.ConfigDict()
  config.model.decoder.type = 'linear'

  # training
  config.trainer_name = 'segvit_trainer'
  config.optimizer = 'adam'
  config.optimizer_configs = ml_collections.ConfigDict()
  config.l2_decay_factor = 0.0
  config.max_grad_norm = 1.0
  config.label_smoothing = None
  num_training_epochs = ml_collections.FieldReference(100)
  config.num_training_epochs = num_training_epochs
  config.batch_size = 128
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # learning rate
  steps_per_epoch = _CITYSCAPES_TRAIN_SIZE // config.batch_size
  # setting 'steps_per_cycle' to total_steps basically means non-cycling cosine.
  config.lr_configs = ml_collections.ConfigDict()
  config.lr_configs.learning_rate_schedule = 'compound'
  config.lr_configs.factors = 'constant * cosine_decay * linear_warmup'
  config.lr_configs.warmup_steps = 1 * steps_per_epoch
  config.lr_configs.steps_per_cycle = num_training_epochs * steps_per_epoch
  config.lr_configs.base_learning_rate = 1e-4

  # model and data dtype
  config.model_dtype_str = 'float32'
  config.data_dtype_str = 'float32'

  # init
  config.init_from = ml_collections.ConfigDict()
  config.init_from.codebase = 'bigvision'
  config.init_from.checkpoint_path = MODEL_PATH
  config.init_from.xm = None
  config.init_from.model_prefix_path = ['backbone', 'resformer']

  # logging
  config.write_summary = True  # write TB and/or XM summary
  config.write_xm_measurements = True  # write XM measurements
  config.xprof = False  # Profile using xprof
  config.checkpoint = True  # do checkpointing
  config.checkpoint_steps = 5 * steps_per_epoch

  config.debug_train = False  # debug mode during training
  config.debug_eval = False  # debug mode during eval
  config.log_eval_steps = 200
  return config


def get_backbone_config(config):
  """Get ViT+-style ViT backbone configs."""
  body_config = ml_collections.ConfigDict()
  body_config.type = 'resformer'
  body_config.patches = None
  if RESNET_SIZE == 50:
    body_config.resnet = ml_collections.ConfigDict()
    body_config.resnet.depth = (3, 4, 6, 3)
    body_config.resnet.width = 1.0
  elif RESNET_SIZE is None:
    body_config.patches = {'size': (STRIDE, STRIDE)}
  else:
    raise NotImplementedError('')

  if RESNET_SIZE and STRIDE == 16:
    depth = body_config.resnet.depth
    depth = depth[:-2] + (sum(depth[-2:]),)
    body_config.resnet.depth = depth

  body_config.transformer = ml_collections.ConfigDict()
  body_config.transformer.dropout_rate = 0.1

  if VIT_SIZE == 'B':
    body_config.transformer.mlp_dim = 3072
    body_config.transformer.num_heads = 12
    body_config.transformer.num_layers = 12
    body_config.hidden_size = 768
  elif VIT_SIZE == 'L':
    body_config.transformer.mlp_dim = 4096
    body_config.transformer.num_heads = 16
    body_config.transformer.num_layers = 24
    body_config.hidden_size = 1024
  else:
    raise NotImplementedError('')

  body_config.classifier = CLASSIFIER
  body_config.representation_size = None

  body_config.grid_size = (
      config.dataset_configs.target_size[0] // STRIDE,
      config.dataset_configs.target_size[1] // STRIDE,
  )

  return body_config


def model(hyper, vit_size, stride, resnet_size, classifier):
  """Defines models for sweep."""
  overwrites = []
  if resnet_size == 50:
    depth = (3, 4, 6, 3)
    if stride == 16:
      depth = depth[:-2] + (sum(depth[-2:]),)
    overwrites.append(
        hyper.sweep('config.model.backbone.body.resnet.depth', [depth]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.resnet.width', [1.0]))
    overwrites.append(hyper.sweep('config.model.backbone.body.patches', [None]))
  elif resnet_size is None:
    overwrites.append(
        hyper.sweep('config.model.backbone.body.patches', [{
            'size': (stride, stride)
        }]))
  else:
    raise NotImplementedError('')

  if vit_size == 'B':
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.mlp_dim', [3072]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.num_heads', [12]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.num_layers', [12]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.hidden_size', [768]))
  elif vit_size == 'L':
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.mlp_dim', [4096]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.num_heads', [16]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.transformer.num_layers', [24]))
    overwrites.append(
        hyper.sweep('config.model.backbone.body.hidden_size', [1024]))
  else:
    raise NotImplementedError('')

  overwrites.append(
      hyper.sweep('config.model.backbone.body.classifier', [classifier]))
  overwrites.append(
      hyper.sweep('config.init_from.checkpoint_path',
                  [MODEL_PATHS[(vit_size, stride, resnet_size, classifier)]]))

  return hyper.product(overwrites)


def get_hyper(hyper):
  """Defines the hyper-parameters sweeps for doing grid search."""
  models = hyper.chainit([
      model(hyper, 'B', 16, RESNET_SIZE, CLASSIFIER),
  ])

  return hyper.product([models])