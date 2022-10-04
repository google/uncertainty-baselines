import ml_collections
import os
import datetime

_CITYSCAPES_TRAIN_SIZE = 2975
_CITYSCAPES_TRAIN_SIZE_SPLIT = 146

# Model specs.
CHECKPOINT_ORIGIN = 'torch-segmm'
VIT_SIZE = 'L'
STRIDE = 16
RESNET_SIZE = None
CLASSIFIER = 'token'
target_size = (768, 768)
EXPERIMENTID = 'torch-segmm-1'

# Upstream
CHECKPOINT_PATHS = {
    ('torch-segmm', 'L', 16, None, 'token', 'torch-segmm-1'):
        'gs://ub-ekb/seg_l16_linear/checkpoint_model.npy',
}


CHECKPOINT_PATH = CHECKPOINT_PATHS[(CHECKPOINT_ORIGIN, VIT_SIZE, STRIDE,
                                    RESNET_SIZE, CLASSIFIER, EXPERIMENTID)]

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


def get_config(runlocal=''):
  """Returns the configuration for Cityscapes segmentation."""

  runlocal = bool(runlocal)

  config = ml_collections.ConfigDict()
  config.experiment_name = 'cityscapes_segmenter_torch_eval'

  # Dataset.
  config.dataset_name = 'cityscapes'
  config.dataset_configs = ml_collections.ConfigDict()
  config.dataset_configs.target_size = (1024, 2048)
  config.dataset_configs.train_split = 'train'
  config.dataset_configs.dataset_name = ''  # name of ood dataset to evaluate

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
  config.model.backbone.dropout_rate = 0.0
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
  config.batch_size = 64
  config.rng_seed = 0
  config.focal_loss_gamma = 0.0

  # Learning rate.
  config.steps_per_epoch = _CITYSCAPES_TRAIN_SIZE // config.get_ref(
      'batch_size')
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
  config.eval_mode = True
  config.eval_configs = ml_collections.ConfigDict()
  config.eval_configs.mode = 'segmm'
  config.eval_configs.window_stride = 512
  config.model.input_shape = target_size

  # Eval parameters for robustness
  config.eval_label_shift = True
  config.eval_covariate_shift = True
  config.eval_robustness_configs = ml_collections.ConfigDict()
  config.eval_robustness_configs.auc_online = True
  config.eval_robustness_configs.method_name = 'nmlogit'
  config.eval_robustness_configs.num_top_k = 1

  # Load checkpoint
  config.checkpoint_configs = ml_collections.ConfigDict()
  config.checkpoint_configs.checkpoint_format = CHECKPOINT_ORIGIN
  config.checkpoint_configs.checkpoint_path = CHECKPOINT_PATH
  config.checkpoint_configs.classifier = 'token'

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
    config.target_size = (128, 128)
    config.batch_size = 8
    config.num_training_epochs = 5
    config.warmup_steps = 0
    config.dataset_configs.train_split = 'train[:5%]'
    config.steps_per_epoch = _CITYSCAPES_TRAIN_SIZE_SPLIT // config.get_ref(
        'batch_size')

  return config


def checkpoint(hyper, backbone_origin, vit_size, stride, resnet_size,
               classifier, upstream_task):
  """Defines checkpoints for sweep."""
  overwrites = []
  if resnet_size is not None:
    raise NotImplementedError('')
  else:
    overwrites.append(
        hyper.sweep('config.model.patches', [{'size': (stride, stride)}]))

  if vit_size == 'B':
    overwrites.append(
        hyper.sweep('config.model.backbone.mlp_dim', [3072]))
    overwrites.append(
        hyper.sweep('config.model.backbone.num_heads', [12]))
    overwrites.append(
        hyper.sweep('config.model.backbone.num_layers', [12]))
    overwrites.append(
        hyper.sweep('config.model.backbone.hidden_size', [768]))
  elif vit_size == 'L':
    overwrites.append(
        hyper.sweep('config.model.backbone.mlp_dim', [4096]))
    overwrites.append(
        hyper.sweep('config.model.backbone.num_heads', [16]))
    overwrites.append(
        hyper.sweep('config.model.backbone.num_layers', [24]))
    overwrites.append(
        hyper.sweep('config.model.backbone.hidden_size', [1024]))
  else:
    raise NotImplementedError('')

  overwrites.append(
      hyper.sweep('config.checkpoint_configs.checkpoint_format',
                  [backbone_origin]))
  overwrites.append(
      hyper.sweep('config.checkpoint_configs.checkpoint_path', [
          CHECKPOINT_PATHS[(backbone_origin, vit_size, stride, resnet_size,
                            classifier, upstream_task)]
      ]))

  return hyper.product(overwrites)


def get_sweep(hyper):
  """Defines the parameters used to compare multiple metrics during eval."""

  checkpoints = hyper.chainit([
      checkpoint(hyper, 'ub', 'L', 16, None, 'token', 'torch-segmm-1'),
  ])

  return hyper.product([checkpoints])
