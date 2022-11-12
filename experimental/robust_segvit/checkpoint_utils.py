# load checkpoints
from scenic.train_lib_deprecated import train_utils
from pretrainer_utils import convert_torch_to_jax_checkpoint  # local file import from experimental.robust_segvit
from scenic.train_lib_deprecated import pretrain_utils
from pretrainer_utils import convert_vision_transformer_to_scenic  # local file import from experimental.robust_segvit


def load_checkpoints_eval(config, model, train_state, workdir):
  checkpoint_configs = config.get('checkpoint_configs', False)
  if checkpoint_configs:
    # Load torch weights
    if 'torch' in checkpoint_configs.checkpoint_format:

      bb_train_state = convert_torch_to_jax_checkpoint(
        checkpoint_path=checkpoint_configs.checkpoint_path,
        config=checkpoint_configs)

      train_state = model.init_backbone_from_train_state(
        train_state,
        bb_train_state,
        config,
        checkpoint_configs
      )
      del bb_train_state

    # Load weights in checkpoint_path or workdir
    else:
      checkpoint_path = checkpoint_configs.get('checkpoint_path', workdir)
      train_state, _ = train_utils.restore_checkpoint(
        checkpoint_path, train_state)
  return train_state


def load_checkpoints_backbone(config, model, train_state, workdir):
  del workdir
  # TODO(kellybuchanan): check out partial loader in
  # https://github.com/google/uncertainty-baselines/commit/083b1dcc52bb1964f8917d15552ece8848d582ae#
  restored_model_cfg = config.get('pretrained_backbone_configs')

  # Load pretrained backbone
  if restored_model_cfg.checkpoint_format in ('ub', 'big_vision', 'scenic'):
    # load params from checkpoint
    bb_train_state = pretrain_utils.convert_big_vision_to_scenic_checkpoint(
        checkpoint_path=restored_model_cfg.checkpoint_path,
        convert_to_linen=False)

    train_state = model.init_backbone_from_train_state(
        train_state,
        bb_train_state,
        config,
        restored_model_cfg,
        model_prefix_path=['backbone'])
    # Free unnecessary memory.
    del bb_train_state
    # Loader from scenic
  elif restored_model_cfg.checkpoint_format in ('vision_transformer'):
      # load params from checkpoint
      bb_train_state = convert_vision_transformer_to_scenic(checkpoint_path=restored_model_cfg.checkpoint_path, convert_to_linen=False)

      train_state = model.init_backbone_from_train_state(
        train_state,
        bb_train_state,
        config,
        restored_model_cfg,
        model_prefix_path=['backbone'])

      # Free unnecessary memory.
      del bb_train_state
  else:
    raise NotImplementedError('')
  return train_state
