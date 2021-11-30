"""
Include utils for pretraining

"""
import importlib
import os
import ml_collections

def load_relative_config(relative_fpath):
  """ Reads config of model in ub.

  Args:
   relative_fpath: path of config file relative to its location in ub.

  """
  # loader = importlib.machinery.SourceFileLoader('get_config', os.path.abspath(relative_fpath))
  # config = loader.load_module()
  # config_module_spec = importlib.util.spec_from_file_location('get_config', os.path.abspath("../../baselines/jft/experiments/imagenet21k_vit_base16.py"))
  # config_module = importlib.util.module_from_spec(config_module_spec)
  # config_module_spec.loader.exec_module(config_module)
  # return config
  raise NotImplementedError("")


def load_bb_config(config):
  """ Temporary toy bb config.

  Args:
    config: model config.

  Returns:
    restored_model_cfg: mock model config
  """
  restored_model_cfg = ml_collections.ConfigDict()
  restored_model_cfg.patches = ml_collections.ConfigDict()
  restored_model_cfg.patches.size = [16, 16]
  restored_model_cfg.classifier = 'token'
  # if config.pretrained_backbone_configs.type == 'base':
  # restored_model_cfg.model.transformer.dropout_rate = 0.1

  #TODO(kellybuchanan): calculate grid given config
  restored_model_cfg.patches.grid = [224//16, 224//16]

  return restored_model_cfg
