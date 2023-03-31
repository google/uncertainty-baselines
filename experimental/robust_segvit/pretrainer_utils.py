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

"""Pretraining utils."""
from typing import Any, Dict

from absl import logging
import flax
import ml_collections
import numpy as np
from scenic.train_lib_deprecated import train_utils
from tensorflow.io import gfile


def load_bb_config(
    config: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Temporary toy bb config.

  Args:
    config: model config.

  Returns:
    restored_model_cfg: mock model config
  """
  del config
  restored_model_cfg = ml_collections.ConfigDict()
  restored_model_cfg.classifier = "token"

  return restored_model_cfg


def convert_from_pytorch(model: Dict[str, Any],
                         config: ml_collections.ConfigDict) -> Dict[str, Any]:
  """Update the variables names from pytorch convention to jax convention.

  Args:
    model: Dictionary of segmenter parameters as numpy arrays from pytorch model
      in https://github.com/rstrudel/segmenter
    config: model configuration

  Returns:
    jax_state: Dictionary of segmenter parameters as numpy arrays with updated
      names to match segmenter-ub model.

  """
  # TODO(kellybuchanan): add configuration files to compute dims of qkv.
  del config

  jax_state = dict(model)

  for key, tensor in model.items():

    # Decoder
    if "decoder.head." in key:
      del jax_state[key]
      key = key.replace("decoder.head.", "output_projection/")
      if "weight" in key:
        key = key.replace("weight", "kernel")
        tensor = np.transpose(tensor)

    # Encoder
    elif "encoder.head." in key:
      del jax_state[key]
      key = key.replace("encoder.head.", "backbone/Transformer/head/")

    elif "encoder.norm." in key:
      del jax_state[key]
      key = key.replace("encoder.norm.", "backbone/Transformer/encoder_norm/")
      if "weight" in key:
        key = key.replace("weight", "scale")

    elif "encoder.patch_embed.proj." in key:
      del jax_state[key]
      key = key.replace("encoder.patch_embed.proj.", "backbone/embedding/")
      if "weight" in key:
        key = key.replace("weight", "kernel")
        # mapping tf -> torch in timm torch model:
        # w.transpose([3,2,0,1])
        tensor = np.transpose(tensor, [2, 3, 1, 0])

    elif "encoder.pos_embed" in key:
      del jax_state[key]
      key = key.replace("encoder.pos_embed",
                        "backbone/Transformer/posembed_input/pos_embedding")

    elif "encoder.cls_token" in key:
      del jax_state[key]
      key = key.replace("encoder.cls_token", "backbone/cls")

    elif "encoder.blocks" in key:
      del jax_state[key]
      key = key.replace("encoder.", "backbone/Transformer/")
      key = key.replace("blocks.", "encoderblock_")

      key = key.replace(".norm1.", "/LayerNorm_0/")
      key = key.replace(".attn.", "/MultiHeadDotProductAttention_1/")
      key = key.replace(".norm2.", "/LayerNorm_2/")
      key = key.replace(".mlp.fc1.", "/MlpBlock_3/Dense_0/")
      key = key.replace(".mlp.fc2.", "/MlpBlock_3/Dense_1/")

      if "LayerNorm" in key:
        key = key.replace("weight", "scale")
      else:
        key = key.replace("weight", "kernel")

        if "Dense_" in key:
          tensor = np.transpose(tensor)

        elif "qkv" in key:
          # slice query key and value
          dims = tensor.shape[0] // 3
          key1 = key.replace("qkv.", "query/")
          key2 = key.replace("qkv.", "key/")
          key3 = key.replace("qkv.", "value/")

          # mapping tf -> torch in timm torch model:
          # cat[x.flatten(1).T for x in qkv] \in (3072,1024)
          # where q, k, v \in  (1024, 16, 64)
          tensor_masks = [
              np.arange(dims),
              np.arange(dims, dims * 2),
              np.arange(dims * 2, dims * 3)
          ]
          tensor_keys = [key1, key2, key3]
          for key_, tensor_m in zip(tensor_keys, tensor_masks):
            tensor_tmp = tensor[tensor_m].reshape(16, 64, -1).squeeze()
            if tensor_tmp.ndim == 3:
              tensor_tmp = tensor_tmp.transpose([2, 0, 1])
            jax_state[key_] = tensor_tmp

          continue

        elif "proj." in key:
          key = key.replace("proj.", "out/")
          # mapping tf -> torch in timm torch model:
          # w.transpose([2,0,1]) + flatten(1) \in (1024, 1024)
          # where w \in (16, 64, 1024)
          tensor = tensor.reshape(-1, 16, 64).squeeze()

          if tensor.ndim == 3:
            tensor = tensor.transpose([1, 2, 0])
          else:
            tensor = tensor.flatten()

        else:
          raise NotImplementedError(
              "Key {} doesn\'t exist in encoder".format(key))

    else:
      raise NotImplementedError("Key {} doesn\'t exist in model".format(key))

    jax_state[key] = tensor

  return jax_state


def convert_torch_to_jax_checkpoint(
    checkpoint_path: str,
    config: ml_collections.ConfigDict) -> train_utils.TrainState:
  """Converts a segmm segmenter model checkpoint to an scenic train state.

  The model weights are extracted.

  Args:
    checkpoint_path: Path to  checkpoint.
    config: config of pretrained model.

  Returns:
    restored_train_state: Scenic train state with model weights, global step
      and accumulated training time.
  """

  logging.info("Loading torch/numpy checkpoint from %s", checkpoint_path)
  checkpoint_data = np.load(
      gfile.GFile(checkpoint_path, "rb"), allow_pickle=True)[()]
  restored_params = convert_from_pytorch(checkpoint_data, config)

  # Construct tree
  restored_params = flax.traverse_util.unflatten_dict(
      {tuple(k.split("/")[:]): v for k, v in restored_params.items()})

  train_state = train_utils.TrainState()
  # pytype: disable=wrong-arg-types
  restored_train_state = train_state.replace(  # pytype: disable=attribute-error
      optimizer={"target": restored_params},)
  # pytype: enable=wrong-arg-types

  # free memory
  del restored_params
  return restored_train_state
