# Copyright 2021 The Scenic Authors.
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

"""Utility functions for using pretrained models.

Edited from scenic/train_lib/pretrain_utils.py
"""

import collections
import os
import re
from typing import Any, Dict, Mapping, List, Optional, Union, Tuple

from absl import logging
import flax
from flax.training import checkpoints
import jax
import numpy as np

from scenic.train_lib import train_utils, pretrain_utils
from tensorflow.io import gfile

# JAX team is working on type annotation for pytree:
# https://github.com/google/jax/issues/1555
PyTree = Union[Mapping[str, Mapping], Any]


def convert_ub_to_scenic_checkpoint(
    checkpoint_path: str,
    train_state: Optional[train_utils.TrainState] = None,
    convert_to_linen: bool = True) -> train_utils.TrainState:
  """Converts a BigVision checkpoint to a scenic train state.

  The model weights, global step and accumulated train time are extracted.
  Optimizer state, such as the momentum, is not extracted.

  Args:
    checkpoint_path: Path to BigVision checkpoint.
    train_state: A Scenic TrainState object.
    convert_to_linen: Whether to convert to Linen format.

  Returns:
    restored_train_state: Scenic train state with model weights, global step
      and accumulated training time.
  """

  def unflatten_dict(flattened: Dict[str, Any],
                     separator: str = '/',
                     leaf_idx: int = -1) -> Dict[str, Any]:
    unflattened = {}
    for k, v in flattened.items():
      subtree = unflattened
      if leaf_idx != 0:
        path = k.split(separator)[:leaf_idx]
      else:
        path = k.split(separator)
      for k2 in path[:-1]:
        if k2 not in subtree:
          subtree[k2] = {}
        subtree = subtree[k2]
      subtree[path[-1]] = v
    return unflattened

  logging.info('Loading bigvision checkpoint from %s', checkpoint_path)
  checkpoint_data = np.load(gfile.GFile(checkpoint_path, 'rb'))
  tree = unflatten_dict(checkpoint_data, separator='/', leaf_idx=0)

  import pdb; pdb.set_trace()
  restored_params = tree['opt']['target']
  if convert_to_linen:
    restored_params = checkpoints.convert_pre_linen(restored_params)
  restored_params = dict(restored_params)
  if train_state:
    restored_params = pretrain_utils.inspect_params(
        expected_params=train_state.optimizer.target,
        restored_params=restored_params,
        fail_if_extra=False,
        fail_if_missing=False,
        fail_if_shapes_mismatch=False)
  else:
    train_state = train_utils.TrainState()
  # pytype: disable=wrong-arg-types
  restored_train_state = train_state.replace(  # pytype: disable=attribute-error
      global_step=int(tree['opt']['state']['step']),
      optimizer={'target': restored_params},
      accum_train_time=int(tree['extra']['accum_train_time']))
  # pytype: enable=wrong-arg-types

  return restored_train_state
