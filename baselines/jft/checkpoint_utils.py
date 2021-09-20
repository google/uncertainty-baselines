# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Checkpointing utilities for the ViT experiments.

Several functions in this file were ported from
https://github.com/google-research/vision_transformer.
"""

import collections
import dataclasses
import io

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np
import scipy
from tensorflow.io import gfile


def _convert_and_recover_bfloat16(x):
  """Converts to JAX arrays, while correctly loading any bfloat16 arrays."""
  if hasattr(x, "dtype") and x.dtype.type is np.void:
    assert x.itemsize == 2, "Unknown dtype!"
    return jnp.array(x.view(jnp.bfloat16))
  else:
    return jnp.array(x)


def _recover_tree(keys, values):
  """Recovers a tree as a nested dict from flat names and values.

  This function is useful to analyze checkpoints that are without need to access
  the exact source code of the experiment. In particular, it can be used to
  extract an reuse various subtrees of the checkpoint, e.g. subtree of
  parameters.

  Args:
    keys: A list of keys, where "/" is used as separator between nodes.
    values: A list of leaf values.

  Returns:
    A JAX pytree whose structure was recovered from the naming of the keys.
  """
  tree = {}
  sub_trees = collections.defaultdict(list)
  for k, v in zip(keys, values):
    if "/" not in k:
      tree[k] = v
    else:
      k_left, k_right = k.split("/", 1)
      sub_trees[k_left].append((k_right, v))
  for k, kv_pairs in sub_trees.items():
    k_subtree, v_subtree = zip(*kv_pairs)
    tree[k] = _recover_tree(k_subtree, v_subtree)
  return tree


def load_checkpoint(tree, path):
  """Loads JAX pytrees that were stored on disk in a NumPy `.npz` file.

  Args:
    tree: Optional JAX pytree to be restored. If None, then the tree will be
      recovered from the naming scheme used within the checkpoint.
    path: A path to the checkpoint.

  Returns:
    A JAX pytree with the same structure as `tree`, but with the leaf values
    restored from the saved checkpoint.
  """
  with gfile.GFile(path, "rb") as f:
    data = f.read()
  keys, values = zip(
      *list(np.load(io.BytesIO(data), allow_pickle=False).items()))
  # NOTE: NumPy loses any bfloat16 dtypes when saving, so we recover them here.
  values = jax.tree_util.tree_map(_convert_and_recover_bfloat16, values)
  if tree:
    treedef = jax.tree_util.tree_structure(tree)
    tree = jax.tree_util.tree_unflatten(treedef, values)
  else:
    tree = _recover_tree(keys, values)
  return tree


def _traverse_with_names(tree):
  """Traverses nested dicts/dataclasses and emits (leaf_name, leaf_val)."""
  if dataclasses.is_dataclass(tree):
    tree = flax.serialization.to_state_dict(tree)
  if isinstance(tree, dict) or isinstance(tree, flax.core.FrozenDict):
    keys = sorted(tree.keys())
    for key in keys:
      for path, v in _traverse_with_names(tree[key]):
        yield (key + "/" + path).rstrip("/"), v
  else:
    yield "", tree


def _tree_flatten_with_names(tree):
  """Populates tree_flatten with leaf names.

  This function populates the output of jax.tree_util.tree_flatten with leaf
  names, using a custom traversal that produces names.

  Args:
    tree: A JAX PyTree.

  Returns:
    A list of values with names: [(name, value), ...].
  """
  vals, tree_def = jax.tree_flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def save_checkpoint(tree, path, step_for_copy=None):
  """Saves the values of JAX pytrees to disk in a NumPy `.npz` file.

  Args:
    tree: A JAX pytree to be saved.
    path: A path to save the checkpoint.
    step_for_copy: Optional integer that, when not None, will be used to save a
      copy of the checkpoint with the name `path-{step_for_copy}`.
  """
  # NOTE: In general, this could be greatly simplified as follows. However, we
  # currently need to store the leaf names as well in order to be able to load
  # and reconstruct the tree directly from the checkpoint when initialized a
  # subset of a model from a pretrained model for fine tuning.
  # ```
  # values, _ = jax.tree_util.tree_flatten(tree)
  # io_buffer = io.BytesIO()
  # np.savez(io_buffer, *values)
  # ```
  names_and_vals, _ = _tree_flatten_with_names(tree)
  io_buffer = io.BytesIO()
  np.savez(io_buffer, **{k: v for k, v in names_and_vals})

  # In order to be robust to interruptions during saving, we first save the
  # checkpoint to a temporary file, and then rename it to the actual path name.
  path_tmp = path + "-TEMPORARY"
  with gfile.GFile(path_tmp, "wb") as f:
    f.write(io_buffer.getvalue())
  gfile.rename(path_tmp, path, overwrite=True)

  if step_for_copy is not None:
    gfile.copy(path, f"{path}-{step_for_copy:09d}", overwrite=True)


def _flatten_dict(d, parent_key="", sep="/"):
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.abc.Mapping):
      items.extend(_flatten_dict(v, path, sep=sep).items())
    else:
      items.append((path, v))

  # Keeps the empty dict if it was set explicitly.
  if parent_key and not d:
    items.append((parent_key, {}))

  return dict(items)


def _inspect_params(*,
                    params,
                    expected,
                    fail_if_extra=True,
                    fail_if_missing=True):
  """Inspects whether the params are consistent with the expected keys."""
  params_flat = _flatten_dict(params)
  expected_flat = _flatten_dict(expected)
  missing_keys = expected_flat.keys() - params_flat.keys()
  extra_keys = params_flat.keys() - expected_flat.keys()

  # Adds back empty dict explicitly, to support layers without weights.
  # Context: FLAX ignores empty dict during serialization.
  empty_keys = set()
  for k in missing_keys:
    if isinstance(expected_flat[k], dict) and not expected_flat[k]:
      params[k] = {}
      empty_keys.add(k)
  missing_keys -= empty_keys

  if empty_keys:
    logging.warning("Inspect recovered empty keys:\n%s", empty_keys)
  if missing_keys:
    logging.info("Inspect missing keys:\n%s", missing_keys)
  if extra_keys:
    logging.info("Inspect extra keys:\n%s", extra_keys)

  if (missing_keys and fail_if_missing) or (extra_keys and fail_if_extra):
    raise ValueError(f"Missing params from checkpoint: {missing_keys}.\n"
                     f"Extra params in checkpoint: {extra_keys}.\n"
                     f"Restored params from checkpoint: {params_flat.keys()}.\n"
                     f"Expected params from code: {expected_flat.keys()}.")
  return params


def _tree_map_with_names(f, tree, *rest):
  """Performs a tree map with a filter on the leaf path name.

  Args:
    f: A function accepting a name (path-like "a/b/c"), a tree, and an optional
      additional list of trees.
    tree: The tree of parameters for which `f` should be applied.
    *rest: More trees of the exact same structure.

  Returns:
    A tree identical in structure to `tree` and `*rest` but with the leaves the
    result of calling `f` on corresponding name/leaves in `tree` and `*rest`.
  """
  names_and_vals, tree_def = _tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*_tree_flatten_with_names(t)[0]))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)


def _reinit(restored_params, init_params, to_reinit):
  """Reinitializes a subset of the parameters in the restored parameter tree."""
  f = lambda name, restored, init: init if name in to_reinit else restored
  return _tree_map_with_names(f, restored_params, init_params)


def load_from_pretrained_checkpoint(init_params, pretrained_path,
                                    model_representation_size, model_classifier,
                                    reinit_params):
  """Initializes (part of) a model from a pretrained checkpoint for fine tuning.

  Args:
    init_params: Tree of (possibly randomly) initialized parameters for the
      model. The structure will be kept, and a subset of the values will be
      replaced with values loaded from the pretrained checkpoint.
    pretrained_path: File pointing to pretrained checkpoint stored in NumPy
      `.npz` file.
    model_representation_size: Optional integer representation size
      hyperparameter for the model. If None, then the representation layer in
      the checkpoint will be removed (if present).
    model_classifier: String containing the classifier hyperparameter used for
      the model.
    reinit_params: List of parameter names to reinitialize.

  Returns:
    A tree of parameters with the same structure as `init_params`, but loaded
    with pretrained weights from `pretrained_path` and adapted accordingly.
  """
  params = load_checkpoint(None, pretrained_path)
  if "opt" in params:
    params = params["opt"]["target"]
  restored_params = _inspect_params(
      params=params,
      expected=init_params,
      fail_if_extra=False,
      fail_if_missing=False)

  # The following allows implementing fine-tuning head variants depending on the
  # value of `representation_size` in the fine-tuning job:
  # - `None`: drop the whole head and attach a nn.Linear.
  # - Same number as in pre-training: keep the head but reset the last
  #    layer (logits) for the new task.
  if model_representation_size is None:
    if "pre_logits" in restored_params:
      logging.info("load_pretrained: drop-head variant")
      del restored_params["pre_logits"]

  if reinit_params:
    restored_params = _reinit(restored_params, init_params, reinit_params)

  if "posembed_input" in restored_params.get("Transformer", {}):
    # Rescale the grid of position embeddings. Param shape is (1,N,1024)
    posemb = restored_params["Transformer"]["posembed_input"]["pos_embedding"]
    posemb_new = init_params["Transformer"]["posembed_input"]["pos_embedding"]
    if posemb.shape != posemb_new.shape:
      logging.info("load_pretrained: resized variant: %s to %s", posemb.shape,
                   posemb_new.shape)
      ntok_new = posemb_new.shape[1]

      if model_classifier == "token":
        posemb_tok, posemb_grid = posemb[:, :1], posemb[0, 1:]
        ntok_new -= 1
      else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]

      gs_old = int(np.sqrt(len(posemb_grid)))
      gs_new = int(np.sqrt(ntok_new))
      logging.info("load_pretrained: grid-size from %s to %s", gs_old, gs_new)
      posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)

      zoom = (gs_new / gs_old, gs_new / gs_old, 1)
      posemb_grid = scipy.ndimage.zoom(posemb_grid, zoom, order=1)
      posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
      posemb = jnp.array(np.concatenate([posemb_tok, posemb_grid], axis=1))
      restored_params["Transformer"]["posembed_input"]["pos_embedding"] = posemb

  return restored_params
