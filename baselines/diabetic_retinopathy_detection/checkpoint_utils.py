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

"""Checkpointing utilities for the ViT experiments.

Several functions in this file were ported from
https://github.com/google-research/vision_transformer.
"""

import collections
import dataclasses
import io
from typing import Any, Iterable, MutableMapping, Optional

from absl import logging
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import scipy
import tensorflow as tf

Params = MutableMapping[str, Any]


@dataclasses.dataclass
class CheckpointData:
  """Container class for data stored and loaded into checkpoints."""
  train_loop_rngs: jnp.ndarray
  optimizer: flax.optim.Optimizer
  accumulated_train_time: float
  fixed_model_states: Optional[Params] = None


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
  with tf.io.gfile.GFile(path, "rb") as f:
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
  vals, tree_def = jax.tree.flatten(tree)

  # "Fake" token tree that is use to track jax internal tree traversal and
  # adjust our custom tree traversal to be compatible with it.
  tokens = range(len(vals))
  token_tree = tree_def.unflatten(tokens)
  val_names, perm = zip(*_traverse_with_names(token_tree))
  inv_perm = np.argsort(perm)

  # Custom traversal should visit the same number of leaves.
  assert len(val_names) == len(vals)

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)], tree_def


def save_checkpoint(tree: Params, path: str,
                    step_for_copy: Optional[int] = None) -> None:
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
  with tf.io.gfile.GFile(path_tmp, "wb") as f:
    f.write(io_buffer.getvalue())
  tf.io.gfile.rename(path_tmp, path, overwrite=True)

  if step_for_copy is not None:
    tf.io.gfile.copy(path, f"{path}-{step_for_copy:09d}", overwrite=True)


def checkpoint_trained_model(
    checkpoint_data: CheckpointData,
    path: str,
    step_for_copy: Optional[int] = None) -> None:
  """Saves all information pertaining to a trained model in .npz format.

  Args:
    checkpoint_data: CheckpointData instance.
    path: A path to save the checkpoint.
    step_for_copy: Optional integer that, when not None, will be used to save a
      copy of the checkpoint with the name `path-{step_for_copy}`.
  """
  # TODO(zmariet, dusenberrymw): Remove intermediate `checkpoint_extra` dict.
  tree = dict(
      opt=checkpoint_data.optimizer,
      extra=dict(
          rngs_loop=checkpoint_data.train_loop_rngs,
          accum_train_time=checkpoint_data.accumulated_train_time),
      )
  if checkpoint_data.fixed_model_states is not None:
    tree["states"] = checkpoint_data.fixed_model_states
  save_checkpoint(tree, path, step_for_copy)


def _flatten_jax_params_dict(d: Params, parent_key: str = "",
                             sep: str = "/") -> Params:
  """Flattens a dictionary, keeping empty leaves."""
  items = []
  for k, v in d.items():
    path = parent_key + sep + k if parent_key else k
    if isinstance(v, collections.abc.Mapping):
      items.extend(_flatten_jax_params_dict(v, path, sep=sep).items())
    else:
      items.append((path, v))

  # Keeps the empty dict if it was set explicitly.
  if parent_key and not d:
    items.append((parent_key, {}))

  return dict(items)


def _unflatten_jax_params_dict(flat_params: Params, sep: str = "/") -> Params:
  """Unflattens a dictionary that maps strings to non-dictionaries.

  Args:
    flat_params: A dictionary mapping strings to non-dictionary values.
    sep: Separator indicating key hierarchy in `flat_params`. For example,
      unflattening {"a/b": 1} with separator "/" will yield {"a": {"b": 1}}.

  Returns:
    A dictionary mapping strings to arbitrary values (including dictionaries).
  """
  tuple_to_value = {tuple(k.split(sep)): v for k, v in flat_params.items()}
  return flax.traverse_util.unflatten_dict(tuple_to_value)


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


def restore_from_pretrained_params(init_params, loaded_params,
                                   model_representation_size, model_classifier,
                                   reinit_params):
  """Initializes (some) model parameters based on pretrained parameters.

  Args:
    init_params: Tree of (possibly randomly) initialized parameters for the
      model. The structure will be kept, and a subset of the values will be
      replaced with values loaded from the pretrained checkpoint.
    loaded_params: Tree with pretrained weights.
    model_representation_size: Optional integer representation size
      hyperparameter for the model. If None, then the representation layer in
      the checkpoint will be removed (if present).
    model_classifier: String containing the classifier hyperparameter used for
      the model.
    reinit_params: List of parameter names to reinitialize.

  Returns:
    A tree of parameters with the same structure as `init_params`, but loaded
    with pretrained weights in `loaded_params` and adapted accordingly.
  """
  if "opt" in loaded_params:
    loaded_params = loaded_params["opt"]["target"]
  restored_params = adapt_upstream_architecture(init_params, loaded_params)

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
    # Rescale the grid of position embeddings. Param shape is (1,N,1024).
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


def maybe_load_checkpoint(train_loop_rngs: jnp.ndarray,
                          save_checkpoint_path: str,
                          init_optimizer: flax.optim.Optimizer,
                          init_params: Params,
                          init_fixed_model_states: Optional[Params],
                          default_reinit_params: Iterable[str],
                          config: ml_collections.ConfigDict) -> CheckpointData:
  """Loads a model from an existing checkpoint if so indicated by the config.

  Whether to resume training, initialize from a previous checkpoint, or do
  nothing is set by the `config` ConfigDict, based on the existence of fields
  `resume` (resume training) or `model_init` (initialize from pretrained
  checkpoint).

  When resuming training, both the model weights and optimizer
  state (including the training step) are restored. When initializing, only
  the model parameters are updated.

  The way in which initializing is prioritized in the following way:
  1. Always resume from an existing checkpoint, e.g. resume a finetune job.
  2. Resume from a previous checkpoint, e.g. start a cooldown training job.
  3. Initialize model from something, e,g, start a fine-tuning job.
  4. Do nothing (training from scratch).

  Args:
    train_loop_rngs: unreplicated jax.PRNGKey.
    save_checkpoint_path: File pointing to pretrained checkpoint stored in NumPy
      `.npz` file.
    init_optimizer: flax.Optimizer to be updated.
    init_params: Tree of (possibly randomly) initialized parameters for the
      model.
    init_fixed_model_states: Optional pytree of non-trainable parameters.
      Currently only passed when using SNGP models.
    default_reinit_params: List of parameter names to reinitialize if not
      provided by the config file.
    config: ConfigDict which contains fields indicating if, and how, to load an
      available checkpoint into the optimizer. If resuming from a previous
      checkpoint *to start a cooldown job*, the flag `resume` must be set. If
      initializing a (subset of) model parameters to start a file tuning job,
      fields `model_init`, `representation_size` and `classifier` must be set.

  Returns:
    A CheckpointData instance containing a new rng key, the new optimizer state,
    the new untrainable parameters (if resuming from a checkpoint), and a
    dictionary of information about the reloaded state.
  """
  optimizer = init_optimizer
  fixed_model_states = init_fixed_model_states

  accum_train_time = 0.0
  # TODO(dusenberrymw, zmariet): Directly return an unreplicated rng and the
  # cumulative training time instead of storing them in `checkpoint_extra`.
  checkpoint_extra = dict(
      accum_train_time=accum_train_time,
      rngs_loop=flax_utils.replicate(train_loop_rngs))

  # Parse config file to figure out which setting we are in.
  resume_from_checkpoint = (
      (save_checkpoint_path is not None and
       tf.io.gfile.exists(save_checkpoint_path))
      or config.get("resume") is not None)
  reinitialize_model = config.get(
      "model_init") is not None and not resume_from_checkpoint

  if resume_from_checkpoint:
    logging.info("Resume training from checkpoint...")
    # Always prioritize loading from a checkpoint from the current training job.
    if save_checkpoint_path and tf.io.gfile.exists(save_checkpoint_path):
      resume_checkpoint_path = save_checkpoint_path
    # Otherwise, we reload from a previous checkpoint provided by the config.
    else:
      resume_checkpoint_path = config.resume

    checkpoint_tree = {"opt": init_optimizer, "extra": checkpoint_extra}
    if init_fixed_model_states is not None:
      checkpoint_tree["states"] = init_fixed_model_states
    checkpoint = load_checkpoint(checkpoint_tree, resume_checkpoint_path)
    optimizer, checkpoint_extra = checkpoint["opt"], checkpoint["extra"]
    fixed_model_states = checkpoint.get("states", None)

  elif reinitialize_model:
    logging.info("Initialize model...")
    reinit_params = config.get("model_reinit_params", default_reinit_params)
    logging.info("Reinitializing these parameters: %s", reinit_params)

    loader = lambda path: load_checkpoint(tree=None, path=path)
    loaded_params = loader(config.model_init)

    loaded_params = restore_from_pretrained_params(
        init_params=init_params,
        loaded_params=loaded_params,
        model_representation_size=config.model.representation_size,
        model_classifier=config.model.classifier,
        reinit_params=reinit_params)

    optimizer = init_optimizer.replace(target=loaded_params)
    if jax.process_index() == 0:
      logging.info("Restored parameter overview:")
      parameter_overview.log_parameter_overview(loaded_params)

  else:
    logging.info("No checkpoint to recover from; using default initialization.")

  return CheckpointData(
      optimizer=optimizer,
      fixed_model_states=fixed_model_states,
      train_loop_rngs=checkpoint_extra["rngs_loop"],
      accumulated_train_time=checkpoint_extra["accum_train_time"])


def adapt_upstream_architecture(
    init_params: Params, loaded_params: Params) -> Params:
  """Align upstream parameters with those expected by the current architecture.

  This function converts the loaded architecture into the architecture expected
  by `init_params` when using a pretrained model of a different architecture
  (e.g., finetuning an SGNP model based on an upstream deterministic model).

  This function relies upon the fact that the parameters in `loaded_params`
  that should be kept will have the same name in `init_params`. If that is not
  the case, loaded parameter values will be lost.

  Args:
    init_params: Tree of (possibly randomly) initialized parameters for the
      model.
    loaded_params: Tree of parameters loaded from a checkpoint (in practice, the
      upstream model).

  Returns:
    A tree with similar structure to that of `init_params`, where values match
    those of `loaded_params` when possible.
  """
  loaded_flat = _flatten_jax_params_dict(loaded_params)
  init_flat = _flatten_jax_params_dict(init_params)

  missing_keys = set(init_flat.keys()) - set(loaded_flat.keys())
  extra_keys = set(loaded_flat.keys()) - set(init_flat.keys())

  logging.info("Deleting %s from checkpoint architecture.", extra_keys)
  logging.info("Adding %s from checkpoint architecture.", missing_keys)

  # Remove extra parameters.
  for extra_key in extra_keys:
    del loaded_flat[extra_key]

  # Add missing parameters using initialized values.
  for missing_key in missing_keys:
    loaded_flat[missing_key] = init_flat[missing_key]

  return _unflatten_jax_params_dict(loaded_flat)
