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

"""Checkpointing utilities for the ViT experiments.

Several functions in this file were ported from
https://github.com/google-research/vision_transformer.
"""

import collections
from concurrent.futures import thread
import dataclasses
import io
from typing import Any, Iterable, Mapping, MutableMapping, Optional

from absl import logging
from clu import parameter_overview
import flax
import flax.jax_utils as flax_utils
import flax.optim
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
from tensorflow.io import gfile

Params = MutableMapping[str, Any]
PyTree = Any


@dataclasses.dataclass
class CheckpointData:
  """Container class for data stored and loaded into checkpoints."""
  train_loop_rngs: jnp.ndarray
  optimizer: flax.optim.Optimizer
  accumulated_train_time: float
  fixed_model_states: Optional[Params] = None


def _recover_bfloat16(x):
  """Recovers the dtype of any bfloat16 array without making a copy."""
  if hasattr(x, "dtype") and x.dtype.type is np.void:
    assert x.itemsize == 2, "Unknown dtype!"
    return x.view(jnp.bfloat16)
  else:
    return x


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


def _read_file(path: str, pool_size: int, buf_size: int) -> bytearray:
  """Reads the contents of a file in parallel, if possible.

  Args:
    path: A path to a file.
    pool_size: Number of threads to use to read chunks of the file in parallel,
      if possible.
    buf_size: The size of each file chunk to be read in parallel, if possible.
      Defaults to 128M buffer sizes.

  Returns:
    The contents of the file.
  """
  with gfile.GFile(path, "rb") as f:
    if f.seekable():
      read_in_parallel = True
      num_bufs = f.size() / buf_size
      logging.debug("num_bufs: %d", num_bufs)
      data = bytearray(f.size())  # Empty array, to be filled below.
    else:
      read_in_parallel = False
      data = f.read()

  if read_in_parallel:

    # Chunked reading from flax.training.checkpoints.restore_checkpoint.
    def read_chunk(i):
      with gfile.GFile(path, "rb") as f:
        f.seek(i * buf_size)
        buf = f.read(buf_size)
        if buf:
          data[i * buf_size:i * buf_size + len(buf)] = buf
        return len(buf) / buf_size

    # Fill in the empty `data` array in parallel.
    pool = thread.ThreadPoolExecutor(pool_size)
    results = pool.map(read_chunk, range(int(num_bufs) + 1))
    pool.shutdown(wait=False)
    logging.debug("results: %s", list(results))
  return data


def load_checkpoint(tree: Optional[Params],
                    path: str,
                    read_in_parallel: bool = True,
                    pool_size: int = 32,
                    buf_size: int = 128 << 20) -> Params:
  """Loads JAX pytrees that were stored on disk in a NumPy `.npz` file.

  Args:
    tree: Optional JAX pytree to be restored. If None, then the tree will be
      recovered from the naming scheme used within the checkpoint.
    path: A path to the checkpoint.
    read_in_parallel: Whether or not to read chunks of the checkpoint file in
      parallel, if possible. Recommend only setting this to False in a
      RAM-constrained environment.
    pool_size: Number of threads to use to read chunks of the checkpoint file in
      parallel, if possible.
    buf_size: The size of each file chunk to be read in parallel, if possible.
      Defaults to 128M buffer sizes.

  Returns:
    A JAX pytree with the same structure as `tree`, but with the leaf values
    restored from the saved checkpoint.
  """
  if read_in_parallel:
    file = io.BytesIO(_read_file(path, pool_size=pool_size, buf_size=buf_size))
  else:
    file = gfile.GFile(path, "rb")
  with np.load(file, allow_pickle=False) as data:
    values = list(data.values())
    if not tree:
      keys = list(data.keys())
  file.close()
  del file  # Free up RAM.
  # NOTE: NumPy loses any bfloat16 dtypes when saving, so we recover them here.
  values = jax.tree_util.tree_map(_recover_bfloat16, values)
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

  return [(val_names[i], v) for i, v in zip(inv_perm, vals)]


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
  # subset of a model from a pretrained model for fine tuning. Also, we use
  # gfile to support multiple storage backends.
  # ```
  # values, _ = jax.tree_util.tree_flatten(tree)
  # io_buffer = io.BytesIO()
  # np.savez(io_buffer, *values)
  # ```
  names_and_vals = {k: v for k, v in _tree_flatten_with_names(tree)}
  io_buffer = io.BytesIO()
  np.savez(io_buffer, **names_and_vals)

  # In order to be robust to interruptions during saving, we first save the
  # checkpoint to a temporary file, and then rename it to the actual path name.
  path_tmp = path + "-TEMPORARY"
  with gfile.GFile(path_tmp, "wb") as f:
    f.write(io_buffer.getvalue())
  gfile.rename(path_tmp, path, overwrite=True)

  if step_for_copy is not None:
    gfile.copy(path, f"{path}-{step_for_copy:09d}", overwrite=True)


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


# Forked from third_party/py/scenic/projects/baselines/clip/model.py.
def _convert_attn_layers(params: Mapping[str, np.ndarray],
                         dim_head: int = 64) -> PyTree:
  """Convert attention parameters."""
  new_params = {}
  processed_attn_layers = []
  for k, v in params.items():
    if "attn." in k:
      base = k[:k.rindex("attn.")+5]
      if base in processed_attn_layers:
        continue
      processed_attn_layers.append(base)
      dim = params[base + "out_proj.bias"].shape[-1]
      heads = dim // dim_head
      new_params[base + "out.weight"] = params[
          base + "out_proj.weight"].T.reshape(heads, dim_head, dim)
      new_params[base + "out.bias"] = params[base + "out_proj.bias"]
      qkv_bias = params[base + "in_proj_bias"].reshape(3, heads, dim_head)
      qkv_kernel = np.transpose(params[base + "in_proj_weight"].reshape(
          3, heads, dim_head, dim), (0, 3, 1, 2))
      for i, kk in enumerate(("query", "key", "value")):
        new_params[base + f"{kk}.bias"] = qkv_bias[i]
        new_params[base + f"{kk}.weight"] = qkv_kernel[i]
    else:
      new_params[k] = v
  return new_params


def _convert_vars(torch_vars: Mapping[str, np.ndarray],
                  dim_head: int = 64) -> PyTree:
  """Convert torch parameters to flax parameters."""
  # Expand QKV dense input projection to separate Q, K, V projections
  # and fix shape/transposing of attention layers.
  torch_vars = _convert_attn_layers(torch_vars, dim_head)
  flax_vars = {}
  torch_vars.pop("context_length", None)
  torch_vars.pop("input_resolution", None)
  torch_vars.pop("vocab_size", None)
  for torch_key, v in torch_vars.items():
    if "num_batches_tracked" in torch_key:
      continue

    if "conv" in torch_key or "downsample.0.weight" in torch_key:
      v = v.transpose(2, 3, 1, 0)
    elif "weight" in torch_key and v.ndim == 2 and "embedding" not in torch_key:
      # Fully connected layers are transposed, embeddings are not
      v = v.T

    jax_key = torch_key.replace("visual.proj", "visual.proj.kernel")
    jax_key = jax_key.replace("text_projection", "text_projection.kernel")
    if "bn" in jax_key or "ln" in jax_key or "downsample.1" in jax_key:
      jax_key = jax_key.replace(".weight", ".scale")
    else:
      jax_key = jax_key.replace(".weight", ".kernel")
    if (jax_key.startswith("transformer") or
        jax_key.startswith("text_projection") or
        jax_key.startswith("ln_final") or
        jax_key.startswith("positional_embedding")):
      jax_key = "text." + jax_key

    jax_key = jax_key.replace(
        "token_embedding.kernel", "text.token_embedding.embedding")

    jax_key = jax_key.replace("attnpool.k_proj", "attnpool.attn.key")
    jax_key = jax_key.replace("attnpool.q_proj", "attnpool.attn.query")
    jax_key = jax_key.replace("attnpool.v_proj", "attnpool.attn.value")
    jax_key = jax_key.replace("attnpool.c_proj", "attnpool.attn.out")
    if "attnpool.attn.out" in jax_key:
      if jax_key.endswith("kernel"):
        v = v.reshape(-1, dim_head, v.shape[-1])
    elif "attnpool.attn" in jax_key:
      if jax_key.endswith("bias"):
        v = v.reshape(-1, dim_head)
      else:
        v = v.reshape(v.shape[0], -1, dim_head)

    if jax_key.endswith("running_mean"):
      jax_key = "batch_stats." + jax_key.replace(".running_mean", ".mean")
    elif jax_key.endswith("running_var"):
      jax_key = "batch_stats." + jax_key.replace(".running_var", ".var")
    else:
      jax_key = "params." + jax_key

    jax_key = jax_key.replace(".", "/")
    jax_key = jax_key.replace("resblocks/", "resblocks.")
    jax_key = jax_key.replace("resblocks/", "resblocks.")

    flax_vars[tuple(jax_key.split("/"))] = jnp.asarray(v, dtype=jnp.float32)

  # Transform the flattened param dict to the original nested structure.
  new_variables = flax.traverse_util.unflatten_dict(flax_vars)
  new_params = new_variables.pop("params")
  new_states = new_variables
  return new_params, new_states


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
  tree_def = jax.tree_util.tree_structure(tree)
  names_and_vals = _tree_flatten_with_names(tree)
  names, vals = zip(*names_and_vals)
  rest_vals = [list(zip(*_tree_flatten_with_names(t)))[1] for t in rest]
  vals = [f(*name_and_vals) for name_and_vals in zip(names, vals, *rest_vals)]
  return tree_def.unflatten(vals)


def _reinit(restored_params, init_params, to_reinit):
  """Reinitializes a subset of the parameters in the restored parameter tree."""
  f = lambda name, restored, init: init if name in to_reinit else restored
  return _tree_map_with_names(f, restored_params, init_params)


def restore_from_pretrained_params(init_params, loaded_params,
                                   reinit_params):
  """Initializes (some) model parameters based on pretrained parameters.

  Args:
    init_params: Tree of (possibly randomly) initialized parameters for the
      model. The structure will be kept, and a subset of the values will be
      replaced with values loaded from the pretrained checkpoint.
    loaded_params: Tree with pretrained weights.
    reinit_params: List of parameter names to reinitialize.

  Returns:
    A tree of parameters with the same structure as `init_params`, but loaded
    with pretrained weights in `loaded_params` and adapted accordingly.
  """
  if "opt" in loaded_params:
    loaded_params = loaded_params["opt"]["target"]

  restored_params = adapt_upstream_architecture(init_params, loaded_params)

  if reinit_params:
    restored_params = _reinit(restored_params, init_params, reinit_params)

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
      field `model_init` must be set.

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
      (save_checkpoint_path is not None and gfile.exists(save_checkpoint_path))
      or config.get("resume") is not None)
  reinitialize_model = config.get(
      "model_init") is not None and not resume_from_checkpoint

  if resume_from_checkpoint:
    logging.info("Resume training from checkpoint...")
    # Always prioritize loading from a checkpoint from the current training job.
    if save_checkpoint_path and gfile.exists(save_checkpoint_path):
      resume_checkpoint_path = save_checkpoint_path
    # Otherwise, we reload from a previous checkpoint provided by the config.
    else:
      resume_checkpoint_path = config.resume

    checkpoint_tree = {"opt": init_optimizer, "extra": checkpoint_extra}
    if init_fixed_model_states is not None:
      checkpoint_tree["states"] = init_fixed_model_states
    checkpoint = load_checkpoint(checkpoint_tree, resume_checkpoint_path)
    optimizer, checkpoint_extra = checkpoint["opt"], checkpoint["extra"]
    fixed_model_states = checkpoint.get("states", {})

  elif reinitialize_model:
    logging.info("Initialize model...")
    reinit_params = config.get("model_reinit_params", default_reinit_params)
    logging.info("Reinitializing these parameters: %s", reinit_params)

    if config.get("convert_pytorch", False):
      with gfile.GFile(config.model_init, "rb") as f:
        np_params = np.load(f, allow_pickle=True).tolist()

      loaded_params, loaded_states = _convert_vars(np_params)
      fixed_model_states = loaded_states
    else:
      loader = lambda path: load_checkpoint(tree=None, path=path)
      loaded_params = loader(config.model_init)
      fixed_model_states = loaded_params.get("states", {})

      # TODO(jjren) Extend restore_from_pretrained_params below so that the
      # states are only loaded if the model uses those states,
      loaded_params = restore_from_pretrained_params(
          init_params=init_params,
          loaded_params=loaded_params,
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
