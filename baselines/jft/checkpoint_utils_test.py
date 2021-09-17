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

"""Tests for the checkpointing utilities used in the ViT experiments."""

import os
import tempfile

from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import uncertainty_baselines as ub
import checkpoint_utils  # local file import


def _make_model(num_classes=21843, representation_size=2):
  config = ml_collections.ConfigDict()
  config.num_classes = num_classes

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 4
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 3
  config.model.transformer.num_heads = 2
  config.model.transformer.num_layers = 1
  config.model.classifier = "token"
  config.model.representation_size = representation_size

  model = ub.models.vision_transformer(
      num_classes=config.num_classes, **config.get("model", {}))
  return model, config


def _make_pytree(key):
  key1, key2, key3, key4, key5 = jax.random.split(key, num=5)
  tree = {
      "a": jax.random.normal(key1, (3, 2)),
      "b": {
          "c": jax.random.uniform(key2, (3, 2)),
          "d": jax.random.normal(key3, (2, 4)),
          "e": {
              "f": jax.random.uniform(key4, (1, 5))
          }
      }
  }
  # Create bfloat params to test saving/loading.
  bfloat_params = jax.random.normal(key5, (3, 2), dtype=jax.dtypes.bfloat16)
  tree = {"a": tree, "b": bfloat_params}
  return tree


class CheckpointUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_checkpointing(self):
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    tree = _make_pytree(subkey)
    checkpoint_utils.save_checkpoint(tree, checkpoint_path)

    key, subkey = jax.random.split(key)
    new_tree = _make_pytree(subkey)

    leaves = jax.tree_util.tree_leaves(tree)
    new_leaves = jax.tree_util.tree_leaves(new_tree)
    for arr, new_arr in zip(leaves, new_leaves):
      self.assertNotAllClose(arr, new_arr)

    restored_tree = checkpoint_utils.load_checkpoint(new_tree, checkpoint_path)
    restored_leaves = jax.tree_util.tree_leaves(restored_tree)
    for arr, restored_arr in zip(leaves, restored_leaves):
      self.assertAllClose(arr, restored_arr)

  def test_checkpointing_model(self):
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))
    key = jax.random.PRNGKey(42)

    model, _ = _make_model()
    input_shape = (2, 224, 224, 3)
    dummy_input = jnp.zeros(input_shape, jnp.float32)

    key, subkey = jax.random.split(key)
    params = model.init(subkey, dummy_input, train=False)["params"]
    checkpoint_utils.save_checkpoint(params, checkpoint_path)

    key, subkey = jax.random.split(key)
    new_params = model.init(subkey, dummy_input, train=False)["params"]
    restored_params = checkpoint_utils.load_checkpoint(new_params,
                                                       checkpoint_path)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)
    leaves = jax.tree_util.tree_leaves(params)
    for arr, restored_arr in zip(leaves, restored_leaves):
      self.assertAllClose(arr, restored_arr)

    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, input_shape, jnp.float32)
    _, out = model.apply({"params": params}, inputs, train=False)
    _, new_out = model.apply({"params": new_params}, inputs, train=False)
    _, restored_out = model.apply({"params": restored_params},
                                  inputs,
                                  train=False)
    self.assertNotAllClose(out["pre_logits"], new_out["pre_logits"])
    self.assertAllClose(out["pre_logits"], restored_out["pre_logits"])



if __name__ == "__main__":
  tf.test.main()
