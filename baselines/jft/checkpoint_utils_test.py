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
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import ml_collections
import tensorflow as tf
import uncertainty_baselines as ub
import checkpoint_utils  # local file import


def _get_config(num_classes=21843, representation_size=2):
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
  return config


def _make_deterministic_model(num_classes=21843, representation_size=2):
  config = _get_config(num_classes=num_classes,
                       representation_size=representation_size)
  model = ub.models.vision_transformer(
      num_classes=config.num_classes, **config.get("model", {}))
  return model, config


def _make_sngp_model(num_classes=21843, representation_size=2):
  config = _get_config(num_classes=num_classes,
                       representation_size=representation_size)
  vit_kwargs = config.get("model")
  gp_layer_kwargs = {"covmat_kwargs": {"momentum": 0.999}}

  model = ub.models.vision_transformer_gp(
      num_classes=config.num_classes,
      use_gp_layer=True,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)
  return model, config


def _init_model(key, model, input_shape=(2, 224, 224, 3)):
  dummy_input = jnp.zeros(input_shape, jnp.float32)
  params = model.init(key, dummy_input, train=False)["params"]
  return params


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


def _reset_head_kernel(params, value):
  params = flax.core.unfreeze(params)
  params["head"]["kernel"] = value * jnp.ones_like(
      params["head"]["kernel"])
  return flax.core.freeze(params)


class CheckpointUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def _save_temp_checkpoint(self, checkpoint_dict):
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))
    checkpoint_utils.save_checkpoint(checkpoint_dict, checkpoint_path)
    return checkpoint_path

  @parameterized.named_parameters(
      dict(testcase_name="_no_parent_key_empty", input_dict={}, sep="/",
           parent_key="", expected_dict={}),
      dict(testcase_name="_parent_key_empty", input_dict={}, sep="/",
           parent_key="a", expected_dict={"a": {}}),
      dict(testcase_name="_no_parent_key",
           input_dict={"a": {"b": [2, 3], "c": 1}, "d": {}},
           sep="/",
           parent_key="",
           expected_dict={"a/b": [2, 3], "a/c": 1, "d": {}}),
      dict(testcase_name="_with_parent_key",
           input_dict={"x": {"y": 2, "z": {"t": 3}}, "u": 4},
           sep=":",
           parent_key="abc",
           expected_dict={"abc:x:y": 2, "abc:x:z:t": 3, "abc:u": 4}),
  )
  def test_flatten_jax_params_dict(
      self, input_dict, sep, parent_key, expected_dict):
    actual_dict = checkpoint_utils._flatten_jax_params_dict(
        input_dict, parent_key=parent_key, sep=sep)
    self.assertDictEqual(actual_dict, expected_dict)

  @parameterized.named_parameters(
      dict(testcase_name="_colon_separator", input_dict={"a/b:c": 1, "d": 3},
           sep=":", expected_dict={"a/b": {"c": 1}, "d": 3}),
      dict(testcase_name="_slash_separator", input_dict={"a/b/c": {}, "d": 3},
           sep="/", expected_dict={"a": {"b": {"c": {}}}, "d": 3}),
      dict(testcase_name="_empty_dict", input_dict={}, sep="",
           expected_dict={}),
  )
  def test_unflatten_jax_params_dict(self, input_dict, sep, expected_dict):
    actual_dict = checkpoint_utils._unflatten_jax_params_dict(
        input_dict, sep=sep)
    self.assertDictEqual(actual_dict, expected_dict)

  def test_checkpointing(self):
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    tree = _make_pytree(subkey)
    checkpoint_path = self._save_temp_checkpoint(tree)

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
    key = jax.random.PRNGKey(42)

    model, _ = _make_deterministic_model()
    input_shape = (2, 224, 224, 3)
    key, subkey = jax.random.split(key)
    params = _init_model(subkey, model, input_shape=input_shape)
    checkpoint_path = self._save_temp_checkpoint(params)

    key, subkey = jax.random.split(key)
    new_params = _init_model(subkey, model, input_shape=input_shape)
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


  @parameterized.named_parameters(
      dict(testcase_name="_from_argument", path_in_config=False),
      dict(testcase_name="_from_config", path_in_config=True),
  )
  def test_checkpointing_resume(self, path_in_config):
    key = jax.random.PRNGKey(42)
    num_steps = 10
    train_time = 3.14
    head_kernel_value = 2.71
    states = 6.62

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))

    # Initialize model.
    model, config = _make_deterministic_model()
    key, init_key, save_ckpt_key = jax.random.split(key, 3)
    params = _reset_head_kernel(_init_model(init_key, model), head_kernel_value)

    # Create optimizer and save checkpoint
    opt = flax.optim.Adam().create(params)
    opt = opt.replace(state=opt.state.replace(step=num_steps))
    checkpoint_utils.checkpoint_trained_model(
        checkpoint_utils.CheckpointData(
            optimizer=opt,
            fixed_model_states=states,
            accumulated_train_time=train_time,
            train_loop_rngs=flax_utils.replicate(save_ckpt_key)),
        path=checkpoint_path)

    # When resuming a checkpoint, we should not load from `config.model_init`.
    config.model_init = "/tmp/fake_ckpt.npz"
    config.resume = "/tmp/fake_ckpt.npz"
    if path_in_config:
      config.resume = checkpoint_path
      checkpoint_path = None

    key, init_key, load_ckpt_key = jax.random.split(key, 3)
    init_params = _init_model(init_key, model)
    init_opt = flax.optim.Adam().create(init_params)
    checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
        load_ckpt_key,
        save_checkpoint_path=checkpoint_path,
        init_optimizer=init_opt,
        init_params=init_params,
        init_fixed_model_states=0.,
        default_reinit_params=["head/kernel"],
        config=config)

    self.assertEqual(checkpoint_data.optimizer.state.step, num_steps)
    self.assertEqual(checkpoint_data.fixed_model_states, states)
    self.assertEqual(checkpoint_data.accumulated_train_time, train_time)
    self.assertAllClose(checkpoint_data.train_loop_rngs,
                        flax_utils.replicate(save_ckpt_key))

    self.assertAllClose(checkpoint_data.optimizer.target, params)
    self.assertNotAllClose(checkpoint_data.optimizer.target, init_params)

  @parameterized.parameters(
      (["head/kernel"], True),
      ([], False),
  )
  def test_checkpointing_reinitialize(
      self, reinit_params, expect_new_head):
    key = jax.random.PRNGKey(42)
    init_head_val = 1.
    ckpt_head_val = 0.
    num_steps = 10
    train_time = 3.14

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))

    # Initialize model and set head kernel to non-default values.
    model, config = _make_deterministic_model()
    key, init_key, save_ckpt_key = jax.random.split(key, 3)
    params = _reset_head_kernel(_init_model(init_key, model), ckpt_head_val)

    # Create optimzier and save checkpoint.
    opt = flax.optim.Adam().create(params)
    opt = opt.replace(state=opt.state.replace(step=num_steps))
    checkpoint_utils.checkpoint_trained_model(
        checkpoint_utils.CheckpointData(
            optimizer=opt,
            fixed_model_states=None,
            accumulated_train_time=train_time,
            train_loop_rngs=flax_utils.replicate(save_ckpt_key)),
        path=checkpoint_path)

    config.model_init = checkpoint_path
    config.model_reinit_params = reinit_params

    # Load checkpoint from disk.
    key, init_key, load_ckpt_key = jax.random.split(key, 3)
    init_params = _reset_head_kernel(
        _init_model(init_key, model), init_head_val)
    init_opt = flax.optim.Adam().create(init_params)
    checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
        load_ckpt_key,
        init_optimizer=init_opt,
        init_params=init_params,
        init_fixed_model_states=0.,
        save_checkpoint_path=None,
        config=config,
        default_reinit_params=("head/kernel", "head/bias"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)
    self.assertAllClose(checkpoint_data.train_loop_rngs,
                        flax_utils.replicate(load_ckpt_key))

    expected_head_val = init_head_val if expect_new_head else ckpt_head_val
    actual_head = checkpoint_data.optimizer.target["head"]["kernel"]
    self.assertAllClose(
        actual_head, jnp.ones_like(actual_head) * expected_head_val)

  def test_checkpointing_no_loading(self):
    key = jax.random.PRNGKey(42)

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))

    # Initialize model.
    model, config = _make_deterministic_model()
    init_key, ckpt_key = jax.random.split(key)
    params = _init_model(init_key, model)
    opt = flax.optim.Adam().create(params)

    checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
        ckpt_key,
        init_optimizer=opt,
        init_params=params,
        init_fixed_model_states=None,
        save_checkpoint_path=None,
        config=config,
        default_reinit_params=("head/kernel", "head/bias"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)

    self.assertAllClose(checkpoint_data.optimizer.target, opt.target)
    self.assertAllClose(checkpoint_data.train_loop_rngs,
                        flax_utils.replicate(ckpt_key))

  def test_adapt_upstream_architecture_toy(self):
    init_params = {"a": {"b": 1, "c": 2}, "d": 3}
    loaded_params = {"a": {"b": 42, "e": 4}}

    expected_params = {"a": {"b": 42, "c": 2}, "d": 3}
    actual_params = checkpoint_utils.adapt_upstream_architecture(
        init_params, loaded_params)
    self.assertDictEqual(expected_params, actual_params)

  def test_adapt_upstream_architecture_sgnp(self):
    key = jax.random.PRNGKey(42)

    upstream_model, _ = _make_deterministic_model()
    key, init_key = jax.random.split(key)
    upstream_params = _init_model(init_key, upstream_model)
    flattened_upstream_params = checkpoint_utils._flatten_jax_params_dict(
        upstream_params)

    downstream_model, _ = _make_sngp_model()
    key, init_key = jax.random.split(key)
    downstream_params = _init_model(init_key, downstream_model)
    flattened_downstream_params = checkpoint_utils._flatten_jax_params_dict(
        downstream_params)

    adapted_params = checkpoint_utils.adapt_upstream_architecture(
        init_params=downstream_params, loaded_params=upstream_params)
    flattened_adapted_params = checkpoint_utils._flatten_jax_params_dict(
        adapted_params)

    shared_keys = set(flattened_upstream_params.keys()).intersection(
        set(flattened_downstream_params.keys()))

    expected_new_keys = {"head/output_layer/kernel", "head/output_layer/bias"}
    expected_missing_keys = {"head/kernel", "head/bias"}
    actual_keys = set(flattened_adapted_params.keys())

    self.assertSetEqual(actual_keys, shared_keys.union(expected_new_keys))

    for key in expected_missing_keys:
      self.assertNotIn(key, actual_keys)

    for key in shared_keys:
      self.assertAllEqual(flattened_adapted_params[key],
                          flattened_upstream_params[key])

    for key in expected_new_keys:
      self.assertAllEqual(flattened_adapted_params[key],
                          flattened_downstream_params[key])


if __name__ == "__main__":
  tf.test.main()
