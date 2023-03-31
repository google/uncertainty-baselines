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

"""Tests for the checkpointing utilities used in the ViT experiments."""

import os
import tempfile

from absl.testing import parameterized
import flax
import flax.jax_utils as flax_utils
import flax.training.checkpoints
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import checkpoint_utils  # local file import from experimental.multimodal


def _get_config():
  config = ml_collections.ConfigDict()
  config.model = ml_collections.ConfigDict()
  config.model.embed_dim = 4
  config.model.vocab_size = 32
  config.model.vision_num_layers = 1
  config.model.vision_features = 128
  config.model.vision_patch_size = 32
  config.model.text_features = 32
  config.model.text_num_heads = 2
  config.model.text_num_layers = 1
  return config


def _make_deterministic_model():
  config = _get_config()
  model = ub.models.clip(**config.model)
  return model, config


def _init_model(key, model, image_input_shape=(2, 224, 224, 3),
                text_input_shape=(2, 512)):
  dummy_image = jnp.zeros(image_input_shape, jnp.float32)
  dummy_text = jnp.zeros(text_input_shape, jnp.int32)
  params = model.init(key, dummy_image, dummy_text)["params"]
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
  params["text"]["text_projection"]["kernel"] = value * jnp.ones_like(
      params["text"]["text_projection"]["kernel"])
  params["visual"]["proj"]["kernel"] = value * jnp.ones_like(
      params["visual"]["proj"]["kernel"])
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
  def test_flatten_jax_params_dict(self, input_dict, sep, parent_key,
                                   expected_dict):
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

  @parameterized.parameters(True, False)
  def test_checkpointing(self, read_in_parallel):
    key = jax.random.PRNGKey(42)

    key, subkey = jax.random.split(key)
    tree = _make_pytree(subkey)
    checkpoint_path = self._save_temp_checkpoint(tree)

    key, subkey = jax.random.split(key)
    new_tree = _make_pytree(subkey)

    leaves = jax.tree_util.tree_leaves(tree)
    new_leaves = jax.tree_util.tree_leaves(new_tree)
    for arr, new_arr in zip(leaves, new_leaves):
      self.assertFalse(jnp.allclose(arr, new_arr), msg=(arr, new_arr))

    restored_tree = checkpoint_utils.load_checkpoint(
        new_tree, checkpoint_path, read_in_parallel=read_in_parallel)
    restored_leaves = jax.tree_util.tree_leaves(restored_tree)
    for arr, restored_arr in zip(leaves, restored_leaves):
      self.assertIsInstance(restored_arr, np.ndarray)
      self.assertTrue(jnp.allclose(arr, restored_arr), msg=(arr, restored_arr))

  @parameterized.parameters(True, False)
  def test_checkpointing_model(self, read_in_parallel):
    key = jax.random.PRNGKey(42)

    model, _ = _make_deterministic_model()
    image_input_shape = (2, 224, 224, 3)
    text_input_shape = (2, 512)
    key, subkey = jax.random.split(key)
    params = _init_model(
        subkey,
        model,
        image_input_shape=image_input_shape,
        text_input_shape=text_input_shape)
    checkpoint_path = self._save_temp_checkpoint(params)

    key, subkey = jax.random.split(key)
    new_params = _init_model(
        subkey,
        model,
        image_input_shape=image_input_shape,
        text_input_shape=text_input_shape)
    restored_params = checkpoint_utils.load_checkpoint(
        new_params, checkpoint_path, read_in_parallel=read_in_parallel)
    restored_leaves = jax.tree_util.tree_leaves(restored_params)
    leaves = jax.tree_util.tree_leaves(params)
    for arr, restored_arr in zip(leaves, restored_leaves):
      self.assertIsInstance(restored_arr, np.ndarray)
      np.testing.assert_allclose(arr, restored_arr)

    key, subkey1, subkey2 = jax.random.split(key, 3)
    image_inputs = jax.random.normal(subkey1, image_input_shape, jnp.float32)
    text_inputs = jax.random.randint(
        subkey2, text_input_shape, 1, 10, dtype=jnp.int32)
    out_image, out_text = model.apply({"params": params}, image_inputs,
                                      text_inputs)
    new_out_image, new_out_text = model.apply({"params": new_params},
                                              image_inputs, text_inputs)
    restored_out_image, restored_out_text = model.apply(
        {"params": restored_params}, image_inputs, text_inputs)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        out_image,
        new_out_image)
    np.testing.assert_allclose(out_image, restored_out_image)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        out_text,
        new_out_text)
    np.testing.assert_allclose(out_text, restored_out_text)

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
    np.testing.assert_allclose(checkpoint_data.train_loop_rngs,
                               flax_utils.replicate(save_ckpt_key))

    def assert_dict_allclose(dict1, dict2):
      leaves1 = jax.tree_util.tree_leaves(dict1)
      leaves2 = jax.tree_util.tree_leaves(dict2)
      for arr1, arr2 in zip(leaves1, leaves2):
        self.assertIsInstance(arr2, np.ndarray)
        np.testing.assert_allclose(arr1, arr2)

    assert_dict_allclose(params, checkpoint_data.optimizer.target)
    np.testing.assert_raises(
        AssertionError, assert_dict_allclose, params, init_params)

  @parameterized.parameters(
      (["visual/proj/kernel", "text/text_projection/kernel"], True),
      ([], False),
  )
  def test_checkpointing_reinitialize_same_models(
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
        default_reinit_params=("visual/proj/kernel",
                               "text/text_projection/kernel"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)
    np.testing.assert_allclose(checkpoint_data.train_loop_rngs,
                               flax_utils.replicate(load_ckpt_key))

    expected_head_val = init_head_val if expect_new_head else ckpt_head_val
    actual_visual_head = checkpoint_data.optimizer.target["visual"]["proj"][
        "kernel"]
    np.testing.assert_allclose(
        actual_visual_head,
        jnp.ones_like(actual_visual_head) * expected_head_val)
    actual_text_head = checkpoint_data.optimizer.target["text"][
        "text_projection"]["kernel"]
    np.testing.assert_allclose(
        actual_text_head,
        jnp.ones_like(actual_text_head) * expected_head_val)

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
        default_reinit_params=("visual/proj/kernel",
                               "text/text_projection/kernel"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)

    restored_leaves = jax.tree_util.tree_leaves(
        checkpoint_data.optimizer.target)
    leaves = jax.tree_util.tree_leaves(opt.target)
    for arr, restored_arr in zip(leaves, restored_leaves):
      np.testing.assert_allclose(arr, restored_arr)

    np.testing.assert_allclose(checkpoint_data.train_loop_rngs,
                               flax_utils.replicate(ckpt_key))


if __name__ == "__main__":
  tf.test.main()
