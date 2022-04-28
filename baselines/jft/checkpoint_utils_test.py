# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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
import numpy as np
import tensorflow as tf
import uncertainty_baselines as ub
import checkpoint_utils  # local file import from baselines.jft


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
  config = _get_config(
      num_classes=num_classes, representation_size=representation_size)
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


def _init_model(key, model, input_shape=(2, 224, 224, 3),
                rand_normal_head_kernel=False):
  dummy_input = jnp.zeros(input_shape, jnp.float32)
  params = model.init(key, dummy_input, train=False)["params"]
  if rand_normal_head_kernel:
    # By default, ViT has its head initialized to zeroes.
    # To test non-trivial predictions, we sometimes need to set the params of
    # the head to non-zero values (here from a normal distribution).
    _, rand_normal_key = jax.random.split(key, 2)
    shape = params["head"]["kernel"].shape
    params = flax.core.unfreeze(params)
    params["head"]["kernel"] = jax.random.normal(rand_normal_key, shape)
    params = flax.core.freeze(params)
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
  params["head"]["kernel"] = value * jnp.ones_like(params["head"]["kernel"])
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
      self.assertFalse(jnp.allclose(arr, new_arr), msg=(arr, new_arr))

    restored_tree = checkpoint_utils.load_checkpoint(new_tree, checkpoint_path)
    restored_leaves = jax.tree_util.tree_leaves(restored_tree)
    for arr, restored_arr in zip(leaves, restored_leaves):
      self.assertIsInstance(restored_arr, np.ndarray)
      self.assertTrue(jnp.allclose(arr, restored_arr), msg=(arr, restored_arr))

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
      self.assertIsInstance(restored_arr, np.ndarray)
      np.testing.assert_allclose(arr, restored_arr)

    key, subkey = jax.random.split(key)
    inputs = jax.random.normal(subkey, input_shape, jnp.float32)
    _, out = model.apply({"params": params}, inputs, train=False)
    _, new_out = model.apply({"params": new_params}, inputs, train=False)
    _, restored_out = model.apply({"params": restored_params},
                                  inputs,
                                  train=False)
    np.testing.assert_raises(
        AssertionError,
        np.testing.assert_allclose,
        out["pre_logits"],
        new_out["pre_logits"])
    np.testing.assert_allclose(out["pre_logits"], restored_out["pre_logits"])


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
      (["head/kernel"], True),
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
        default_reinit_params=("head/kernel", "head/bias"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)
    np.testing.assert_allclose(checkpoint_data.train_loop_rngs,
                               flax_utils.replicate(load_ckpt_key))

    expected_head_val = init_head_val if expect_new_head else ckpt_head_val
    actual_head = checkpoint_data.optimizer.target["head"]["kernel"]
    np.testing.assert_allclose(
        actual_head,
        jnp.ones_like(actual_head) * expected_head_val)

  @parameterized.parameters(
      (["pre_logits/kernel"], True, "smaller"),
      (["pre_logits/kernel"], True, "same"),
      (["pre_logits/kernel"], True, "larger"),
      ([], False, "smaller"),
  )
  def test_checkpointing_reinitialize_different_models(
      self, reinit_params, expect_new_pre_logits_kernel,
      change_downstream_pytree_size_compared_with_upstream):
    key = jax.random.PRNGKey(42)
    num_steps = 10
    train_time = 3.14

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    checkpoint_path = os.path.join(output_dir, "checkpoint.npz")
    self.assertFalse(os.path.exists(checkpoint_path))

    # Initialize a upstream deterministic model.
    model_det, _ = _make_deterministic_model()
    key, init_key, save_ckpt_key = jax.random.split(key, 3)
    params_det = _init_model(init_key, model_det)

    # Create optimzier and save checkpoint.
    opt = flax.optim.Adam().create(params_det)
    opt = opt.replace(state=opt.state.replace(step=num_steps))
    checkpoint_utils.checkpoint_trained_model(
        checkpoint_utils.CheckpointData(
            optimizer=opt,
            fixed_model_states=None,
            accumulated_train_time=train_time,
            train_loop_rngs=flax_utils.replicate(save_ckpt_key)),
        path=checkpoint_path)

    # Initialize a downstream SNGP model.
    model_sngp, config_sngp = _make_sngp_model()
    key, init_key, load_ckpt_key = jax.random.split(key, 3)
    init_params = _init_model(init_key, model_sngp)
    if change_downstream_pytree_size_compared_with_upstream == "smaller":
      # We delete the head/output_layer/bias to make SNGP parameter pytree
      # smaller than the upstream model.
      init_params_flat = checkpoint_utils._flatten_jax_params_dict(init_params)
      del init_params_flat["head/output_layer/bias"]
      init_params = checkpoint_utils._unflatten_jax_params_dict(
          init_params_flat)
    elif change_downstream_pytree_size_compared_with_upstream == "larger":
      # We add the head/additional_kernel to make SNGP parameter pytree
      # larger than the upstream model.
      init_params_flat = checkpoint_utils._flatten_jax_params_dict(init_params)
      init_params_flat["head/additional_kernel"] = jnp.ones((2, 2))
      init_params = checkpoint_utils._unflatten_jax_params_dict(
          init_params_flat)

    # We make the SNGP's pre_logits layer kernel a dummpy jnp.ones matrix,
    # which makes it easy to test whether the reinitialization
    # works as expected.
    init_params = flax.core.unfreeze(init_params)
    init_params["pre_logits"]["kernel"] = jnp.ones_like(
        init_params["pre_logits"]["kernel"])
    init_params = flax.core.freeze(init_params)

    init_opt = flax.optim.Adam().create(init_params)
    config_sngp.model_init = checkpoint_path
    config_sngp.model_reinit_params = reinit_params

    # Load checkpoint from disk.
    checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
        load_ckpt_key,
        init_optimizer=init_opt,
        init_params=init_params,
        init_fixed_model_states=0.,
        save_checkpoint_path=None,
        config=config_sngp,
        default_reinit_params=("head/kernel", "head/bias"))

    self.assertEqual(checkpoint_data.optimizer.state.step, 0)
    self.assertEqual(checkpoint_data.accumulated_train_time, 0.)
    self.assertAllClose(checkpoint_data.train_loop_rngs,
                        flax_utils.replicate(load_ckpt_key))

    actual_kernel = checkpoint_data.optimizer.target["pre_logits"]["kernel"]
    if expect_new_pre_logits_kernel:
      self.assertAllClose(actual_kernel, jnp.ones_like(actual_kernel))
    else:
      self.assertAllClose(actual_kernel, params_det["pre_logits"]["kernel"])

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

    restored_leaves = jax.tree_util.tree_leaves(
        checkpoint_data.optimizer.target)
    leaves = jax.tree_util.tree_leaves(opt.target)
    for arr, restored_arr in zip(leaves, restored_leaves):
      np.testing.assert_allclose(arr, restored_arr)

    np.testing.assert_allclose(checkpoint_data.train_loop_rngs,
                               flax_utils.replicate(ckpt_key))

  def test_adapt_upstream_architecture_toy(self):
    init_params = {"a": {"b": 1, "c": 2}, "d": 3}
    loaded_params = {"a": {"b": 42, "e": 4}}

    expected_params = {"a": {"b": 42, "c": 2}, "d": 3}
    actual_params = checkpoint_utils.adapt_upstream_architecture(
        init_params, loaded_params)
    self.assertDictEqual(expected_params, actual_params)


def _get_deterministic_params():
  key = jax.random.PRNGKey(42)
  model, _ = _make_deterministic_model()
  return _init_model(key, model)


def _get_sngp_params():
  key = jax.random.PRNGKey(42)
  config = _get_config()
  vit_kwargs = config.get("model")
  gp_layer_kwargs = {"covmat_kwargs": {"momentum": 0.999}}

  model = ub.models.vision_transformer_gp(
      num_classes=config.num_classes,
      use_gp_layer=True,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)

  return _init_model(key, model)


def _get_batchensemble_params():
  key = jax.random.PRNGKey(42)
  config = _get_config()
  config.model.transformer.ens_size = 3
  config.model.transformer.be_layers = (0,)
  config.model.transformer.random_sign_init = .5
  model = ub.models.vision_transformer_be(
      num_classes=config.num_classes, **config.model)
  return _init_model(key, model)


def _get_heteroscedastic_params():
  key = jax.random.PRNGKey(42)

  config = _get_config()
  config.model.multiclass = False
  config.model.temperature = 0.4
  config.model.mc_samples = 1000
  config.model.num_factors = 50
  config.model.param_efficient = True
  model = ub.models.vision_transformer_het(
      num_classes=config.num_classes, **config.get("model", {}))

  key, diag_key, noise_key = jax.random.split(key, 3)
  rng = {
      "params": key,
      "diag_noise_samples": diag_key,
      "standard_norm_noise_samples": noise_key
  }
  return _init_model(rng, model)


def _get_hetsngp_params():
  key = jax.random.PRNGKey(42)
  config = _get_config()
  config.het = ml_collections.ConfigDict()
  config.het.multiclass = False
  config.het.temperature = 1.5
  config.het.mc_samples = 1000
  config.het.num_factors = 50
  config.het.param_efficient = True

  gp_layer_kwargs = dict(covmat_kwargs={"momentum": .999})
  vit_kwargs = config.get("model")
  het_kwargs = config.get("het")

  model = ub.models.vision_transformer_hetgp(
      num_classes=config.num_classes,
      use_gp_layer=True,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs,
      multiclass=het_kwargs.multiclass,
      temperature=het_kwargs.temperature,
      mc_samples=het_kwargs.mc_samples,
      num_factors=het_kwargs.num_factors,
      param_efficient=het_kwargs.param_efficient)

  key, diag_key, noise_key = jax.random.split(key, 3)
  rng = {
      "params": key,
      "diag_noise_samples": diag_key,
      "standard_norm_noise_samples": noise_key
  }
  return _init_model(rng, model)


_MLP_PREFIX = "Transformer/encoderblock_0/MlpBlock_3"
_MODEL_TO_PARAMS_AND_KEYS = {
    "deterministic":
        dict(
            params_fn=_get_deterministic_params,
            extra_keys=[
                "head/kernel",
                "head/bias",
                f"{_MLP_PREFIX}/Dense_0/bias",
                f"{_MLP_PREFIX}/Dense_1/bias",
            ]),
    "sngp":
        dict(
            params_fn=_get_sngp_params,
            extra_keys=[
                "head/output_layer/kernel",
                "head/output_layer/bias",
                f"{_MLP_PREFIX}/Dense_0/bias",
                f"{_MLP_PREFIX}/Dense_1/bias",
            ]),
    "heteroscedastic":
        dict(
            params_fn=_get_heteroscedastic_params,
            extra_keys=[
                "multilabel_head/diag_layer/kernel",
                "multilabel_head/diag_layer/bias",
                "multilabel_head/loc_layer/kernel",
                "multilabel_head/loc_layer/bias",
                "multilabel_head/scale_layer_homoscedastic/kernel",
                "multilabel_head/scale_layer_homoscedastic/bias",
                "multilabel_head/scale_layer_heteroscedastic/kernel",
                "multilabel_head/scale_layer_heteroscedastic/bias",
                f"{_MLP_PREFIX}/Dense_0/bias",
                f"{_MLP_PREFIX}/Dense_1/bias",
            ]),
    "hetsngp":
        dict(
            params_fn=_get_hetsngp_params,
            extra_keys=[
                "head/loc_layer/output_layer/kernel",
                "head/loc_layer/output_layer/bias",
                "head/scale_layer_homoscedastic/kernel",
                "head/scale_layer_homoscedastic/bias",
                "head/scale_layer_heteroscedastic/bias",
                "head/scale_layer_heteroscedastic/kernel",
                "head/diag_layer/kernel",
                "head/diag_layer/bias",
                f"{_MLP_PREFIX}/Dense_0/bias",
                f"{_MLP_PREFIX}/Dense_1/bias",
            ]),
    "batchensemble":
        dict(
            params_fn=_get_batchensemble_params,
            extra_keys=[
                "pre_logits/fast_weight_gamma",
                "pre_logits/fast_weight_alpha",
                "batchensemble_head/kernel",
                "batchensemble_head/bias",
                "batchensemble_head/fast_weight_alpha",
                "batchensemble_head/fast_weight_gamma",
                f"{_MLP_PREFIX}/Dense_1/fast_weight_alpha",
                f"{_MLP_PREFIX}/Dense_1/fast_weight_gamma",
                f"{_MLP_PREFIX}/Dense_1/bias",
                f"{_MLP_PREFIX}/Dense_0/fast_weight_gamma",
                f"{_MLP_PREFIX}/Dense_0/fast_weight_alpha",
                f"{_MLP_PREFIX}/Dense_0/bias",
            ])
}


class AdaptUpstreamModelTest(parameterized.TestCase):

  @parameterized.product(
      upstream_model=[
          "deterministic", "sngp", "heteroscedastic", "hetsngp", "batchensemble"
      ],
      downstream_model=[
          "deterministic", "sngp", "heteroscedastic", "hetsngp", "batchensemble"
      ],
  )
  def test_adapt_upstream_architecture(self, upstream_model, downstream_model):

    upstream_keys = _MODEL_TO_PARAMS_AND_KEYS[upstream_model]["extra_keys"]
    upstream_params = _MODEL_TO_PARAMS_AND_KEYS[upstream_model]["params_fn"]()
    flattened_upstream_params = checkpoint_utils._flatten_jax_params_dict(
        upstream_params)

    downstream_keys = _MODEL_TO_PARAMS_AND_KEYS[downstream_model]["extra_keys"]
    downstream_params = _MODEL_TO_PARAMS_AND_KEYS[downstream_model][
        "params_fn"]()
    flattened_downstream_params = checkpoint_utils._flatten_jax_params_dict(
        downstream_params)

    adapted_params = checkpoint_utils.adapt_upstream_architecture(
        init_params=downstream_params, loaded_params=upstream_params)
    flattened_adapted_params = checkpoint_utils._flatten_jax_params_dict(
        adapted_params)

    shared_keys = set(flattened_upstream_params.keys()).intersection(
        set(flattened_downstream_params.keys()))
    actual_keys = set(flattened_adapted_params.keys())

    expected_new_keys = set(downstream_keys) - set(upstream_keys)
    expected_removed_keys = set(upstream_keys) - set(downstream_keys)

    self.assertSetEqual(actual_keys, shared_keys.union(expected_new_keys))

    for key in expected_removed_keys:
      self.assertNotIn(key, actual_keys)

    for key in shared_keys:
      np.testing.assert_array_equal(flattened_adapted_params[key],
                                    flattened_upstream_params[key])

    for key in expected_new_keys:
      np.testing.assert_array_equal(flattened_adapted_params[key],
                                    flattened_downstream_params[key])


if __name__ == "__main__":
  tf.test.main()
