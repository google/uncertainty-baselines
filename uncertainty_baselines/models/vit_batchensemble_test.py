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

"""Tests for vit_batchensemble.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp

import uncertainty_baselines as ub


class BatchEnsembleMlpBlockTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    if jax.device_count() != 8 or jax.local_device_count() != 8:
      self.skipTest("This test only runs on hosts with 8 local devices.")

  @parameterized.parameters(
      (4, [3, 20], 0.0, 10),
      (2, [5, 7, 3], -0.5, 10),
  )
  def test_params_shapes(self, ens_size, inputs_shape, random_sign_init,
                         mlp_dim):
    """Test the shapes of the parameters of a BE MLP block."""
    be_mlp_block = ub.models.vit_batchensemble.BatchEnsembleMlpBlock(
        mlp_dim=mlp_dim,
        ens_size=ens_size,
        random_sign_init=random_sign_init)
    inputs = jax.random.normal(
        jax.random.PRNGKey(0), inputs_shape, dtype=jnp.float32)
    params = be_mlp_block.init(
        jax.random.PRNGKey(0), inputs, deterministic=False)["params"]
    params_shape = jax.tree_map(lambda x: x.shape, params)
    expected_kernel_shape = (inputs_shape[-1], mlp_dim)
    expected_alpha_shape = (ens_size, inputs_shape[-1])
    expected_gamma_shape = (ens_size, mlp_dim)
    self.assertEqual(expected_kernel_shape,
                     params_shape["DenseBatchEnsemble_0"]["kernel"])
    self.assertEqual(expected_alpha_shape,
                     params_shape["DenseBatchEnsemble_0"]["fast_weight_alpha"])
    self.assertEqual(expected_gamma_shape,
                     params_shape["DenseBatchEnsemble_0"]["fast_weight_gamma"])

  @parameterized.parameters(
      (4, [3, 20], 0.0, 10),
      (2, [5, 7, 3], -0.5, 10),
  )
  def test_output_shapes(self, ens_size, inputs_shape, random_sign_init,
                         mlp_dim):
    """Test the shapes of the outputs of a BE MLP block."""
    be_mlp_block = ub.models.vit_batchensemble.BatchEnsembleMlpBlock(
        mlp_dim=mlp_dim,
        ens_size=ens_size,
        random_sign_init=random_sign_init)

    @functools.partial(jax.pmap, axis_name="batch")
    def apply(_):
      # Initialize and run TokenMoeBlock on device.
      inputs = jax.random.normal(
          jax.random.PRNGKey(0), inputs_shape, dtype=jnp.float32)
      return be_mlp_block.init_with_output(
          jax.random.PRNGKey(0), inputs, deterministic=False)[0]

    outputs = apply(jnp.arange(jax.local_device_count()))

    expected_outputs_shape = [
        jax.local_device_count(), inputs_shape[0]] + inputs_shape[1:]
    self.assertEqual(expected_outputs_shape, list(outputs.shape))


class BatchEnsembleEncoderTest(parameterized.TestCase):

  def test_params_shapes(self):
    """Tests that the parameters in the Encoder have the correct shape."""
    ens_size = 2
    encoder = ub.models.vit_batchensemble.BatchEnsembleEncoder(
        train=False,
        num_heads=4,
        num_layers=2,
        mlp_dim=128,
        ens_size=ens_size,
        random_sign_init=0.5)
    inputs = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 48))
    params = encoder.init(jax.random.PRNGKey(0), inputs)["params"]
    params_shapes = jax.tree_map(lambda x: x.shape, params)
    # First layer in the encoder is a regular MLPBlock.
    self.assertEqual(
        (48, 128), params_shapes["encoderblock_0"]["mlp"]["Dense_0"]["kernel"])
    self.assertEqual(
        (128, 48), params_shapes["encoderblock_0"]["mlp"]["Dense_1"]["kernel"])
    # Second layer is a BE block with DenseBatchEnsemble layers.
    self.assertEqual((48, 128), params_shapes["encoderblock_1"]["mlp"]
                     ["DenseBatchEnsemble_0"]["kernel"])
    self.assertEqual((128, 48), params_shapes["encoderblock_1"]["mlp"]
                     ["DenseBatchEnsemble_1"]["kernel"])

  @parameterized.parameters(
      (2, None, 4, 0.5),           # Only 1 BE layer, layer 1.
      (4, [0, 1], 2, 0.5),         # Two BE layers: layer 0 and layer 1.
  )
  def test_outputs_shapes(self, num_layers, be_layers, ens_size,
                          random_sign_init):
    encoder = ub.models.vit_batchensemble.BatchEnsembleEncoder(
        train=False,
        num_heads=4,
        num_layers=num_layers,
        be_layers=be_layers,
        ens_size=ens_size,
        random_sign_init=random_sign_init,
        mlp_dim=128)

    @functools.partial(jax.pmap, axis_name="batch")
    def apply(_):
      inputs = jax.random.normal(jax.random.PRNGKey(0), (4, 16, 48))
      return encoder.init_with_output(jax.random.PRNGKey(1), inputs)[0]

    outputs, _ = apply(jnp.arange(jax.local_device_count()))
    self.assertEqual((jax.local_device_count(), 4, 16, 48),
                     outputs.shape)


class PatchTransformerBETest(parameterized.TestCase):

  @parameterized.parameters(
      ((4, 4), None, 10, None, (32, 32), "token"),
      (None, (4, 4), 100, 64, (32, 64), "token"),
      ((4, 4), None, 10, None, (32, 32), "gap"),
      (None, (4, 4), 100, 64, (32, 64), "gap"),
      ((4, 4), None, 10, None, (32, 32), "map"),
      (None, (4, 4), 100, 64, (32, 64), "map"),
  )
  def test_params_and_outputs_shapes(self, patch_size, patch_grid, num_classes,
                                     representation_size,
                                     expected_pre_logits_params_shape,
                                     classifier):
    ens_size = 2
    model = ub.models.PatchTransformerBE(
        train=False,
        patch_size=patch_size,
        patch_grid=patch_grid,
        num_classes=num_classes,
        representation_size=representation_size,
        hidden_size=32,
        transformer=dict(
            num_heads=4,
            num_layers=2,
            mlp_dim=128,
            ens_size=ens_size,
            random_sign_init=0.5),
        classifier=classifier)

    @functools.partial(jax.pmap, axis_name="batch")
    def apply(_):
      inputs = jax.random.normal(jax.random.PRNGKey(0), (3, 16, 16, 3))
      return model.init_with_output(jax.random.PRNGKey(1), inputs)

    (outputs, _), params = apply(jnp.arange(jax.local_device_count()))
    self.assertEqual((jax.local_device_count(), 3, num_classes),
                     outputs.shape)

    if representation_size:
      self.assertEqual(
          (jax.local_device_count(), *expected_pre_logits_params_shape),
          params["params"]["pre_logits"]["kernel"].shape)

    params_shapes = jax.tree_map(lambda x: x.shape, params)
    be_params = params_shapes["params"]["BatchEnsembleTransformer"][
        "encoderblock_1"]
    self.assertEqual((jax.local_device_count(), 32, 128),
                     be_params["mlp"]["DenseBatchEnsemble_0"]["kernel"])
    self.assertEqual((jax.local_device_count(), 128, 32),
                     be_params["mlp"]["DenseBatchEnsemble_1"]["kernel"])
    self.assertEqual(
        (jax.local_device_count(), 2, 32),
        be_params["mlp"]["DenseBatchEnsemble_0"]["fast_weight_alpha"])
    self.assertEqual(
        (jax.local_device_count(), 2, 128),
        be_params["mlp"]["DenseBatchEnsemble_1"]["fast_weight_alpha"])


if __name__ == "__main__":
  absltest.main()
