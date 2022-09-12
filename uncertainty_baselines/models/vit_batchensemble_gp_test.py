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

"""Tests for the ViT-BatchEnsemble-GP model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import uncertainty_baselines as ub


class VisionTransformerBEGPTest(parameterized.TestCase):

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
    batch_size = 3
    # Gaussian process kwargs.
    hidden_features = 1024
    gp_layer_kwargs = dict(hidden_features=hidden_features)

    model = ub.models.vision_transformer_be_gp(
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
        classifier=classifier,
        gp_layer_kwargs=gp_layer_kwargs)

    inputs = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 16, 16, 3))
    variables = model.init(jax.random.PRNGKey(0), inputs)
    logits, extra = model.apply(variables, inputs)
    self.assertEqual((ens_size * batch_size, num_classes), logits.shape)
    self.assertEqual(extra["covmat"].shape, (ens_size * batch_size,))
    if representation_size:
      self.assertEqual(expected_pre_logits_params_shape,
                       variables["params"]["pre_logits"]["kernel"].shape)


if __name__ == "__main__":
  absltest.main()
