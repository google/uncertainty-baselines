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

"""Tests for the segmenterGP ViT model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub


class SegVitGPTest(parameterized.TestCase):

  @parameterized.parameters(
      ('gap', 2, 16),
      ('token', 2, 16),
  )
  def test_segmenter_gp(self, classifier, num_classes, hidden_size):
    # VisionTransformer.
    img_h = 224
    img_w = 224
    patch_size = 4
    config = ml_collections.ConfigDict()

    config.num_classes = num_classes

    config.patches = ml_collections.ConfigDict()
    config.patches.size = [patch_size, patch_size]

    config.backbone_configs = ml_collections.ConfigDict()
    config.backbone_configs.type = 'vit'
    config.backbone_configs.hidden_size = hidden_size
    config.backbone_configs.attention_dropout_rate = 0.
    config.backbone_configs.dropout_rate = 0.
    config.backbone_configs.mlp_dim = 2
    config.backbone_configs.num_heads = 1
    config.backbone_configs.num_layers = 1
    config.backbone_configs.classifier = classifier

    config.decoder_configs = ml_collections.ConfigDict()
    config.decoder_configs.type = 'gp'

    # GP layer params
    config.decoder_configs.gp_layer = ml_collections.ConfigDict()
    config.decoder_configs.gp_layer.covmat_kwargs = ml_collections.ConfigDict()
    config.decoder_configs.gp_layer.covmat_kwargs.ridge_penalty = -1.
    # Disable momentum in order to use exact covariance update for finetuning.
    # Disable to allow exact cov update.
    config.decoder_configs.gp_layer.covmat_kwargs.momentum = 0.99
    config.decoder_configs.mean_field_factor = 1.

    num_examples = 3
    inputs = jnp.ones([num_examples, img_h, img_w, 3], jnp.float32)
    model = ub.models.SegVitGP(**config)

    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, train=False)

    logits, outputs = model.apply(variables, inputs, train=False)

    self.assertEqual(logits.shape, (num_examples, img_h, img_w, num_classes))
    self.assertEqual(
        set(outputs.keys()),
        set(('stem', 'transformed', 'logits', 'logits_gp', 'covmat_gp')))
    self.assertEqual(
        outputs['stem'].shape,
        (num_examples, img_h // patch_size, img_w // patch_size, hidden_size))

    num_tokens = img_h // patch_size * img_w // patch_size
    num_tokens = num_tokens + 1 if classifier == 'token' else num_tokens

    self.assertEqual(outputs['transformed'].shape,
                     (num_examples, num_tokens, hidden_size))
    self.assertEqual(
        outputs['logits_gp'].shape,
        (num_examples * img_h // patch_size * img_w // patch_size, num_classes))
    self.assertEqual(
        outputs['covmat_gp'].shape,
        (num_examples * img_h // patch_size * img_w // patch_size,))
    self.assertEqual(outputs['logits'].shape,
                     (num_examples, img_h, img_w, num_classes))


if __name__ == '__main__':
  absltest.main()
