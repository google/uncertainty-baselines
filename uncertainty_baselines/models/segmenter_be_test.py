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

"""Tests for the segmenter ViT model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub


class SegVitBETest(parameterized.TestCase):

  @parameterized.parameters(
      (2, 16, 224, 224, 'vit', 'linear', 1),
      (2, 16, 224, 224, 'vit_be', 'linear', 3),
      (2, 16, 224, 224, 'vit_be', 'linear_be', 3),
  )
  def test_segmenter_be_transformer(self, num_classes, hidden_size, img_h,
                                    img_w, encoder_type, decoder_type,
                                    ens_size):
    # VisionTransformer.
    config = ml_collections.ConfigDict()

    config.num_classes = num_classes

    config.patches = ml_collections.ConfigDict()
    config.patches.size = [4, 4]

    config.backbone_configs = ml_collections.ConfigDict()
    config.backbone_configs.type = encoder_type
    config.backbone_configs.hidden_size = hidden_size
    config.backbone_configs.attention_dropout_rate = 0.
    config.backbone_configs.dropout_rate = 0.
    config.backbone_configs.mlp_dim = 2
    config.backbone_configs.num_heads = 1
    config.backbone_configs.num_layers = 1
    config.backbone_configs.classifier = 'gap'

    config.decoder_configs = ml_collections.ConfigDict()
    config.decoder_configs.type = decoder_type

    # BE params
    config.backbone_configs.ens_size = ens_size
    config.backbone_configs.random_sign_init = -0.5
    config.backbone_configs.be_layers = (0,)

    num_examples = 2
    inputs = jnp.ones([num_examples, img_h, img_w, 3], jnp.float32)
    model = ub.models.SegVitBE(**config)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, train=False)

    logits, outputs = model.apply(variables, inputs, train=False)

    self.assertEqual(logits.shape,
                     (num_examples * ens_size, img_h, img_w, num_classes))

    self.assertEqual(
        set(outputs.keys()), set(('stem', 'transformed', 'logits')))


if __name__ == '__main__':
  absltest.main()
