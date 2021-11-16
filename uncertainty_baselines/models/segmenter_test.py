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

"""Tests for the segmenter ViT model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub


class SegVitTest(parameterized.TestCase):

  @parameterized.parameters(
      (2, 16, 224, 224),
  )
  def test_segmenter_transformer(self, num_classes, hidden_size, img_h, img_w):
    # VisionTransformer.
    config = ml_collections.ConfigDict()

    config.num_classes = num_classes

    config.patches = ml_collections.ConfigDict()
    config.patches.size = [4, 4]

    config.backbone_configs = ml_collections.ConfigDict()
    config.backbone_configs.type = 'vit'
    config.backbone_configs.hidden_size = hidden_size
    config.backbone_configs.attention_dropout_rate = 0.
    config.backbone_configs.dropout_rate = 0.
    config.backbone_configs.mlp_dim = 2
    config.backbone_configs.num_heads = 1
    config.backbone_configs.num_layers = 1
    # TODO(kellybuchanan): include 'token' test
    config.backbone_configs.classifier = 'gap'

    config.decoder_configs = ml_collections.ConfigDict()
    config.decoder_configs.type = 'linear'

    num_examples = 2
    inputs = jnp.ones([num_examples, img_h, img_w, 3], jnp.float32)
    model = ub.models.segmenter_transformer(**config)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, train=False)

    logits, outputs = model.apply(variables, inputs, train=False)
    
    self.assertEqual(logits.shape, (num_examples, img_h, img_w, num_classes))
    self.assertEqual(
        set(outputs.keys()),
        set(('stem', 'transformed', 'logits')))


if __name__ == '__main__':
  absltest.main()
