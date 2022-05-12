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

"""Tests for the segmenterHet ViT model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub


class SegVitHetTest(parameterized.TestCase):

  @parameterized.parameters(
      ('gap', 3, 16),
      ('token', 3, 16),
  )
  def test_segmenter_het(self, classifier, num_classes, hidden_size):
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
    config.decoder_configs.type = 'het'

    # Het layer params
    # temp: wide sweep [0.15, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
    config.decoder_configs.temperature = 1.0
    # low rank approx ~ FxK where K is the classes.
    config.decoder_configs.param_efficient = False
    # F in low rank approx of KxK matrix changes as:
    # imagenet~15, jft~50, cifar~6, cityscapes~5
    config.decoder_configs.num_factors = 1
    # use as much as we can afford, ideally > 10.
    config.decoder_configs.mc_samples = 1000
    config.decoder_configs.return_locs = False
    # turn on if we want to run an approx on KHW x KHW instead of KxK.
    config.decoder_configs.share_samples_across_batch = False

    num_examples = 3
    inputs = jnp.ones([num_examples, img_h, img_w, 3], jnp.float32)
    model = ub.models.SegVitHet(**config)

    # we need to pass random seeds for init and apply
    # standard_norm_noise_samples only used when num_factors > 0
    seed = config.get('seed', 0)
    keys = ['diag_noise_samples', 'standard_norm_noise_samples', 'params']
    rngs = dict(zip(keys, jax.random.split(jax.random.PRNGKey(seed), 3)))

    variables = model.init(rngs, inputs, train=False)

    # For the test code we use the same rngs but all these should be updated.
    logits, outputs = model.apply(variables, inputs, train=False, rngs=rngs)

    self.assertEqual(logits.shape, (num_examples, img_h, img_w, num_classes))
    self.assertEqual(
        set(outputs.keys()),
        set(('stem', 'transformed', 'logits', 'logits_het')))
    self.assertEqual(
        outputs['stem'].shape,
        (num_examples, img_h // patch_size, img_w // patch_size, hidden_size))

    num_tokens = img_h // patch_size * img_w // patch_size
    num_tokens = num_tokens + 1 if classifier == 'token' else num_tokens

    self.assertEqual(outputs['transformed'].shape,
                     (num_examples, num_tokens, hidden_size))
    self.assertEqual(
        outputs['logits_het'].shape,
        (num_examples * img_h // patch_size * img_w // patch_size, num_classes))
    self.assertEqual(outputs['logits'].shape,
                     (num_examples, img_h, img_w, num_classes))


if __name__ == '__main__':
  absltest.main()
