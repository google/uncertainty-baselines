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

"""Tests for the ViT-GP model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub


class VitTest(parameterized.TestCase):

  @parameterized.parameters(
      ('token', 3, 2079663),
      ('token', None, 2078630),
      ('gap', 3, 2079659),
      ('gap', None, 2078626),
  )
  def test_vision_transformer(self, classifier, representation_size,
                              expected_param_count):
    # TODO(dusenberrymw): Clean this up once config dict is cleaned up in
    # VisionTransformer.
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict()
    config.patches.size = [16, 16]
    config.hidden_size = 2
    config.transformer = ml_collections.ConfigDict()
    config.transformer.attention_dropout_rate = 0.
    config.transformer.dropout_rate = 0.
    config.transformer.mlp_dim = 2
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.classifier = classifier
    config.representation_size = representation_size

    # Gaussian process kwargs.
    hidden_features = 1024
    gp_layer_kwargs = dict(hidden_features=hidden_features)

    # Build model.
    num_examples = 2
    num_classes = 1000
    inputs = jnp.ones([num_examples, 224, 224, 3], jnp.float32)
    model = ub.models.vision_transformer_gp(
        num_classes=num_classes,
        use_gp_layer=True,
        vit_kwargs=config,
        gp_layer_kwargs=gp_layer_kwargs)

    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, train=False)

    param_count = sum(p.size for p in jax.tree_flatten(variables)[0])
    self.assertEqual(param_count, expected_param_count)

    logits, outputs = model.apply(variables, inputs, train=False)
    self.assertEqual(logits.shape, (num_examples, num_classes))
    self.assertEqual(
        set(outputs.keys()),
        set(('stem', 'transformed', 'head_input', 'pre_logits', 'logits',
             'covmat')))
    self.assertEqual(outputs['stem'].shape,
                     (num_examples, 14, 14, config.hidden_size))
    self.assertEqual(outputs['covmat'].shape, (num_examples,))

    if config.classifier == 'token':
      self.assertEqual(outputs['transformed'].shape,
                       (num_examples, 197, config.hidden_size))
    else:  # 'gap'
      self.assertEqual(outputs['transformed'].shape,
                       (num_examples, 196, config.hidden_size))
    self.assertEqual(outputs['head_input'].shape,
                     (num_examples, config.hidden_size))
    if config.representation_size is not None:
      self.assertEqual(outputs['pre_logits'].shape,
                       (num_examples, config.representation_size))
    else:
      self.assertEqual(outputs['pre_logits'].shape,
                       (num_examples, config.hidden_size))
    self.assertEqual(outputs['logits'].shape, (num_examples, num_classes))


if __name__ == '__main__':
  absltest.main()
