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

"""Tests for vit_batchensemble.py."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub

import flax



class VitTest(parameterized.TestCase):

  @parameterized.parameters(
      ('gap', 3, 5987),
  )
  def test_vision_transformer(self, classifier, representation_size,
                              expected_param_count):
    # TODO(dusenberrymw): Clean this up once config dict is cleaned up in
    # VisionTransformer.
    def getList(parent, dict):
      for key, value in dict.items():
        var_name = '{}/{}'.format(parent, key)
        if isinstance(value, jax.numpy.ndarray):
          print('{}, {}'.format(var_name, value.shape))
        else:
          getList(var_name, value)
      return dict.keys()

    DEBUG=1 #also visualize params for vit model
    config = ml_collections.ConfigDict()
    # Model parameters.
    config.model = ml_collections.ConfigDict()
    config.model.patches = ml_collections.ConfigDict()
    config.model.patches.size = [16, 16]
    config.model.hidden_size = 768
    config.model.representation_size = 768
    config.model.classifier = 'token'
    config.model.transformer = ml_collections.ConfigDict()
    config.model.transformer.num_layers = 12
    config.model.transformer.dropout_rate = 0.0
    config.model.transformer.mlp_dim = 3072
    config.model.transformer.num_heads = 12
    config.model.transformer.attention_dropout_rate = 0.0

    num_examples = 2
    num_classes = 1000
    inputs = jnp.ones([num_examples, 224, 224, 3], jnp.float32)

    if DEBUG ==1:
      model = ub.models.vision_transformer(num_classes=num_classes, **config.model)

      key = jax.random.PRNGKey(0)
      variables = model.init(key, inputs, train=False)

      param_count = sum(p.size for p in jax.tree_flatten(variables)[0])
      print(param_count)
      getList('variables', variables)

      logits, outputs = model.apply(variables, inputs, train=False)
      self.assertEqual(logits.shape, (num_examples, num_classes))
      self.assertEqual(
          set(outputs.keys()),
          set(('stem', 'transformed', 'head_input', 'pre_logits', 'logits')))

    # BatchEnsemble parameters.
    config.model.transformer.be_layers = (9, 11)
    config.model.transformer.ens_size = 3
    config.model.transformer.random_sign_init = 0.5
    config.fast_weight_lr_multiplier = 1.0

    model = ub.models.PatchTransformerBE(num_classes=num_classes, **config.model)

    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, train=False)

    param_count = sum(p.size for p in jax.tree_flatten(variables)[0])
    print(param_count)
    getList('variables', variables)

    logits, outputs = model.apply(variables, inputs, train=False)
    self.assertEqual(logits.shape, (num_examples * config.model.transformer.ens_size, num_classes))

    self.assertEqual(
      set(outputs.keys()), set(('pre_logits',)))



if __name__ == "__main__":
  absltest.main()
