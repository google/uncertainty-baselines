# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Tests for the ViT model with a TRAM extension."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub

PI_INPUT_DIM = 5
HIDDEN_SIZE = 2
NUM_CLASSES = 1000

# The PI tower adds two MLPs with one hidden-layer each, together with a dense
# layer to precict the NUM_CLASSES classes.
# By default, their mlp_dim and out_dim are set to HIDDEN_SIZE.
# The config below leads to (when representation_size is assumed None)
#    (PI_INPUT_DIM + 1) * pp_mlp_dim + (pp_mlp_dim + 1) * pp_out_dim
#  + (HIDDEN_SIZE + pp_out_dim + 1) * jn_mlp_dim
#  + (jn_mlp_dim + 1) * (HIDDEN_SIZE + pp_out_dim)
#  + (2 * HIDDEN_SIZE + pp_out_dim + 1) * NUM_CLASSES
#  = 12131
# extra parameters. A similar computation can be made when representation_size
# is set and/or when pi_tower is not specified, with default values of
# pp_mlp_dim=pp_out_dim=jn_mlp_dim=HIDDEN_SIZE.
PI_TOWER_CONFIG = {
    'pp_mlp_dim': 3,
    'pp_out_dim': 7,
    'jn_mlp_dim': 4,
}


class VitTramTest(parameterized.TestCase):

  @parameterized.parameters(
      ('token', 3, 20131, PI_TOWER_CONFIG),
      ('token', None, 4982+12131, PI_TOWER_CONFIG),
      ('gap', 3, 15032, None),
      ('gap', None, 4978+12131, PI_TOWER_CONFIG),
  )
  def test_vision_transformer_tram(self, classifier, representation_size,
                                   expected_param_count, pi_tower=None):
    # TODO(dusenberrymw): Clean this up once config dict is cleaned up in
    # VisionTransformer.
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict()
    config.patches.size = [16, 16]
    config.hidden_size = HIDDEN_SIZE
    config.transformer = ml_collections.ConfigDict()
    config.transformer.attention_dropout_rate = 0.
    config.transformer.dropout_rate = 0.
    config.transformer.mlp_dim = 2
    config.transformer.num_heads = 1
    config.transformer.num_layers = 1
    config.classifier = classifier
    config.representation_size = representation_size
    if pi_tower is not None:
      config.pi_tower = pi_tower

    num_examples = 2
    num_classes = NUM_CLASSES
    inputs = jnp.ones([num_examples, 224, 224, 3], jnp.float32)
    pi_inputs = jnp.ones([num_examples, PI_INPUT_DIM], jnp.float32)
    model = ub.models.vision_transformer_tram(num_classes=num_classes, **config)
    key = jax.random.PRNGKey(0)
    variables = model.init(key, inputs, pi_inputs, train=False)

    param_count = sum(p.size for p in jax.tree.flatten(variables)[0])
    self.assertEqual(param_count, expected_param_count)

    logits, outputs = model.apply(variables, inputs, pi_inputs, train=False)
    self.assertEqual(logits.shape, (num_examples, num_classes))
    self.assertEqual(
        set(outputs.keys()),
        set(('stem', 'transformed', 'head_input', 'pre_logits', 'logits',
             'pi_logits')))
    self.assertEqual(outputs['stem'].shape,
                     (num_examples, 14, 14, config.hidden_size))
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
    self.assertEqual(outputs['pi_logits'].shape, (num_examples, num_classes))


if __name__ == '__main__':
  absltest.main()
