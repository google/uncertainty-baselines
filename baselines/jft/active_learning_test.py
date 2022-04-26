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

"""Tests for the for the Active Learning with a pre-trained model script."""
# TODO(joost,andreas): Refactor active_learning.py and use this test for smaller
# components including acquisition functions and other utility functions.
# pylint: disable=pointless-string-statement

import os
import pathlib
import tempfile

from absl import flags
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import active_learning  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import test_utils  # local file import from baselines.jft

flags.adopt_module_key_flags(active_learning)
FLAGS = flags.FLAGS


class ActiveLearningTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    self.data_dir = os.path.join(baseline_root_dir, 'testing_data')

  @parameterized.parameters(
      # NOTE: For max_training_set_size == initial_training_set_size, all
      # methods should yield the same ids since no aquisition step takes place.
      # Uniform.
      ('deterministic', 'uniform', 4, 4, [0, 1, 3, 7]),
      ('deterministic', 'uniform', 4, 0, [17, 24, 34, 43]),
      ('batchensemble', 'uniform', 4, 1, [0, 1, 3, 7]),
      # Entropy.
      ('deterministic', 'entropy', 4, 4, [0, 1, 3, 7]),
      ('batchensemble', 'entropy', 4, 1, [0, 1, 2, 6]),
      # Margin.
      ('deterministic', 'margin', 4, 4, [0, 1, 3, 7]),
      ('batchensemble', 'margin', 4, 1, [0, 1, 6, 8]),
      # MSP.
      ('deterministic', 'msp', 4, 4, [0, 1, 3, 7]),
      ('batchensemble', 'msp', 4, 1, [0, 1, 6, 9]),
      # BALD
      ('batchensemble', 'bald', 4, 4, [0, 1, 3, 7]),
      ('batchensemble', 'bald', 4, 1, [0, 1, 2, 4]),
      # Density
      ('deterministic', 'density', 4, 4, [0, 1, 3, 7]),
      ('batchensemble', 'density', 4, 1, [0, 1, 3, 8]),
  )
  def test_active_learning_script(self, model_type, acquisition_method,
                                  max_training_set_size,
                                  initial_training_set_size, gt_ids):
    data_dir = self.data_dir

    # Create a dummy checkpoint
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    dataset_name = 'cifar10'
    config = test_utils.get_config(
        dataset_name=dataset_name, classifier='token', representation_size=2)

    config.model_type = model_type

    if model_type == 'batchensemble':
      config.model.transformer.num_layers = 2
      config.model.transformer.ens_size = 2
      config.model.transformer.random_sign_init = 0.5
      config.model.transformer.be_layers = (1,)
      config.fast_weight_lr_multiplier = 1.0
      config.weight_decay = 0.1
      model = ub.models.vision_transformer_be(
          num_classes=config.num_classes, **config.model)
      config.total_steps = 5
    else:
      model = ub.models.vision_transformer(
          num_classes=config.num_classes, **config.get('model', {}))

    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)

    dummy_input = jnp.zeros((config.batch_size, 224, 224, 3), jnp.float32)
    params = flax.core.unfreeze(model.init(rng_init, dummy_input,
                                           train=False))['params']

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    checkpoint_utils.save_checkpoint(params, checkpoint_path)

    config.model_init = checkpoint_path
    config.test_split = 'test'

    # Active learning settings
    config.acquisition_method = acquisition_method
    config.max_training_set_size = max_training_set_size
    config.initial_training_set_size = initial_training_set_size
    config.acquisition_batch_size = 1
    config.early_stopping_patience = 6  # not tested yet

    with tfds.testing.mock_data(num_examples=10, data_dir=data_dir):
      ids, _ = active_learning.main(config, output_dir)

    # Get the warmup batch
    self.assertEqual(ids, set(gt_ids))


if __name__ == '__main__':
  tf.test.main()
