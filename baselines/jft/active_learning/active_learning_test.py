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
# pylint: disable=pointless-string-statement
"""import os.path import pathlib import tempfile

from absl import flags
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from .. import checkpoint_utils  # local file import
from .. import test_utils  # local file import
import active_learning  # local file import
flags.adopt_module_key_flags(active_learning)
FLAGS = flags.FLAGS


class ActiveLearningTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()

    baseline_root_dir = pathlib.Path(__file__).parents[1]
    self.data_dir = os.path.join(baseline_root_dir, 'testing_data')

  @parameterized.parameters(
      ('uniform', [840763837, 1167338319]),
      # NOTE: ideally margin and entropy don't have matching ids for test
      ('margin', [809552352, 758271112]),
      ('entropy', [809552352, 758271112]),
      ('density', [1546174544, 834394044]),
  )
  def test_active_learning_script(self, acquisition_method, gt_ids):

    data_dir = self.data_dir

    # Create a dummy checkpoint
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    config = test_utils.get_config(
        dataset_name='cifar10', classifier='token', representation_size=2)

    model = ub.models.vision_transformer(
        num_classes=config.num_classes, **config.get('model', {}))

    rng = jax.random.PRNGKey(42)
    rng, rng_init = jax.random.split(rng)

    dummy_input = jnp.zeros((config.batch_size, 224, 224, 3), jnp.float32)
    params = flax.core.unfreeze(model.init(rng_init, dummy_input,
                                           train=False))['params']

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    checkpoint_utils.save_checkpoint(params, checkpoint_path)

    # Active Learn on CIFAR-10
    config.dataset = 'cifar10'
    config.val_split = 'train[98%:]'
    config.train_split = 'train[:98%]'
    config.batch_size = 8
    config.max_labels = 4
    config.acquisition_batch_size = 2
    config.total_steps = 6
    config.dataset_dir = data_dir
    config.model_init = checkpoint_path
    config.acquisition_method = acquisition_method

    with tfds.testing.mock_data(num_examples=50, data_dir=data_dir):
      ids, _ = active_learning.main(config)

    # Get the warmup batch
    gt_ids = [934744266, 986104245] + gt_ids
    self.assertEqual(ids, set(gt_ids))


if __name__ == '__main__':
  tf.test.main()
"""
