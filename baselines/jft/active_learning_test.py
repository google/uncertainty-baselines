"""Tests for the for the Active Learning with a pre-trained model script."""
import os.path
import pathlib
import tempfile

import flax
import jax
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from absl import flags

import active_learning
import checkpoint_utils
import test_utils

flags.adopt_module_key_flags(active_learning)
FLAGS = flags.FLAGS


class ActiveLearningTest(tf.test.TestCase):
  def setUp(self):
    super().setUp()

    baseline_root_dir = pathlib.Path(__file__).parents[1]
    self.data_dir = os.path.join(baseline_root_dir, 'testing_data')

  def test_active_learning_script(self):
    data_dir = self.data_dir

    # Create a dummy checkpoint
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    config = test_utils.get_config(
        dataset_name="cifar10",
        classifier="token",
        representation_size=2)

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
    config.max_labels = 3
    config.acquisition_batch_size = 2
    config.dataset_dir = data_dir
    config.model_init = checkpoint_path

    with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
      ids, test_accs = active_learning.main(config, output_dir)

    self.assertAllClose(test_accs, [0.125, 0.125])
    self.assertAllEqual(ids,  [1751673441, 543581031, 1701210465, 1701209956])

if __name__ == '__main__':
  tf.test.main()
