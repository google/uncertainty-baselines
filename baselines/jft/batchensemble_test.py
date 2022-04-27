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

"""Tests for the batchensemble ViT on JFT-300M model script."""
import os
import pathlib
import tempfile

from absl import flags
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import batchensemble  # local file import from baselines.jft
import batchensemble_utils  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import test_utils  # local file import from baselines.jft

flags.adopt_module_key_flags(batchensemble)
FLAGS = flags.FLAGS


def get_config(dataset_name, classifier, representation_size):
  """Config."""
  config = test_utils.get_config(
      dataset_name=dataset_name,
      classifier=classifier,
      representation_size=representation_size)

  # BatchEnsemble parameters
  config.model.transformer.num_layers = 2
  config.model.transformer.ens_size = 2
  config.model.transformer.random_sign_init = 0.5
  config.model.transformer.be_layers = (1,)
  config.fast_weight_lr_multiplier = 1.0
  config.weight_decay = 0.1

  return config


class BatchEnsembleTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    data_dir = os.path.join(baseline_root_dir, 'testing_data')
    logging.info('data_dir contents: %s', os.listdir(data_dir))
    self.data_dir = data_dir

  @parameterized.parameters(1, 3, 5)
  def test_log_average_probs(self, ensemble_size):
    batch_size, num_classes = 16, 3
    logits_shape = (ensemble_size, batch_size, num_classes)
    np.random.seed(42)
    ensemble_logits = jnp.asarray(np.random.normal(size=logits_shape))

    actual_logits = batchensemble_utils.log_average_softmax_probs(
        ensemble_logits)
    self.assertEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.softmax(ensemble_logits), axis=0)
    np.testing.assert_allclose(jax.nn.softmax(actual_logits), expected_probs,
                               rtol=1e-06, atol=1e-06)

    actual_logits = batchensemble_utils.log_average_sigmoid_probs(
        ensemble_logits)
    self.assertEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.sigmoid(ensemble_logits), axis=0)
    np.testing.assert_allclose(jax.nn.sigmoid(actual_logits), expected_probs,
                               rtol=1e-06, atol=1e-06)

  @parameterized.parameters(
      ('imagenet2012', 'token', 2, 611.4291, 580.2454969618055, False),
      ('imagenet2012', 'token', 2, 611.4291, 580.2454969618055, True),
      ('imagenet2012', 'token', None, 551.31506, 657.0848659939236, False),
      ('imagenet2012', 'gap', 2, 617.892, 595.8379313151041, False),
      ('imagenet2012', 'gap', None, 591.90234, 591.7898356119791, False),
  )
  @flagsaver.flagsaver
  def test_batchensemble_script(self, dataset_name, classifier,
                                representation_size, correct_train_loss,
                                correct_val_loss, simulate_failure):
    data_dir = self.data_dir
    config = get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir
    num_examples = config.batch_size * config.total_steps

    if not simulate_failure:
      # Check for any errors.
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        # TODO(dusenberrymw): Test the fewshot results once deterministic.
        train_loss, val_loss, _ = batchensemble.main(config, output_dir)
    else:
      # Check for the ability to restart from a previous checkpoint (after
      # failure, etc.).
      # NOTE: Use this flag to simulate failing at a certain step.
      config.testing_failure_step = config.total_steps - 1
      config.checkpoint_steps = config.testing_failure_step
      config.keep_checkpoint_steps = config.checkpoint_steps
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        batchensemble.main(config, output_dir)

      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))
      checkpoint = checkpoint_utils.load_checkpoint(None, checkpoint_path)
      self.assertEqual(
          int(checkpoint['opt']['state']['step']), config.testing_failure_step)

      # This should resume from the failed step.
      del config.testing_failure_step
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, _ = batchensemble.main(config, output_dir)

    # Check for reproducibility.
    logging.info('(train_loss, val_loss) = %s, %s', train_loss, val_loss['val'])
    np.testing.assert_allclose(train_loss, correct_train_loss,
                               rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(val_loss['val'], correct_val_loss,
                               rtol=1e-06, atol=1e-06)

  @parameterized.parameters(
      ('imagenet2012', 'token', 2),
      ('imagenet2012', 'token', None),
      ('imagenet2012', 'gap', 2),
      ('imagenet2012', 'gap', None),
  )
  @flagsaver.flagsaver
  def test_load_model(self, dataset_name, classifier, representation_size):
    config = get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = self.data_dir
    config.total_steps = 2
    num_examples = config.batch_size * config.total_steps

    with tfds.testing.mock_data(
        num_examples=num_examples, data_dir=self.data_dir):
      _, val_loss, _ = batchensemble.main(config, output_dir)
      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))

      # Set different output directory so that the logic doesn't think we are
      # resuming from a previous checkpoint.
      output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      config.model_init = checkpoint_path
      # Reload model from checkpoint.
      # Currently, we don't have a standalone evaluation function, so we check
      # that the loaded model has the same performance as the saved model by
      # running training with a learning rate of 0 to obtain the train and eval
      # metrics.
      # TODO(zmariet, dusenberrymw): write standalone eval function.
      config.lr.base = 0.0
      config.lr.linear_end = 0.0
      config.lr.warmup_steps = 0
      config.model_reinit_params = []

      _, loaded_val_loss, _ = batchensemble.main(config, output_dir)

    # We can't compare training losses, since `batchensemble.main()` reports the
    # loss *before* applying the last SGD update: the reported training loss is
    # different from the loss of the checkpointed model.
    self.assertEqual(val_loss['val'], loaded_val_loss['val'])


if __name__ == '__main__':
  tf.test.main()
