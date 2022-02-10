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
import ml_collections
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import batchensemble  # local file import from baselines.jft
import batchensemble_utils  # local file import from baselines.jft

flags.adopt_module_key_flags(batchensemble)
FLAGS = flags.FLAGS


def get_config(classifier, representation_size):
  """Config."""
  config = ml_collections.ConfigDict()
  config.seed = 0

  config.batch_size = 3
  config.total_steps = 3

  num_examples = config.batch_size * config.total_steps

  # TODO(dusenberrymw): JFT + mocking is broken.
  # config.dataset = 'jft/entity:1.0.0'
  # config.val_split = 'test[:49511]'  # aka tiny_test/test[:5%] in task_adapt
  # config.train_split = 'train'  # task_adapt used train+validation so +64167
  # config.num_classes = 18291

  # NOTE: TFDS mocking currently ignores split slices.
  config.dataset = 'imagenet21k'
  config.val_split = f'full[:{num_examples}]'
  config.train_split = f'full[{num_examples}:{num_examples*2}]'
  config.num_classes = 21843

  config.prefetch_to_device = 1
  config.shuffle_buffer_size = 20
  config.val_cache = False

  config.log_training_steps = config.total_steps
  config.log_eval_steps = config.total_steps
  config.checkpoint_steps = config.total_steps
  config.keep_checkpoint_steps = config.total_steps
  config.backup_checkpoint_steps = None
  config.write_checkpoint_every_n_steps = config.total_steps
  config.log_training_every_n_steps = config.total_steps
  config.checkpoint_write_timeout_secs = 100

  pp_common = '|value_range(-1, 1)'
  pp_common += f'|onehot({config.num_classes})'
  pp_common += '|keep(["image", "labels"])'
  # TODO(dusenberrymw): Mocking doesn't seem to encode into jpeg format.
  # config.pp_train = 'decode_jpeg_and_inception_crop(224)|flip_lr' + pp_common
  config.pp_train = 'decode|inception_crop(224)|flip_lr' + pp_common
  config.pp_eval = 'decode|resize_small(256)|central_crop(224)' + pp_common

  config.init_head_bias = 1e-3

  # Model section
  config.model = ml_collections.ConfigDict()
  config.model.patches = ml_collections.ConfigDict()
  config.model.patches.size = [16, 16]
  config.model.hidden_size = 4
  config.model.transformer = ml_collections.ConfigDict()
  config.model.transformer.attention_dropout_rate = 0.
  config.model.transformer.dropout_rate = 0.
  config.model.transformer.mlp_dim = 3
  config.model.transformer.num_heads = 2
  config.model.transformer.num_layers = 2
  config.model.classifier = classifier
  config.model.representation_size = representation_size

  # BatchEnsemble parameters
  config.model.transformer.ens_size = 2
  config.model.transformer.random_sign_init = 0.5
  config.model.transformer.be_layers = (1,)
  config.fast_weight_lr_multiplier = 1.0

  # Optimizer section
  config.optim_name = 'Adam'
  config.optim = ml_collections.ConfigDict()
  config.optim.weight_decay = 0.1
  config.optim.beta1 = 0.9
  config.optim.beta2 = 0.999
  config.weight_decay = .1

  config.lr = ml_collections.ConfigDict()
  config.lr.base = 0.1
  config.lr.warmup_steps = 1
  config.lr.decay_type = 'linear'
  config.lr.linear_end = 1e-3

  return config


class BatchEnsembleTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Go two directories up to the root of the UB directory.
    ub_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(ub_root_dir) + '/.tfds/metadata'
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
    self.assertAllEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.softmax(ensemble_logits), axis=0)
    self.assertAllClose(jax.nn.softmax(actual_logits), expected_probs)

    actual_logits = batchensemble_utils.log_average_sigmoid_probs(
        ensemble_logits)
    self.assertAllEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.sigmoid(ensemble_logits), axis=0)
    self.assertAllClose(jax.nn.sigmoid(actual_logits), expected_probs)

  @parameterized.parameters(
      ('token', 2, 13350.85, 12320.728299),
      ('token', None, 12024.248, 14304.993924),
      ('gap', 2, 13562.656, 13176.437066),
      ('gap', None, 13262.055, 12964.980469),
  )
  @flagsaver.flagsaver
  def test_batchensemble_script(self, classifier, representation_size,
                                correct_train_loss, correct_val_loss):
    # Set flags.
    FLAGS.xm_runlocal = True
    FLAGS.config = get_config(
        classifier=classifier, representation_size=representation_size)
    FLAGS.output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    FLAGS.config.dataset_dir = self.data_dir
    num_examples = FLAGS.config.batch_size * FLAGS.config.total_steps

    # Check for any errors.
    with tfds.testing.mock_data(
        num_examples=num_examples, data_dir=self.data_dir):
      train_loss, val_loss, _ = batchensemble.main(None)

    # Check for reproducibility.
    logging.info('(train_loss, val_loss) = %s, %s', train_loss, val_loss['val'])
    self.assertAllClose(train_loss, correct_train_loss)
    self.assertAllClose(val_loss['val'], correct_val_loss)

  @parameterized.parameters(
      ('token', 2),
      ('token', None),
      ('gap', 2),
      ('gap', None),
  )
  @flagsaver.flagsaver
  def test_load_model(self, classifier, representation_size):
    FLAGS.xm_runlocal = True
    FLAGS.config = get_config(
        classifier=classifier, representation_size=representation_size)
    FLAGS.output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    FLAGS.config.dataset_dir = self.data_dir
    FLAGS.config.total_steps = 2
    num_examples = FLAGS.config.batch_size * FLAGS.config.total_steps

    with tfds.testing.mock_data(
        num_examples=num_examples, data_dir=self.data_dir):
      _, val_loss, _ = batchensemble.main(None)
      checkpoint_path = os.path.join(FLAGS.output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))

      # Set different output directory so that the logic doesn't think we are
      # resuming from a previous checkpoint.
      FLAGS.output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      FLAGS.config.model_init = checkpoint_path
      # Reload model from checkpoint.
      # Currently, we don't have a standalone evaluation function, so we check
      # that the loaded model has the same performance as the saved model by
      # running training with a learning rate of 0 to obtain the train and eval
      # metrics.
      # TODO(zmariet, dusenberrymw): write standalone eval function.
      FLAGS.config.lr.base = 0.0
      FLAGS.config.lr.linear_end = 0.0
      FLAGS.config.lr.warmup_steps = 0
      FLAGS.config.model_reinit_params = []

      _, loaded_val_loss, _ = batchensemble.main(None)

    # We can't compare training losses, since `batchensemble.main()` reports the
    # loss *before* applying the last SGD update: the reported training loss is
    # different from the loss of the checkpointed model.
    self.assertEqual(val_loss['val'], loaded_val_loss['val'])


if __name__ == '__main__':
  tf.test.main()
