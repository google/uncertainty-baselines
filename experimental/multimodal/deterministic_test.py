# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Tests for the deterministic CLIP model script."""
import os.path
import pathlib
import tempfile

from absl import flags
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
import jax
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import checkpoint_utils  # local file import from experimental.multimodal
import deterministic  # local file import from experimental.multimodal
import test_utils  # local file import from experimental.multimodal

flags.adopt_module_key_flags(deterministic)
FLAGS = flags.FLAGS


class DeterministicTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    data_dir = os.path.join(baseline_root_dir, 'testing_data')
    logging.info('data_dir contents: %s', os.listdir(data_dir))
    self.data_dir = data_dir

  # TODO(jallingham): Switch to MS COCO once tfds.testing.mock works:
  # https://github.com/tensorflow/datasets/issues/4125
  @parameterized.parameters(
      ('imagenet2012', 'vit', 1.0986123, 2.3658294, 0.3333333, 8.314780, False),
      ('imagenet2012', 'vit', 1.0986123, 2.3658294, 0.3333333, 8.314780, True),
      # The fewshot function has not been adapted for including states.
      ('imagenet2012', 'resnet', 1.0986123, 2.3658294, -float('inf'), 8.3147805,
       False),
  )

  @flagsaver.flagsaver
  def test_deterministic_script(self, dataset_name, model_type,
                                correct_train_loss, correct_val_loss,
                                correct_fewshot_acc_sum,
                                correct_zeroshot_imagenet_loss,
                                simulate_failure):
    data_dir = self.data_dir
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config = test_utils.get_config(
        dataset_name=dataset_name, model_type=model_type)
    config.dataset_dir = data_dir
    num_examples = config.batch_size * config.total_steps

    if not simulate_failure:
      # Check for any errors.
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = deterministic.main(
            config, output_dir)
    else:
      # Check for the ability to restart from a previous checkpoint (after
      # failure, etc.).
      # NOTE: Use this flag to simulate failing at a certain step.
      config.testing_failure_step = config.total_steps - 1
      config.checkpoint_steps = config.testing_failure_step
      config.keep_checkpoint_steps = config.checkpoint_steps
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        deterministic.main(config, output_dir)

      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))
      checkpoint = checkpoint_utils.load_checkpoint(None, checkpoint_path)
      self.assertEqual(
          int(checkpoint['opt']['state']['step']),
          config.testing_failure_step)

      # This should resume from the failed step.
      del config.testing_failure_step
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = deterministic.main(
            config, output_dir)

    # Check for reproducibility.
    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info(
        '(train_loss, val_loss, fewshot_acc_sum, zeroshot_imagenet_loss) = %s, %s, %s, %s',
        train_loss, val_loss['val'], fewshot_acc_sum,
        val_loss['zeroshot_imagenet'])
    np.testing.assert_allclose(train_loss, correct_train_loss,
                               rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(val_loss['val'], correct_val_loss,
                               rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(val_loss['zeroshot_imagenet'],
                               correct_zeroshot_imagenet_loss,
                               rtol=1e-06, atol=1e-06)

  @parameterized.parameters(
      ('imagenet2012', 1.0986125, 1.9434237, 0.333333343, 8.91884316338433),)
  @flagsaver.flagsaver
  def test_loading_pretrained_model(self, dataset_name, correct_train_loss,
                                    correct_val_loss, correct_fewshot_acc_sum,
                                    correct_zeroshot_imagenet_loss):
    data_dir = self.data_dir
    config = test_utils.get_config(dataset_name=dataset_name)
    config.dataset_dir = data_dir
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    # Run to save a checkpoint, then use that as a pretrained model.
    num_examples = config.batch_size * config.total_steps
    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      deterministic.main(config, output_dir)

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    self.assertTrue(os.path.exists(checkpoint_path))

    new_output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.model_init = checkpoint_path
    config.total_steps = 3

    num_examples = config.batch_size * config.total_steps

    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      train_loss, val_loss, fewshot_results = deterministic.main(
          config, new_output_dir)

    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info(
        '(train_loss, val_loss, fewshot_acc_sum, zeroshot_imagenet_loss) = %s, %s, %s, %s',  # pylint: disable=line-too-long
        train_loss, val_loss['val'], fewshot_acc_sum,
        val_loss['zeroshot_imagenet'])
    np.testing.assert_allclose(train_loss, correct_train_loss,
                               rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(val_loss['val'], correct_val_loss,
                               rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(val_loss['zeroshot_imagenet'],
                               correct_zeroshot_imagenet_loss,
                               rtol=1e-06, atol=1e-06)


if __name__ == '__main__':
  tf.test.main()
