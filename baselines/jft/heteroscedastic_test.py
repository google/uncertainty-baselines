# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""Tests for the heteroscedastic ViT on JFT-300M model script."""
import os.path
import pathlib
import tempfile

from absl import flags
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
import jax
# import ml_collections
import tensorflow as tf
import tensorflow_datasets as tfds
from  baselines.jft import checkpoint_utils  # local file import
from  baselines.jft import heteroscedastic  # local file import
from  baselines.jft import test_utils  # local file import

FLAGS = flags.FLAGS


class HeteroscedasticTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    self.data_dir = os.path.join(baseline_root_dir, 'testing_data')

  @parameterized.parameters(
      ('imagenet2012', 'token', 2, 670.56195, 592.604492, 0.18, False),
      ('imagenet2012', 'token', 2, 670.56195, 592.604492, 0.18, True),
      # ('imagenet2012', 'token', None, 17691.684, 13866.16232638, 0.16, False),
      ('imagenet2012', 'gap', 2, 681.64264, 660.940186, 0.17, False),
      ('imagenet2012', 'gap', None, 689.7084, 643.398709, 0.25, False),
      ('imagenet2012', 'gap', None, 689.7084, 643.398709, 0.25, True),
  )
  @flagsaver.flagsaver
  def test_heteroscedastic_script(self, dataset_name, classifier,
                                  representation_size, correct_train_loss,
                                  correct_val_loss, correct_fewshot_acc_sum,
                                  simulate_failure):
    data_dir = self.data_dir
    config = test_utils.get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir

    if not simulate_failure:
      # Check for any errors.
      with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = heteroscedastic.main(
            config, output_dir)
    else:
      # Check for the ability to restart from a previous checkpoint (after
      # failure, etc.).
      output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      # NOTE: Use this flag to simulate failing at a certain step.
      config.testing_failure_step = config.total_steps - 1
      config.checkpoint_steps = config.testing_failure_step
      config.keep_checkpoint_steps = config.checkpoint_steps
      with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
        heteroscedastic.main(config, output_dir)

      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))
      checkpoint = checkpoint_utils.load_checkpoint(None, checkpoint_path)
      self.assertEqual(
          int(checkpoint['opt']['state']['step']),
          config.testing_failure_step)

      # This should resume from the failed step.
      del config.testing_failure_step
      with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = heteroscedastic.main(
            config, output_dir)

    # Check for reproducibility.
    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info('(train_loss, val_loss, fewshot_acc_sum) = %s, %s, %s',
                 train_loss, val_loss['val'], fewshot_acc_sum)
    self.assertAllClose(train_loss, correct_train_loss)
    self.assertAllClose(val_loss['val'], correct_val_loss)

  @parameterized.parameters(
      ('imagenet2012', 'token', 2, 548.9606, 517.419963, 0.10, 'imagenet'),
  )
  @flagsaver.flagsaver
  def test_loading_pretrained_model(self, dataset_name, classifier,
                                    representation_size, correct_train_loss,
                                    correct_val_loss, correct_fewshot_acc_sum,
                                    finetune_dataset_name):
    data_dir = self.data_dir
    config = test_utils.get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir

    # Run to save a checkpoint, then use that as a pretrained model.
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
      heteroscedastic.main(config, output_dir)

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    self.assertTrue(os.path.exists(checkpoint_path))

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.model_init = checkpoint_path
    config.model.representation_size = None
    if finetune_dataset_name == 'cifar':
      config.dataset = 'cifar10'
      config.val_split = 'train[:9]'
      config.train_split = 'train[30:60]'
      config.num_classes = 10
      config.ood_dataset = 'cifar100'
      config.ood_split = 'test[10:20]'
      config.ood_methods = ['maha', 'rmaha', 'msp']
      config.eval_on_cifar_10h = True
      config.cifar_10h_split = 'test[:9]'
      config.pp_eval_cifar_10h = (
          'decode|resize(384)|value_range(-1, 1)|keep(["image", "labels"])')  # pylint: disable=line-too-long
    elif finetune_dataset_name == 'imagenet':
      config.dataset = 'imagenet2012'
      config.val_split = 'train[:9]'
      config.train_split = 'train[30:60]'
      config.num_classes = 1000
      config.eval_on_imagenet_real = True
      config.imagenet_real_split = 'validation[:9]'
      config.pp_eval_imagenet_real = (
          'decode|resize(384)|value_range(-1, 1)|keep(["image", "labels"])')  # pylint: disable=line-too-long
    pp_common = '|value_range(-1, 1)'
    pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
    pp_common += '|keep(["image", "labels"])'
    config.pp_train = 'decode|resize_small(512)|random_crop(384)' + pp_common
    config.pp_eval = 'decode|resize(384)' + pp_common
    config.fewshot.pp_train = 'decode|resize_small(512)|central_crop(384)|value_range(-1,1)'
    config.fewshot.pp_eval = 'decode|resize(384)|value_range(-1,1)'

    with tfds.testing.mock_data(num_examples=100, data_dir=data_dir):
      train_loss, val_loss, fewshot_results = heteroscedastic.main(
          config, output_dir)

    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info('(train_loss, val_loss, fewshot_acc_sum) = %s, %s, %s',
                 train_loss, val_loss['val'], fewshot_acc_sum)
    self.assertAllClose(train_loss, correct_train_loss)
    self.assertAllClose(val_loss['val'], correct_val_loss)


if __name__ == '__main__':
  tf.test.main()
