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

"""Tests for the ViT-SNGP on JFT-300M model script."""
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
import checkpoint_utils  # local file import from baselines.jft
import sngp  # local file import from baselines.jft
import test_utils  # local file import from baselines.jft

FLAGS = flags.FLAGS


class SNGPTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    data_dir = os.path.join(baseline_root_dir, 'testing_data')
    logging.info('data_dir contents: %s', os.listdir(data_dir))
    self.data_dir = data_dir

  @parameterized.parameters(
      ('imagenet2012', 'token', 2, 65.89198, 70.65269470214844, 0.56, False),
      ('imagenet2012', 'token', 2, 65.89198, 70.65269470214844, 0.56, True),
      ('imagenet2012', 'token', None, 9.569044, 11.68548117743598, 1.11, False),
      ('imagenet2012', 'gap', 2, 65.891975, 70.65279981825087, 1.00, False),
      ('imagenet2012', 'gap', None, 27.400314, 96.92751587761774, 1.56, False),
      ('imagenet2012', 'gap', None, 27.400314, 96.92751587761774, 1.56, True),
  )
  @flagsaver.flagsaver
  def test_sngp_script(self, dataset_name, classifier, representation_size,
                       correct_train_loss, correct_val_loss,
                       correct_fewshot_acc_sum, simulate_failure):
    data_dir = self.data_dir
    config = test_utils.get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size,
        use_sngp=True,
        use_gp_layer=True)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir
    num_examples = config.batch_size * config.total_steps

    if not simulate_failure:
      # Check for any errors.
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = sngp.main(config, output_dir)
    else:
      # Check for the ability to restart from a previous checkpoint (after
      # failure, etc.).
      output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      # NOTE: Use this flag to simulate failing at a certain step.
      config.testing_failure_step = config.total_steps - 1
      config.checkpoint_steps = config.testing_failure_step
      config.keep_checkpoint_steps = config.checkpoint_steps
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        sngp.main(config, output_dir)

      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))
      checkpoint = checkpoint_utils.load_checkpoint(None, checkpoint_path)
      self.assertEqual(
          int(checkpoint['opt']['state']['step']),
          config.testing_failure_step)

      # This should resume from the failed step.
      del config.testing_failure_step
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, fewshot_results = sngp.main(config, output_dir)

    # Check for reproducibility.
    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info('(train_loss, val_loss, fewshot_acc_sum) = %s, %s, %s',
                 train_loss, val_loss['val'], fewshot_acc_sum)
    # TODO(dusenberrymw): Determine why the SNGP script is non-deterministic.
    np.testing.assert_allclose(train_loss, correct_train_loss,
                               atol=0.025, rtol=0.3)
    np.testing.assert_allclose(val_loss['val'], correct_val_loss,
                               atol=0.02, rtol=0.3)

  @parameterized.parameters(
      ('imagenet2012', 'token', None, True, 29.3086, 17.3351, 0.67, 'imagenet'),
  )
  @flagsaver.flagsaver
  def test_loading_pretrained_model(self, dataset_name, classifier,
                                    representation_size, use_gp_layer,
                                    correct_train_loss,
                                    correct_val_loss, correct_fewshot_acc_sum,
                                    finetune_dataset_name):
    data_dir = self.data_dir
    config = test_utils.get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size,
        use_sngp=True,
        use_gp_layer=use_gp_layer)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir
    num_examples = config.batch_size * config.total_steps

    # Run to save a checkpoint, then use that as a pretrained model.
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      sngp.main(config, output_dir)

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    self.assertTrue(os.path.exists(checkpoint_path))

    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.model_init = checkpoint_path
    config.model.representation_size = None
    if finetune_dataset_name == 'cifar':
      config.dataset = 'cifar10'
      config.val_split = f'train[:{num_examples}]'
      config.train_split = f'train[{num_examples}:{num_examples*2}]'
      config.num_classes = 10
      config.ood_datasets = ['cifar100']
      config.ood_num_classes = [100]
      config.ood_split = f'test[{num_examples*2}:{num_examples*3}]'
      config.ood_methods = ['maha', 'entropy', 'rmaha', 'msp']
      config.eval_on_cifar_10h = True
      config.cifar_10h_split = f'test[:{num_examples}]'
      config.pp_eval_cifar_10h = (
          'decode|resize(384)|value_range(-1, 1)|keep(["image", "labels"])')
    elif finetune_dataset_name == 'imagenet':
      config.dataset = 'imagenet2012'
      config.val_split = f'train[:{num_examples}]'
      config.train_split = f'train[{num_examples}:{num_examples*2}]'
      config.num_classes = 1000
    pp_common = '|value_range(-1, 1)'
    pp_common += f'|onehot({config.num_classes}, key="label", key_result="labels")'  # pylint: disable=line-too-long
    pp_common += '|keep(["image", "labels"])'
    config.pp_train = 'decode|resize_small(512)|random_crop(384)' + pp_common
    config.pp_eval = 'decode|resize(384)' + pp_common
    config.fewshot.pp_train = 'decode|resize_small(512)|central_crop(384)|value_range(-1,1)|drop("segmentation_mask")'
    config.fewshot.pp_eval = 'decode|resize(384)|value_range(-1,1)|drop("segmentation_mask")'
    if config.get('ood_num_classes'):
      pp_eval_ood = []
      for num_classes in config.ood_num_classes:
        pp_eval_ood.append(
            config.pp_eval.replace(f'onehot({config.num_classes}',
                                   f'onehot({num_classes}'))
      config.pp_eval_ood = pp_eval_ood

    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      logging.info('!!!config %s', config)
      train_loss, val_loss, fewshot_results = sngp.main(config, output_dir)

    fewshot_acc_sum = sum(jax.tree_util.tree_flatten(fewshot_results)[0])
    logging.info('(train_loss, val_loss, fewshot_acc_sum) = %s, %s, %s',
                 train_loss, val_loss['val'], fewshot_acc_sum)
    # TODO(dusenberrymw,jjren): Add a reproducibility test for OOD eval.
    # TODO(dusenberrymw): Determine why the SNGP script is non-deterministic.
    np.testing.assert_allclose(train_loss, correct_train_loss,
                               atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(val_loss['val'], correct_val_loss,
                               atol=2e-3, rtol=2e-3)


if __name__ == '__main__':
  tf.test.main()
