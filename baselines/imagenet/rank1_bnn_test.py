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

"""Tests for the Rank-1 BNN ResNet-50 on ImageNet baseline script."""

import os.path
import pathlib
import tempfile

from absl import flags
from absl import logging
from absl.testing import flagsaver
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import rank1_bnn  # local file import from baselines.imagenet

flags.adopt_module_key_flags(rank1_bnn)
FLAGS = flags.FLAGS


class Rank1BnnTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    data_dir = os.path.join(baseline_root_dir, 'testing_data')
    logging.info('data_dir contents: %s', os.listdir(data_dir))
    self.data_dir = data_dir

  @flagsaver.flagsaver
  def test_script(self):
    data_dir = self.data_dir
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())

    FLAGS.data_dir = data_dir
    FLAGS.output_dir = output_dir
    FLAGS.use_gpu = True

    FLAGS.kl_annealing_epochs = 1
    FLAGS.ensemble_size = 2
    FLAGS.per_core_batch_size = 3
    FLAGS.num_cores = 1
    FLAGS.train_epochs = 1
    FLAGS.steps_per_epoch_train = 1
    FLAGS.steps_per_epoch_eval = FLAGS.steps_per_epoch_train
    FLAGS.corruptions_interval = -1  # TODO(dusenberrymw): Speed-up & re-enable.
    FLAGS.checkpoint_interval = 1
    FLAGS.num_eval_samples = 1

    num_examples = (
        FLAGS.per_core_batch_size * FLAGS.num_cores * FLAGS.train_epochs *
        FLAGS.steps_per_epoch_train)

    with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
      # Check for any errors. TensorFlow isn't reproducible, so we can't check
      # for exact reproducibility (like we can with JAX).
      rank1_bnn.main(None)

    checkpoint_path = os.path.join(output_dir, 'checkpoint')
    self.assertTrue(os.path.exists(checkpoint_path))


if __name__ == '__main__':
  tf.test.main()
