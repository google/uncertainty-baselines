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

"""Tests for tpu related functions."""

from typing import Sequence

from absl import flags
from absl.testing import parameterized
import tensorflow as tf
import train_lib  # local file import from experimental.language_structure.vrnn

FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')


def _create_step_fn(num_outputs: int):

  @tf.function
  def _test_step_fn(inputs: Sequence[tf.Tensor]):
    if num_outputs == 0:
      return
    elif num_outputs == 1:
      return inputs[0]
    else:
      return inputs

  return _test_step_fn


class CreateStepFnTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.named_parameters(('case1', 0, True), ('case2', 1, True),
                                  ('case3', 1, False), ('case4', 2, True),
                                  ('case5', 2, False))
  def test_run_in_tpu_strategy(self, num_outputs, distributed):
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    step_fn = _create_step_fn(num_outputs)

    step_fn_inputs = [
        tf.constant([[1, 2], [3, 4]]),
        tf.constant([[5, 6], [7, 8]])
    ]
    output_types = [tf.int32, tf.int32]
    dataset = tf.data.Dataset.from_tensor_slices(
        (step_fn_inputs[0], step_fn_inputs[1]))
    if distributed:
      dataset = strategy.experimental_distribute_dataset(dataset)

    run_steps_fn = train_lib.create_run_steps_fn(step_fn, strategy, distributed,
                                                 output_types[:num_outputs])
    step_fn_outputs = run_steps_fn(iter(dataset), tf.constant(2))

    self.assertLen(step_fn_outputs, num_outputs)
    for step_fn_output, step_fn_input in zip(step_fn_outputs, step_fn_inputs):
      expected = tf.reshape(step_fn_input, [-1])
      self.assertAllEqual(expected, step_fn_output)


if __name__ == '__main__':
  tf.test.main()
