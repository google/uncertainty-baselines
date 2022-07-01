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

"""Tests for train_lib."""

from typing import Optional

import numpy as np
import tensorflow as tf
import train_lib  # local file import from experimental.language_structure.vrnn


def _repr_fn(features: tf.Tensor,
             labels: tf.Tensor,
             mask: Optional[tf.Tensor] = None):
  del mask
  return features, labels


class TrainLibTest(tf.test.TestCase):

  def test_build_hidden_state_model(self):
    input_size = 5
    output_size = 3

    model = train_lib.build_hidden_state_model(
        input_size, output_size, learning_rate=1e-3)

    self.assertLen(model.inputs, 1)
    self.assertEqual(model.inputs[0].shape.as_list(), [None, input_size])
    self.assertLen(model.outputs, 1)
    self.assertEqual(model.outputs[0].shape.as_list(), [None, output_size])



if __name__ == '__main__':
  tf.test.main()
