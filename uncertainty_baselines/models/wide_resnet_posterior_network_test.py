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

"""Tests for Wide Posterior ResNet."""

import tensorflow as tf
import uncertainty_baselines as ub


class WideResnetPosteriorNetworkTest(tf.test.TestCase):

  def testWideResnetPosteriorNetwork(self):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 5
    input_shape = (32, 32, 1)
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    class_counts = tf.cast(tf.math.bincount(tf.cast(tf.reshape(labels, -1),
                                                    tf.int32)), tf.float32)

    for flow_type in ['maf', 'affine', 'radial']:
      model = ub.models.wide_resnet_posterior_network(
          input_shape=input_shape,
          depth=10,
          width_multiplier=1,
          num_classes=num_classes,
          l2=0.,
          version=2,
          class_counts=class_counts,
          flow_type=flow_type)
      uce_loss_fn = ub.models.uce_loss(sparse=True,
                                       num_classes=num_classes)
      model.compile('adam', loss=uce_loss_fn)
      history = model.fit(
          dataset, steps_per_epoch=dataset_size // batch_size, epochs=2)

      loss_history = history.history['loss']
      self.assertAllGreaterEqual(loss_history, 0.)


if __name__ == '__main__':
  tf.test.main()
