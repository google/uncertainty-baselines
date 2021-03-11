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

"""Tests for ResNet50 variational inference."""

import tensorflow as tf

import uncertainty_baselines as ub


class Resnet50VariationalTest(tf.test.TestCase):

  def testResNet50Variational(self):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 5
    input_shape = (224, 224, 3)
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.resnet50_variational(
        input_shape=input_shape,
        num_classes=num_classes,
        prior_stddev=0.1,
        dataset_size=dataset_size,
        stddev_mean_init=1e-3,
        stddev_stddev_init=1e-3)
    model.compile(
        'adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(
        dataset, steps_per_epoch=dataset_size // batch_size, epochs=2)

    loss_history = history.history['loss']
    self.assertAllGreaterEqual(loss_history, 0.)
    self.assertGreater(loss_history[0], loss_history[-1])


if __name__ == '__main__':
  tf.test.main()
