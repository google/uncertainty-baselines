# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

# Lint as: python3
"""Tests for MIMO on ResNet-50."""
import tensorflow as tf
import uncertainty_baselines as ub


class Resnet50MimoTest(tf.test.TestCase):

  def testResNet50Mimo(self):
    tf.random.set_seed(839382)
    dataset_size = 10
    batch_size = 6
    input_shape = (4, 224, 224, 3)
    ensemble_size = 4
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 4)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.resnet50_mimo(
        input_shape=input_shape,
        width_multiplier=2,
        num_classes=num_classes,
        ensemble_size=ensemble_size)
    model.compile(
        'adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(dataset,
                        steps_per_epoch=dataset_size // batch_size,
                        epochs=2)

    loss_history = history.history['loss']
    self.assertAllGreaterEqual(loss_history, 0.)


if __name__ == '__main__':
  tf.test.main()

