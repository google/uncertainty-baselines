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

# Lint as: python3
"""Tests for MIMO with Heteroscedastic Approach on ResNet-50."""
import tensorflow as tf
import uncertainty_baselines as ub


class Resnet50HetMimoTest(tf.test.TestCase):

  def testResNet50HetMimo(self):
    tf.random.set_seed(839382)
    dataset_size = 30
    batch_size = 6
    ensemble_size = 2
    input_shape = (ensemble_size, 224, 224, 3)
    num_classes = 5
    temperature = 1.5
    num_factors = 3
    num_mc_samples = 10
    width_multiplier = 1
    share_het_layer = True

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, ensemble_size)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.resnet50_het_mimo(
        input_shape=(ensemble_size, 224, 224, 3),
        num_classes=num_classes,
        ensemble_size=ensemble_size,
        num_factors=num_factors,
        temperature=temperature,
        num_mc_samples=num_mc_samples,
        share_het_layer=share_het_layer,
        width_multiplier=width_multiplier
        )
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
