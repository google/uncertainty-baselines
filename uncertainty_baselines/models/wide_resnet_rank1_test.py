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

"""Tests for Rank1 BNNs on a Wide ResNet."""

import tensorflow as tf
import uncertainty_baselines as ub


class WideResnetRank1(tf.test.TestCase):

  def testWideResnetRank1(self):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 4  # must be divisible by ensemble_size
    input_shape = (32, 32, 1)
    num_classes = 2

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.wide_resnet_rank1(
        input_shape=input_shape,
        depth=10,
        width_multiplier=1,
        num_classes=num_classes,
        alpha_initializer='trainable_deterministic',
        gamma_initializer='trainable_deterministic',
        alpha_regularizer='',
        gamma_regularizer='',
        use_additive_perturbation=False,
        ensemble_size=2,
        random_sign_init=-0.5,
        dropout_rate=1e-3,
        prior_mean=1,
        prior_stddev=0.1)
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
