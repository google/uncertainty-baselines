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

"""Tests for Rank-1 BNN with Heteroscedastic Approach on ResNet-50."""
import tensorflow as tf
import uncertainty_baselines as ub


class Resnet50HetRank1Test(tf.test.TestCase):

  def testResNet50HetRank1(self):
    tf.random.set_seed(839382)
    temperature = 1.5
    num_factors = 3
    num_mc_samples = 10

    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 4  # must be divisible by ensemble_size
    input_shape = (32, 32, 1)
    num_classes = 4

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.resnet50_het_rank1(
        input_shape=input_shape,
        num_classes=num_classes,
        alpha_initializer='trainable_normal',
        gamma_initializer='trainable_normal',
        alpha_regularizer='normal_kl_divergence',
        gamma_regularizer='normal_kl_divergence',
        use_additive_perturbation=False,
        ensemble_size=4,
        random_sign_init=0.75,
        dropout_rate=0.001,
        prior_stddev=0.05,
        use_tpu=True,
        use_ensemble_bn=False,
        num_factors=num_factors,
        temperature=temperature,
        num_mc_samples=num_mc_samples)
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
