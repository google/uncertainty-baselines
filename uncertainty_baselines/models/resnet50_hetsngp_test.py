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

"""Tests for HetSNGP ResNet-50."""

import tensorflow as tf
import uncertainty_baselines as ub


class Resnet50HetSNGPTest(tf.test.TestCase):

  def testResnet50HetSNGP(self):
    tf.random.set_seed(83922)
    dataset_size = 10
    batch_size = 5
    input_shape = (32, 32, 1)
    num_classes = 3

    features = tf.random.normal((dataset_size,) + input_shape)
    coeffs = tf.random.normal([tf.reduce_prod(input_shape), num_classes])
    net = tf.reshape(features, [dataset_size, -1])
    logits = tf.matmul(net, coeffs)
    labels = tf.random.categorical(logits, 1)
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.resnet50_hetsngp(
        input_shape=input_shape,
        batch_size=batch_size,
        num_classes=num_classes,
        num_factors=num_classes,
        use_mc_dropout=False,
        dropout_rate=0.,
        filterwise_dropout=False,
        use_gp_layer=True,
        gp_hidden_dim=1024,
        gp_scale=1.,
        gp_bias=0.,
        gp_input_normalization=False,
        gp_random_feature_type='orf',
        gp_cov_discount_factor=-1.,
        gp_cov_ridge_penalty=1.,
        gp_output_imagenet_initializer=False,
        use_spec_norm=True,
        spec_norm_iteration=1,
        spec_norm_bound=6.,
        temperature=1.,
        num_mc_samples=1000,
        eps=1e-5,
        sngp_var_weight=1.,
        het_var_weight=1.,
        omit_last_layer=False)

    model.compile(
        'adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = model.fit(
        dataset, steps_per_epoch=dataset_size // batch_size, epochs=2)

    loss_history = history.history['loss']
    self.assertAllGreaterEqual(loss_history, 0.)


if __name__ == '__main__':
  tf.test.main()
