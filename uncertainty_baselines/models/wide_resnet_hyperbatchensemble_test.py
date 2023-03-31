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

"""Tests for BatchEnsemble on a Wide ResNet."""

import tensorflow as tf
import uncertainty_baselines as ub


class WideResnetBatchensemble(tf.test.TestCase):

  def testWideResnetHyperBatchensemble(self):
    tf.random.set_seed(83922)
    depth = 28
    width = 10
    batch_size = 4  # must be divisible by ensemble_size
    input_shape = (32, 32, 1)
    num_classes = 2
    ensemble_size = 2
    random_sign_init = 0.5
    l2_batchnorm = 0.1

    # build model
    # configs
    dict_ranges = {'min': 0.01, 'max': 100.}
    ranges = [dict_ranges for _ in range(6)]
    model_config = {
        'key_to_index': {
            'input_conv_l2_kernel': 0,
            'group_l2_kernel': 1,
            'group_1_l2_kernel': 2,
            'group_2_l2_kernel': 3,
            'dense_l2_kernel': 4,
            'dense_l2_bias': 5,
        },
        'ranges': ranges,
        'test': None
    }
    lambdas_config = ub.models.HyperBatchEnsembleLambdaConfig(
        model_config['ranges'], model_config['key_to_index'])

    # build embedding model for lambdas (hyperparameters)
    # TODO(florianwenzel): We may move the wrn specific e_head_dims also to
    #                      ub.models.wide_resnet_hyperbatchensemble
    filters_resnet = [16]
    for i in range(0, 3):  # 3 groups of blocks
      filters_resnet.extend([16 * width * 2**i] * 9)  # 9 layers in each block

    e_head_dims = [x for x in filters_resnet] + [2 * num_classes]

    e_models = ub.models.hyperbatchensemble_e_factory(
        lambdas_config.input_shape,
        e_head_dims=e_head_dims,
        e_body_arch=(),
        e_shared_arch=(),
        activation='tanh',
        use_bias=False)

    # build hyperbatchensemble
    model = ub.models.wide_resnet_hyperbatchensemble(
        input_shape=input_shape,
        depth=depth,
        width_multiplier=width,
        num_classes=num_classes,
        ensemble_size=ensemble_size,
        random_sign_init=random_sign_init,
        config=lambdas_config,
        e_models=e_models,
        l2_batchnorm_layer=l2_batchnorm,
        regularize_fast_weights=False,
        fast_weights_eq_contraint=True,
        version=2)

    # evaluate model at random input and test output shape
    input_data = tf.random.uniform((batch_size,) + input_shape)
    input_lambdas = tf.random.uniform((batch_size,) +
                                      lambdas_config.input_shape)
    logits = model([input_data, input_lambdas])
    self.assertEqual(logits.shape, (batch_size, num_classes))


if __name__ == '__main__':
  tf.test.main()
