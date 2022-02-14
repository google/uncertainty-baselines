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

"""Tests for classifier utilities."""
import tensorflow as tf

from uncertainty_baselines.models import classifier_utils


class ClassifierUtilsTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.random_seed = 42
    self.num_classes = 2
    self.batch_size = 4
    self.hidden_dim = 8

  def test_mpnn_gp_classifier(self):
    """Tests if GP classifier can be compiled successfully."""
    # Compile a mock input model
    gp_layer_kwargs = dict(
        num_inducing=1024,
        gp_kernel_scale=1.,
        gp_output_bias=0.,
        normalize_input=True,
        gp_cov_momentum=0.999,
        gp_cov_ridge_penalty=1e-6)

    # Compiles classifier model.
    model = classifier_utils.build_classifier(
        num_classes=self.num_classes,
        gp_layer_kwargs=gp_layer_kwargs,
        use_gp_layer=True)

    # Computes output.
    tf.random.set_seed(self.random_seed)
    inputs_tensor = tf.random.normal((self.batch_size, self.hidden_dim))
    logits, covmat = model(inputs_tensor, training=False)

    # Check if output tensors have correct shapes.
    logits_shape_observed = logits.shape.as_list()
    covmat_shape_observed = covmat.shape.as_list()

    logits_shape_expected = [self.batch_size, self.num_classes]
    covmat_shape_expected = [self.batch_size, self.batch_size]

    self.assertEqual(logits_shape_observed, logits_shape_expected)
    self.assertEqual(covmat_shape_observed, covmat_shape_expected)


if __name__ == "__main__":
  tf.test.main()
