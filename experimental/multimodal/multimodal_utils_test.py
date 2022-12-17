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

"""Tests for multimodal_utils."""

from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import multimodal_utils  # local file import from experimental.multimodal


class MultimodalUtilsTest(tf.test.TestCase):

  def test_contrastive_loss_logits(self):
    zimg = jnp.array([[1., 2., 3.], [4., 5., 6.], [1., 0., 0.]])
    ztext = jnp.array([[-1., -2., -3.], [1., 2., 3.], [1., 0., 0.]])

    _, logits = multimodal_utils.bidirectional_contrastive_loss(zimg, ztext)

    np.testing.assert_allclose(
        logits,
        jnp.array([[-14., 14., 1.], [-32., 32., 4.], [-1., 1., 1.]]))

  def test_contrastive_loss_no_reduction_no_mask(self):
    zimg = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ztext = jnp.array([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]])

    loss, logits = multimodal_utils.bidirectional_contrastive_loss(
        zimg, ztext, mask=None, reduction=False)

    np.testing.assert_allclose(
        logits,
        jnp.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 1.]]))

    expected_loss = -0.5 * jnp.array([
        jnp.log(jnp.e**2 / (jnp.e + 2)**2),
        jnp.log(1 / (3 * (jnp.e + 2))),
        jnp.log(jnp.e**2 / ((2 + jnp.e) * (1 + 2 * jnp.e)))
    ])

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)

  def test_contrastive_loss_reduction_no_mask(self):
    zimg = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ztext = jnp.array([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]])

    loss, logits = multimodal_utils.bidirectional_contrastive_loss(
        zimg, ztext, mask=None, reduction=True)

    np.testing.assert_allclose(
        logits,
        jnp.array([[1., 0., 0.], [0., 0., 0.], [0., 1., 1.]]))

    expected_loss = jnp.mean(-0.5 * jnp.array([
        jnp.log(jnp.e**2 / (jnp.e + 2)**2),
        jnp.log(1 / (3 * (jnp.e + 2))),
        jnp.log(jnp.e**2 / ((2 + jnp.e) * (1 + 2 * jnp.e)))
    ]))

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)

  def test_contrastive_loss_no_reduction_mask(self):
    zimg = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ztext = jnp.array([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]])

    loss, logits = multimodal_utils.bidirectional_contrastive_loss(
        zimg, ztext, mask=jnp.array([1, 1, 0]), reduction=False)

    np.testing.assert_allclose(
        logits,
        jnp.array([[1., 0., -jnp.inf],
                   [0., 0., -jnp.inf],
                   [-jnp.inf, -jnp.inf, -jnp.inf]]))

    expected_loss = -0.5 * jnp.array([
        jnp.log(jnp.e**2 / (jnp.e + 1)**2),
        jnp.log(1 / 4),
        0
    ])

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)

  def test_contrastive_loss_reduction_mask(self):
    zimg = jnp.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    ztext = jnp.array([[1., 0., 0.], [0., 0., 1.], [0., 0., 1.]])

    loss, logits = multimodal_utils.bidirectional_contrastive_loss(
        zimg, ztext, mask=jnp.array([1, 1, 0]), reduction=True)

    np.testing.assert_allclose(
        logits,
        jnp.array([[1., 0., -jnp.inf],
                   [0., 0., -jnp.inf],
                   [-jnp.inf, -jnp.inf, -jnp.inf]]))

    expected_loss = jnp.sum(-0.5 * jnp.array([
        jnp.log(jnp.e**2 / (jnp.e + 1)**2),
        jnp.log(1 / 4)
    ])) / 2

    np.testing.assert_allclose(loss, expected_loss, atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
  tf.test.main()
