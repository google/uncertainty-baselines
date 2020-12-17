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

"""Augmix utilities."""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

CIFAR10_MEAN = tf.constant([0.4914, 0.4822, 0.4465])
CIFAR10_STD = tf.constant([0.2023, 0.1994, 0.2010])


def normalize_convert_image(input_image,
                            dtype,
                            mean=CIFAR10_MEAN,
                            std=CIFAR10_STD):
  input_image = tf.image.convert_image_dtype(input_image, dtype)
  return (input_image - mean) / std


def augment_and_mix(image,
                    depth,
                    width,
                    prob_coeff,
                    augmenter,
                    dtype,
                    mean=CIFAR10_MEAN,
                    std=CIFAR10_STD):
  """Apply mixture of augmentations to image."""

  mix_weight = tf.squeeze(tfd.Beta([prob_coeff], [prob_coeff]).sample([1]))

  if width > 1:
    branch_weights = tf.squeeze(tfd.Dirichlet([prob_coeff] * width).sample([1]))
  else:
    branch_weights = tf.constant([1.])

  if depth < 0:
    depth = tf.random.uniform([width],
                              minval=1,
                              maxval=4,
                              dtype=tf.dtypes.int32)
  else:
    depth = tf.constant([depth] * width)

  mix = tf.cast(tf.zeros_like(image), tf.float32)
  for i in tf.range(width):
    branch_img = tf.identity(image)
    for _ in tf.range(depth[i]):
      branch_img = augmenter.distort(branch_img)
    branch_img = normalize_convert_image(branch_img, dtype, mean, std)
    mix += branch_weights[i] * branch_img

  return mix_weight * mix + (1 - mix_weight) * normalize_convert_image(
      image, dtype, mean, std)


def do_augmix(image,
              params,
              augmenter,
              dtype,
              mean=CIFAR10_MEAN,
              std=CIFAR10_STD):
  """Apply augmix augmentation to image."""
  depth = params['augmix_depth']
  width = params['augmix_width']
  prob_coeff = params['augmix_prob_coeff']
  count = params['aug_count']

  augmented = [
      augment_and_mix(image, depth, width, prob_coeff, augmenter, dtype, mean,
                      std) for _ in range(count)
  ]
  image = normalize_convert_image(image, dtype, mean, std)
  return tf.stack([image] + augmented, 0)


def mixup(batch_size, aug_params, images, labels):
  """Applies Mixup regularization to a batch of images and labels.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  augmix = aug_params.get('augmix', False)
  alpha = aug_params.get('mixup_alpha', 0.)
  aug_count = aug_params.get('aug_count', 3)

  # 4 is hard-coding to aug_count=3. Fix this later!
  if augmix:
    mix_weight = tfd.Beta(alpha, alpha).sample([batch_size, aug_count + 1, 1])
  else:
    mix_weight = tfd.Beta(alpha, alpha).sample([batch_size, 1])
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  if augmix:
    images_mix_weight = tf.reshape(mix_weight,
                                   [batch_size, aug_count + 1, 1, 1, 1])
  else:
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))

  if augmix:
    labels = tf.reshape(
        tf.tile(labels, [1, aug_count + 1]), [batch_size, aug_count + 1, -1])
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
    labels_mix = tf.reshape(
        tf.transpose(labels_mix, [1, 0, 2]), [batch_size * (aug_count + 1), -1])
  else:
    labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
  return images_mix, labels_mix


def adaptive_mixup_aug(batch_size, aug_params, images, labels):
  """Applies Confidence Adjusted Mixup (CAMixup) regularization.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied.
  """
  augmix = aug_params['augmix']
  ensemble_size = aug_params['ensemble_size']
  mixup_coeff = aug_params['mixup_coeff']
  scalar_labels = tf.argmax(labels, axis=1)
  alpha = tf.gather(mixup_coeff, scalar_labels, axis=-1)  # 4 x Batch_size

  # Need to filter out elements in alpha which equal to 0.
  greater_zero_indicator = tf.cast(alpha > 0, alpha.dtype)
  less_one_indicator = tf.cast(alpha < 1, alpha.dtype)
  valid_alpha_indicator = tf.cast(greater_zero_indicator * less_one_indicator,
                                  tf.bool)
  sampled_alpha = tf.where(valid_alpha_indicator, alpha, 0.1)
  mix_weight = tfd.Beta(sampled_alpha, sampled_alpha).sample()
  mix_weight = tf.where(valid_alpha_indicator, mix_weight, alpha)
  mix_weight = tf.reshape(mix_weight, [ensemble_size * batch_size, 1])
  mix_weight = tf.clip_by_value(mix_weight, 0, 1)
  mix_weight = tf.maximum(mix_weight, 1. - mix_weight)
  images_mix_weight = tf.reshape(mix_weight,
                                 [ensemble_size * batch_size, 1, 1, 1])
  # Mixup on a single batch is implemented by taking a weighted sum with the
  # same batch in reverse.
  if augmix:
    images_shape = tf.shape(images)
    images = tf.reshape(
        tf.transpose(images, [1, 0, 2, 3, 4]),
        [-1, images_shape[2], images_shape[3], images_shape[4]])
  else:
    images = tf.tile(images, [ensemble_size, 1, 1, 1])
  labels = tf.tile(labels, [ensemble_size, 1])
  images_mix = (
      images * images_mix_weight + images[::-1] * (1. - images_mix_weight))
  labels_mix = labels * mix_weight + labels[::-1] * (1. - mix_weight)
  return images_mix, labels_mix
