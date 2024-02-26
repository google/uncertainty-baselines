# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

# We use the convention of using mean = np.mean(train_images, axis=(0,1,2))
# and std = np.std(train_images, axis=(0,1,2)).
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465])
CIFAR10_STD = np.array([0.2470, 0.2435, 0.2616])
# Previously we used std = np.mean(np.std(train_images, axis=(1, 2)), axis=0)
# which gave std = tf.constant([0.2023, 0.1994, 0.2010], dtype=dtype), however
# we change convention to use the std over the entire training set instead.


def normalize_convert_image(input_image,
                            dtype,
                            mean=CIFAR10_MEAN,
                            std=CIFAR10_STD):
  if input_image.dtype == tf.uint8:
    input_image = tf.image.convert_image_dtype(input_image, dtype)
  return ((input_image - tf.constant(mean, dtype=dtype)) /
          tf.constant(std, dtype=dtype))


def augment_and_mix(image,
                    depth,
                    width,
                    prob_coeff,
                    augmenter,
                    dtype,
                    mean=CIFAR10_MEAN,
                    std=CIFAR10_STD,
                    seed=None):
  """Apply mixture of augmentations to image."""
  if seed is None:
    seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)
  # We need three seeds, one for sampling from the Beta distribution, one for
  # sampling from the Dirichlet distribution, one for sampling the depth, and a
  # fourth seed to split for each individual RandAugment augmentation.
  augment_seeds = tf.cast(tf.random.experimental.stateless_split(seed, num=4),
                          tf.int32)

  # If seed is (2,), then sample returns a deterministically random sample.
  mix_weight = tf.squeeze(tfd.Beta([prob_coeff], [prob_coeff]).sample(
      [1], seed=augment_seeds[0]))

  if width > 1:
    branch_weights = tf.squeeze(tfd.Dirichlet([prob_coeff] * width).sample(
        [1], seed=augment_seeds[1]))
  else:
    branch_weights = tf.constant([1.])

  if depth < 0:
    depth = tf.random.stateless_uniform([width],
                                        augment_seeds[2],
                                        minval=1,
                                        maxval=4,
                                        dtype=tf.dtypes.int32)
  else:
    depth = tf.constant([depth] * width)

  mix = tf.cast(tf.zeros_like(image), tf.float32)
  # Generate width * sum(depth) seeds for each individual augmentation.
  distort_seeds = tf.random.experimental.stateless_split(
      seed, num=width * tf.reduce_sum(depth))
  seed_count = 0
  for i in tf.range(width):
    branch_img = tf.identity(image)
    for _ in tf.range(depth[i]):
      branch_img = augmenter.distort(branch_img, distort_seeds[seed_count])
      seed_count += 1
    branch_img = normalize_convert_image(branch_img, dtype, mean, std)
    mix += branch_weights[i] * branch_img

  return mix_weight * mix + (1 - mix_weight) * normalize_convert_image(
      image, dtype, mean, std)


def do_augmix(image,
              params,
              augmenter,
              dtype,
              mean=CIFAR10_MEAN,
              std=CIFAR10_STD,
              seed=None):
  """Apply augmix augmentation to image."""
  depth = params['augmix_depth']
  width = params['augmix_width']
  prob_coeff = params['augmix_prob_coeff']
  count = params['aug_count']
  if seed is None:
    seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)
  augment_seeds = tf.random.experimental.stateless_split(seed, num=count)

  augmented = [
      augment_and_mix(image, depth, width, prob_coeff, augmenter, dtype, mean,
                      std, seed=augment_seeds[c]) for c in range(count)
  ]
  image = normalize_convert_image(image, dtype, mean, std)
  return tf.stack([image] + augmented, 0)


def mixup(batch_size, aug_params, images, labels, return_weights=False):
  """Applies Mixup regularization to a batch of images and labels.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  `aug_params` can have the follwing fields:
    augmix: whether or not to run AugMix.
    mixup_alpha: the alpha to use in the Beta distribution.
    aug_count: the number of augmentations to use in AugMix.
    same_mix_weight_per_batch: whether to use the same mix coef over the batch.
    use_truncated_beta: whether to sample from Beta_[0,1](alpha, alpha) or from
       the truncated distribution Beta_[1/2, 1](alpha, alpha).
    use_random_shuffling: Whether to pair images by random shuffling
      (default is a deterministic pairing by reversing the batch).

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]
    return_weights: Whether the mixing weights and ordering of the mixup images
      is returned alongside the new images and labels.

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied if return_weights is False, and (images,
    labels, mix_weights, mixup_index) otherwise.
  """
  augmix = aug_params.get('augmix', False)
  alpha = aug_params.get('mixup_alpha', 0.)
  aug_count = aug_params.get('aug_count', 3)
  same_mix_weight_per_batch = aug_params.get('same_mix_weight_per_batch', False)
  use_truncated_beta = aug_params.get('use_truncated_beta', True)
  use_random_shuffling = aug_params.get('use_random_shuffling', False)

  if augmix and same_mix_weight_per_batch:
    raise ValueError(
        'Can only set one of `augmix` or `same_mix_weight_per_batch`.')

  # 4 is hard-coding to aug_count=3. Fix this later!
  if augmix:
    mix_weight = ed.Beta(
        alpha, alpha, sample_shape=[batch_size, aug_count + 1, 1])
  elif same_mix_weight_per_batch:
    mix_weight = ed.Beta(alpha, alpha, sample_shape=[1, 1])
    mix_weight = tf.tile(mix_weight, [batch_size, 1])
  else:
    mix_weight = ed.Beta(alpha, alpha, sample_shape=[batch_size, 1])

  if use_truncated_beta:
    mix_weight = tf.maximum(mix_weight, 1. - mix_weight)

  if augmix:
    images_mix_weight = tf.reshape(mix_weight,
                                   [batch_size, aug_count + 1, 1, 1, 1])
  else:
    images_mix_weight = tf.reshape(mix_weight, [batch_size, 1, 1, 1])
  images_mix_weight = tf.cast(images_mix_weight, images.dtype)

  if use_random_shuffling:
    mixup_index = tf.random.shuffle(tf.range(batch_size))
  else:
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    mixup_index = tf.reverse(tf.range(batch_size), axis=[0])

  images_mix = (
      images * images_mix_weight + tf.gather(images, mixup_index) *
      (1. - images_mix_weight))

  mix_weight = tf.cast(mix_weight, labels.dtype)
  if augmix:
    labels = tf.reshape(
        tf.tile(labels, [1, aug_count + 1]), [batch_size, aug_count + 1, -1])
    labels_mix = (
        labels * mix_weight + tf.gather(labels, mixup_index) *
        (1. - mix_weight))
    labels_mix = tf.reshape(
        tf.transpose(labels_mix, [1, 0, 2]), [batch_size * (aug_count + 1), -1])
  else:
    labels_mix = (
        labels * mix_weight + tf.gather(labels, mixup_index) *
        (1. - mix_weight))

  if return_weights:
    return images_mix, labels_mix, mix_weight, mixup_index
  else:
    return images_mix, labels_mix


def adaptive_mixup(batch_size,
                   aug_params,
                   images,
                   labels,
                   return_weights=False):
  """Applies Confidence Adjusted Mixup (CAMixup) regularization.

  [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412

  Arguments:
    batch_size: The input batch size for images and labels.
    aug_params: Dict of data augmentation hyper parameters.
    images: A batch of images of shape [batch_size, ...]
    labels: A batch of labels of shape [batch_size, num_classes]
    return_weights: Whether the mixing weights and ordering of the mixup images
      is returned alongside the new images and labels.

  Returns:
    A tuple of (images, labels) with the same dimensions as the input with
    Mixup regularization applied if return_weights is False, and (images,
    labels, mix_weights, mixup_index) otherwise.
  """
  augmix = aug_params.get('augmix', False)
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
  images_mix_weight = tf.cast(images_mix_weight, images.dtype)
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

  if return_weights:
    # Mixup on a single batch is implemented by taking a weighted sum with the
    # same batch in reverse.
    mixup_index = tf.reverse(tf.range(batch_size), axis=[0])
    return images_mix, labels_mix, mix_weight, mixup_index
  else:
    return images_mix, labels_mix
