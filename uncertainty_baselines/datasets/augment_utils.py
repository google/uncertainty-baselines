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

"""RandAugment policies for enhanced image preprocessing."""

import math
import tensorflow as tf

# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.keras.layers.preprocessing import image_preprocessing as image_ops
# pylint:enable=g-direct-tensorflow-import

# This signifies the max integer that the controller RNN could predict for the
# augmentation scheme.
MAX_LEVEL = 10.


def to_4d(image):
  """Converts an input Tensor to 4 dimensions.

  4D image => [N, H, W, C] or [N, C, H, W]
  3D image => [1, H, W, C] or [1, C, H, W]
  2D image => [1, H, W, 1]

  Args:
    image: The 2/3/4D input tensor.

  Returns:
    A 4D image tensor.

  Raises:
    `TypeError` if `image` is not a 2/3/4D tensor.

  """
  shape = tf.shape(image)
  original_rank = tf.rank(image)
  left_pad = tf.cast(tf.less_equal(original_rank, 3), dtype=tf.int32)
  right_pad = tf.cast(tf.equal(original_rank, 2), dtype=tf.int32)
  new_shape = tf.concat(
      [
          tf.ones(shape=left_pad, dtype=tf.int32),
          shape,
          tf.ones(shape=right_pad, dtype=tf.int32),
      ],
      axis=0,
  )
  return tf.reshape(image, new_shape)


def from_4d(image, ndims):
  """Converts a 4D image back to `ndims` rank."""
  shape = tf.shape(image)
  begin = tf.cast(tf.less_equal(ndims, 3), dtype=tf.int32)
  end = 4 - tf.cast(tf.equal(ndims, 2), dtype=tf.int32)
  new_shape = shape[begin:end]
  return tf.reshape(image, new_shape)


def _convert_translation_to_transform(translations):
  """Converts translations to a projective transform.

  The translation matrix looks like this:
    [[1 0 -dx]
     [0 1 -dy]
     [0 0 1]]

  Args:
    translations: The 2-element list representing [dx, dy], or a matrix of
      2-element lists representing [dx dy] to translate for each image. The
      shape must be static.

  Returns:
    The transformation matrix of shape (num_images, 8).

  Raises:
    `TypeError` if
      - the shape of `translations` is not known or
      - the shape of `translations` is not rank 1 or 2.

  """
  translations = tf.convert_to_tensor(translations, dtype=tf.float32)
  if translations.get_shape().ndims is None:
    raise TypeError('translations rank must be statically known')
  elif len(translations.get_shape()) == 1:
    translations = translations[None]
  elif len(translations.get_shape()) != 2:
    raise TypeError('translations should have rank 1 or 2.')
  num_translations = tf.shape(translations)[0]

  return tf.concat(
      values=[
          tf.ones((num_translations, 1), tf.dtypes.float32),
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          -translations[:, 0, None],
          tf.zeros((num_translations, 1), tf.dtypes.float32),
          tf.ones((num_translations, 1), tf.dtypes.float32),
          -translations[:, 1, None],
          tf.zeros((num_translations, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def _convert_angles_to_transform(angles, image_width, image_height):
  """Converts an angle or angles to a projective transform.

  Args:
    angles: A scalar to rotate all images, or a vector to rotate a batch of
      images. This must be a scalar.
    image_width: The width of the image(s) to be transformed.
    image_height: The height of the image(s) to be transformed.

  Returns:
    A tensor of shape (num_images, 8).

  Raises:
    `TypeError` if `angles` is not rank 0 or 1.

  """
  angles = tf.convert_to_tensor(angles, dtype=tf.float32)
  if len(angles.get_shape()) == 0:  # pylint:disable=g-explicit-length-test
    angles = angles[None]
  elif len(angles.get_shape()) != 1:
    raise TypeError('Angles should have a rank 0 or 1.')
  x_offset = ((image_width - 1) -
              (tf.math.cos(angles) * (image_width - 1) - tf.math.sin(angles) *
               (image_height - 1))) / 2.0
  y_offset = ((image_height - 1) -
              (tf.math.sin(angles) * (image_width - 1) + tf.math.cos(angles) *
               (image_height - 1))) / 2.0
  num_angles = tf.shape(angles)[0]
  return tf.concat(
      values=[
          tf.math.cos(angles)[:, None],
          -tf.math.sin(angles)[:, None],
          x_offset[:, None],
          tf.math.sin(angles)[:, None],
          tf.math.cos(angles)[:, None],
          y_offset[:, None],
          tf.zeros((num_angles, 2), tf.dtypes.float32),
      ],
      axis=1,
  )


def transform(image, transforms):
  """Prepares input data for `image_ops.transform`."""
  original_ndims = tf.rank(image)
  transforms = tf.convert_to_tensor(transforms, dtype=tf.float32)
  if len(tf.shape(transforms)) == 1:
    transforms = transforms[None]
  image = to_4d(image)
  image = image_ops.transform(
      images=image,
      transforms=transforms,
      interpolation='nearest')
  return from_4d(image, original_ndims)


def translate(image, translations):
  """Translates image(s) by provided vectors.

  Args:
    image: An image Tensor of type uint8.
    translations: A vector or matrix representing [dx dy].

  Returns:
    The translated version of the image.

  """
  transforms = _convert_translation_to_transform(translations)
  return transform(image, transforms=transforms)


def rotate(image, degrees):
  """Rotates the image by degrees either clockwise or counterclockwise.

  Args:
    image: An image Tensor of type uint8.
    degrees: Float, a scalar angle in degrees to rotate all images by. If
      degrees is positive the image will be rotated clockwise otherwise it will
      be rotated counterclockwise.

  Returns:
    The rotated version of image.

  """
  # Convert from degrees to radians.
  degrees_to_radians = math.pi / 180.0
  radians = degrees * degrees_to_radians

  original_ndims = tf.rank(image)
  image = to_4d(image)

  image_height = tf.cast(tf.shape(image)[1], tf.float32)
  image_width = tf.cast(tf.shape(image)[2], tf.float32)
  transforms = _convert_angles_to_transform(angles=radians,
                                            image_width=image_width,
                                            image_height=image_height)
  # In practice, we should randomize the rotation degrees by flipping
  # it negatively half the time, but that's done on 'degrees' outside
  # of the function.
  image = transform(image, transforms=transforms)
  return from_4d(image, original_ndims)


def blend(image1, image2, factor):
  """Blend image1 and image2 using 'factor'.

  Factor can be above 0.0.  A value of 0.0 means only image1 is used.
  A value of 1.0 means only image2 is used.  A value between 0.0 and
  1.0 means we linearly interpolate the pixel values between the two
  images.  A value greater than 1.0 "extrapolates" the difference
  between the two pixel values, and we clip the results to values
  between 0 and 255.

  Args:
    image1: An image Tensor of type uint8.
    image2: An image Tensor of type uint8.
    factor: A floating point value above 0.0.

  Returns:
    A blended image Tensor of type uint8.
  """
  if factor == 0.0:
    return tf.convert_to_tensor(image1)
  if factor == 1.0:
    return tf.convert_to_tensor(image2)

  image1 = tf.cast(image1, tf.float32)
  image2 = tf.cast(image2, tf.float32)

  difference = image2 - image1
  scaled = factor * difference

  # Do addition in float.
  temp = tf.cast(image1, tf.float32) + scaled

  # Interpolate
  if factor > 0.0 and factor < 1.0:
    # Interpolation means we always stay within 0 and 255.
    return tf.cast(temp, tf.uint8)

  # Extrapolate:
  #
  # We need to clip and then cast.
  return tf.cast(tf.clip_by_value(temp, 0.0, 255.0), tf.uint8)


def solarize(image, threshold=128):
  # For each pixel in the image, select the pixel
  # if the value is less than the threshold.
  # Otherwise, subtract 255 from the pixel.
  return tf.where(image < threshold, image, 255 - image)


def color(image, factor):
  """Equivalent of PIL Color."""
  degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
  return blend(degenerate, image, factor)


def posterize(image, bits):
  """Equivalent of PIL Posterize."""
  shift = 8 - bits
  return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)


def wrapped_rotate(image, degrees, replace):
  """Applies rotation with wrap/unwrap."""
  image = rotate(wrap(image), degrees=degrees)
  return unwrap(image, replace)


def translate_x(image, pixels, replace):
  """Equivalent of PIL Translate in X dimension."""
  image = translate(wrap(image), [-pixels, 0])
  return unwrap(image, replace)


def translate_y(image, pixels, replace):
  """Equivalent of PIL Translate in Y dimension."""
  image = translate(wrap(image), [0, -pixels])
  return unwrap(image, replace)


def shear_x(image, level, replace):
  """Equivalent of PIL Shearing in X dimension."""
  # Shear parallel to x axis is a projective transform
  # with a matrix form of:
  # [1  level
  #  0  1].
  image = transform(image=wrap(image),
                    transforms=[1., level, 0., 0., 1., 0., 0., 0.])
  return unwrap(image, replace)


def shear_y(image, level, replace):
  """Equivalent of PIL Shearing in Y dimension."""
  # Shear parallel to y axis is a projective transform
  # with a matrix form of:
  # [1  0
  #  level  1].
  image = transform(image=wrap(image),
                    transforms=[1., 0., 0., level, 1., 0., 0., 0.])
  return unwrap(image, replace)


def autocontrast(image):
  """Implements Autocontrast function from PIL using TF ops.

  Args:
    image: A 3D uint8 tensor.

  Returns:
    The image after it has had autocontrast applied to it and will be of type
    uint8.
  """

  def scale_channel(image):
    """Scale the 2D image using the autocontrast rule."""
    # A possibly cheaper version can be done using cumsum/unique_with_counts
    # over the histogram values, rather than iterating over the entire image.
    # to compute mins and maxes.
    lo = tf.cast(tf.reduce_min(image), tf.float32)
    hi = tf.cast(tf.reduce_max(image), tf.float32)

    # Scale the image, making the lowest value 0 and the highest value 255.
    def scale_values(im):
      scale = 255.0 / (hi - lo)
      offset = -lo * scale
      im = tf.cast(im, tf.float32) * scale + offset
      im = tf.clip_by_value(im, 0.0, 255.0)
      return tf.cast(im, tf.uint8)

    result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
    return result

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image[:, :, 0])
  s2 = scale_channel(image[:, :, 1])
  s3 = scale_channel(image[:, :, 2])
  image = tf.stack([s1, s2, s3], 2)
  return image


def equalize(image):
  """Implements Equalize function from PIL using TF ops."""
  def scale_channel(im, c):
    """Scale the data in the channel to implement equalize."""
    im = tf.cast(im[:, :, c], tf.int32)
    # Compute the histogram of the image channel.
    histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

    # For the purposes of computing the step, filter out the nonzeros.
    nonzero = tf.where(tf.not_equal(histo, 0))
    nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
    step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

    def build_lut(histo, step):
      # Compute the cumulative sum, shifting by step // 2
      # and then normalization by step.
      lut = (tf.cumsum(histo) + (step // 2)) // step
      # Shift lut, prepending with 0.
      lut = tf.concat([[0], lut[:-1]], 0)
      # Clip the counts to be in range.  This is done
      # in the C code for image.point.
      return tf.clip_by_value(lut, 0, 255)

    # If step is zero, return the original image.  Otherwise, build
    # lut from the full histogram and step and then index from it.
    result = tf.cond(tf.equal(step, 0),
                     lambda: im,
                     lambda: tf.gather(build_lut(histo, step), im))

    return tf.cast(result, tf.uint8)

  # Assumes RGB for now.  Scales each channel independently
  # and then stacks the result.
  s1 = scale_channel(image, 0)
  s2 = scale_channel(image, 1)
  s3 = scale_channel(image, 2)
  image = tf.stack([s1, s2, s3], 2)
  return image


def wrap(image):
  """Returns 'image' with an extra channel set to all 1s."""
  shape = tf.shape(image)
  extended_channel = tf.ones([shape[0], shape[1], 1], image.dtype)
  extended = tf.concat([image, extended_channel], axis=2)
  return extended


def unwrap(image, replace):
  """Unwraps an image produced by wrap.

  Where there is a 0 in the last channel for every spatial position,
  the rest of the three channels in that spatial dimension are grayed
  (set to 128).  Operations like translate and shear on a wrapped
  Tensor will leave 0s in empty locations.  Some transformations look
  at the intensity of values to do preprocessing, and we want these
  empty pixels to assume the 'average' value, rather than pure black.


  Args:
    image: A 3D Image Tensor with 4 channels.
    replace: A one or three value 1D tensor to fill empty pixels.

  Returns:
    image: A 3D image Tensor with 3 channels.
  """
  image_shape = tf.shape(image)
  # Flatten the spatial dimensions.
  flattened_image = tf.reshape(image, [-1, image_shape[2]])

  # Find all pixels where the last channel is zero.
  alpha_channel = tf.expand_dims(flattened_image[:, 3], axis=-1)

  replace = tf.concat([replace, tf.ones([1], image.dtype)], 0)

  # Where they are zero, fill them in with 'replace'.
  flattened_image = tf.where(
      tf.equal(alpha_channel, 0),
      tf.ones_like(flattened_image, dtype=image.dtype) * replace,
      flattened_image)

  image = tf.reshape(flattened_image, image_shape)
  image = tf.slice(image, [0, 0, 0], [image_shape[0], image_shape[1], 3])
  return image


def _randomly_negate_tensor(tensor, seed):
  """With 50% prob turn the tensor negative."""
  should_flip = tf.cast(tf.floor(tf.random.stateless_uniform([], seed) + 0.5),
                        tf.bool)
  final_tensor = tf.cond(should_flip, lambda: tensor, lambda: -tensor)
  return final_tensor


def _rotate_level_to_arg(level, seed, randomly_negate_level=True):
  level = (level / MAX_LEVEL) * 30.
  if randomly_negate_level:
    level = _randomly_negate_tensor(level, seed)
  return (level,)


def _shrink_level_to_arg(level):
  """Converts level to ratio by which we shrink the image content."""
  if level == 0:
    return (1.0,)  # if level is zero, do not shrink the image
  # Maximum shrinking ratio is 2.9.
  level = 2. / (MAX_LEVEL / level) + 0.9
  return (level,)


def _enhance_level_to_arg(level):
  return ((level / MAX_LEVEL) * 1.8 + 0.1,)


def _shear_level_to_arg(level, seed, randomly_negate_level=True):
  level = (level / MAX_LEVEL) * 0.3
  if randomly_negate_level:
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, seed)
  return (level,)


def _translate_level_to_arg(level, translate_const,
                            seed, randomly_negate_level=True):
  level = (level / MAX_LEVEL) * float(translate_const)
  if randomly_negate_level:
    # Flip level to negative with 50% chance.
    level = _randomly_negate_tensor(level, seed)
  return (level,)


def _mult_to_arg(level, multiplier=1.):
  return (int((level / MAX_LEVEL) * multiplier),)


def level_to_arg(translate_const, seed, randomly_negate_level=True):
  """Creates a dict mapping image operation names to their arguments."""

  no_arg = lambda level: ()
  posterize_arg = lambda level: _mult_to_arg(level, 4)
  solarize_arg = lambda level: _mult_to_arg(level, 256)
  # pylint: disable=g-long-lambda
  rotate_arg = lambda level: _rotate_level_to_arg(
      level, seed, randomly_negate_level=randomly_negate_level)
  shear_arg = lambda level: _shear_level_to_arg(
      level, seed, randomly_negate_level=randomly_negate_level)
  translate_arg = lambda level: _translate_level_to_arg(
      level, translate_const, seed, randomly_negate_level=randomly_negate_level)
  # pylint: enable=g-long-lambda

  args = {
      'AutoContrast': no_arg,
      'Equalize': no_arg,
      'Rotate': rotate_arg,
      'Posterize': posterize_arg,
      'Solarize': solarize_arg,
      'Color': _enhance_level_to_arg,
      'ShearX': shear_arg,
      'ShearY': shear_arg,
      'TranslateX': translate_arg,
      'TranslateY': translate_arg,
  }
  return args


NAME_TO_FUNC = {
    'AutoContrast': autocontrast,
    'Equalize': equalize,
    'Rotate': wrapped_rotate,
    'Posterize': posterize,
    'Solarize': solarize,
    'Color': color,
    'ShearX': shear_x,
    'ShearY': shear_y,
    'TranslateX': translate_x,
    'TranslateY': translate_y,
}

# Functions that have a 'replace' parameter
REPLACE_FUNCS = frozenset({
    'Rotate',
    'TranslateX',
    'ShearX',
    'ShearY',
    'TranslateY',
})


def _parse_policy_info(name,
                       prob,
                       level,
                       replace_value,
                       translate_const,
                       seed,
                       randomly_negate_level=True):
  """Return the function that corresponds to `name` and update `level` param."""
  func = NAME_TO_FUNC[name]
  # Pass in the seed to level_to_arg for the _randomly_negate_tensor calls.
  args = level_to_arg(
      translate_const, seed, randomly_negate_level=randomly_negate_level)[name](
          level)

  if name in REPLACE_FUNCS:
    # Add in replace arg if it is required for the function that is called.
    args = tuple(list(args) + [replace_value])

  return func, prob, args


class RandAugment(object):
  """Applies the RandAugment policy to images.

  RandAugment is from the paper https://arxiv.org/abs/1909.13719,
  """

  def __init__(self,
               num_layers=1,
               magnitude=10,
               translate_const=100,
               randomly_negate_level=True):
    """Applies the RandAugment policy to images.

    Args:
      num_layers: Integer, the number of augmentation transformations to apply
        sequentially to an image. Represented as (N) in the paper. Usually best
        values will be in the range [1, 3].
      magnitude: Integer, shared magnitude across all augmentation operations.
        Represented as (M) in the paper. Usually best values are in the range
        [5, 10].
      translate_const: multiplier for applying translation.
      randomly_negate_level: whether to randomly flip the sign of levels used
        for rotation, translation, and shear augmentations.
    """
    self.num_layers = num_layers
    self.magnitude = float(magnitude)
    self.translate_const = float(translate_const)
    self.available_ops = [
        'AutoContrast', 'Equalize', 'Rotate', 'Posterize', 'Solarize',
        'Color', 'ShearX', 'ShearY', 'TranslateX', 'TranslateY',
    ]
    self.randomly_negate_level = randomly_negate_level

  def distort(self, image, seed=None):
    """Applies the RandAugment policy to `image`.

    Args:
      image: `Tensor` of shape [height, width, 3] representing an image.
      seed: An optional seed to control the deterministic nature of
        augmentation.

    Returns:
      The augmented version of `image`.
    """
    if seed is None:
      seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)
    # Generate stateless seeds for each random sampling in the code.
    # Number of seeds = self.num_layers * self.available_ops + self.num_layers.
    # seeds_for_layers is (self.num_layers, 2).
    # seeds_for_ops is (self.num_layers * self.available_ops, 2)
    seeds = tf.random.experimental.stateless_split(
        seed, num=self.num_layers + self.num_layers * len(self.available_ops))
    seeds_for_layers = seeds[:self.num_layers]
    seeds_for_ops = seeds[self.num_layers:]

    input_image_type = image.dtype

    if input_image_type != tf.uint8:
      image = tf.clip_by_value(image, 0.0, 255.0)
      image = tf.cast(image, dtype=tf.uint8)

    replace_value = [128] * 3
    min_prob, max_prob = 0.2, 0.8

    for n in range(self.num_layers):
      op_to_select = tf.random.stateless_uniform(
          [], seeds_for_layers[n], maxval=len(self.available_ops) + 1,
          dtype=tf.int32)

      branch_fns = []
      for (i, op_name) in enumerate(self.available_ops):
        prob = tf.random.stateless_uniform(
            [],
            seeds_for_ops[n * len(self.available_ops) + i],
            minval=min_prob,
            maxval=max_prob,
            dtype=tf.float32)
        seed_for_random_negation = tf.random.experimental.stateless_fold_in(
            seeds_for_ops[n * len(self.available_ops) + i],
            n * len(self.available_ops) + i)
        func, _, args = _parse_policy_info(op_name,
                                           prob,
                                           self.magnitude,
                                           replace_value,
                                           self.translate_const,
                                           seed_for_random_negation,
                                           self.randomly_negate_level)
        branch_fns.append((
            i,
            # pylint:disable=g-long-lambda
            lambda selected_func=func, selected_args=args: selected_func(
                image, *selected_args)))
            # pylint:enable=g-long-lambda

      image = tf.switch_case(branch_index=op_to_select,
                             branch_fns=branch_fns,
                             default=lambda: tf.identity(image))

    image = tf.cast(image, dtype=input_image_type)
    return image
