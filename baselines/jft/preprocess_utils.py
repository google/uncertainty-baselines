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

"""Preprocessing utilities."""

from collections import abc
import dataclasses
from typing import List, Optional, Tuple, Union

from clu import preprocess_spec
import tensorflow as tf

Features = preprocess_spec.Features


def _maybe_repeat(arg, n_reps):
  if not isinstance(arg, abc.Sequence):
    arg = (arg,) * n_reps
  return arg


def all_ops():
  """Returns all preprocessing ops defined in this module."""
  return preprocess_spec.get_all_ops(__name__)


@dataclasses.dataclass
class Decode:
  """Decodes an encoded image string, see tf.io.decode_image.

  Attributes:
    channels: Number of image channels.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
  """

  channels: int = 3
  key: str = "image"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image_data = features[self.key]
    decoded_image = tf.io.decode_image(
        image_data, channels=self.channels, expand_animations=False)
    features[self.key_result or self.key] = decoded_image
    return features


@dataclasses.dataclass
class Resize:
  """Resizes an image to a given size.

  Attributes:
    resize_size: Either an integer H, where H is both the new height and width
      of the resized image, or a list or tuple [H, W] of integers, where H and W
      are new image's height and width respectively.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
  """

  resize_size: Union[int, Tuple[int, int], List[int]]
  key: str = "image"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    resize_size = _maybe_repeat(self.resize_size, 2)
    resized_image = tf.cast(tf.image.resize(image, resize_size), image.dtype)  # pytype: disable=attribute-error  # allow-recursive-types
    features[self.key_result or self.key] = resized_image
    return features


@dataclasses.dataclass
class ResizeSmall:
  """Resizes the smaller side to `smaller_size` while  keeping the aspect ratio.

  Attributes:
    smaller_size: An integer that represents a new size of the smaller side of
      an input image.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
  """

  smaller_size: int
  key: str = "image"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    h, w = tf.shape(image)[0], tf.shape(image)[1]
    # Figure out the necessary h/w.
    ratio = (
        tf.cast(self.smaller_size, tf.float32) /
        tf.cast(tf.minimum(h, w), tf.float32))
    h = tf.cast(tf.round(tf.cast(h, tf.float32) * ratio), tf.int32)
    w = tf.cast(tf.round(tf.cast(w, tf.float32) * ratio), tf.int32)
    resized_image = tf.image.resize(image, [h, w], method="area")
    features[self.key_result or self.key] = resized_image
    return features


@dataclasses.dataclass
class InceptionCrop:
  """Performs an Inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Attributes:
    resize_size: Optional size to which to resize the image after a crop. Either
      an integer H, where H is both the new height and width of the resized
      image, or a list or tuple [H, W] of integers, where H and W are new
      image's height and width respectively.
    area_min: Minimal crop area.
    area_max: Maximal crop area.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
    rng_key: Key of the random number used for
      `tf.image.stateless_sample_distorted_bounding_box`.
  """

  resize_size: Optional[int] = None
  area_min: int = 5
  area_max: int = 100
  key: str = "image"
  key_result: Optional[str] = None
  rng_key: str = "rng"

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    rng = features[self.rng_key]
    begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
        tf.shape(image),
        tf.zeros([0, 0, 4], tf.float32),
        seed=rng,
        area_range=(self.area_min / 100, self.area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)
    crop = tf.slice(image, begin, size)
    # Unfortunately, the above operation loses the depth-dimension. So we need
    # to restore it the manual way.
    crop.set_shape([None, None, image.shape[-1]])  # pytype: disable=attribute-error  # allow-recursive-types
    if self.resize_size:
      crop = Resize([self.resize_size, self.resize_size])({
          "image": crop
      })["image"]
    features[self.key_result or self.key] = crop
    return features


@dataclasses.dataclass
class DecodeJpegAndInceptionCrop:
  """Performs a JPEG decoding followed by an Inception-style image crop.

  Inception-style crop is a random image crop (its size and aspect ratio are
  random) that was used for training Inception models, see
  https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf.

  Attributes:
    resize_size: Optional size to which to resize the image after a crop. Either
      an integer H, where H is both the new height and width of the resized
      image, or a list or tuple [H, W] of integers, where H and W are new
      image's height and width respectively.
    area_min: Minimal crop area.
    area_max: Maximal crop area.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
    rng_key: Key of the random number used for
      `tf.image.stateless_sample_distorted_bounding_box`.
  """

  resize_size: Optional[int] = None
  area_min: int = 5
  area_max: int = 100
  key: str = "image"
  key_result: Optional[str] = None
  rng_key: str = "rng"

  def __call__(self, features: Features) -> Features:
    image_data = features[self.key]
    rng = features[self.rng_key]
    shape = tf.image.extract_jpeg_shape(image_data)
    begin, size, _ = tf.image.stateless_sample_distorted_bounding_box(
        shape,
        tf.zeros([0, 0, 4], tf.float32),
        seed=rng,
        area_range=(self.area_min / 100, self.area_max / 100),
        min_object_covered=0,  # Don't enforce a minimum area.
        use_image_if_no_bounding_boxes=True)

    # Crop the image to the specified bounding box.
    offset_y, offset_x, _ = tf.unstack(begin)
    target_height, target_width, _ = tf.unstack(size)
    crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
    image = tf.image.decode_and_crop_jpeg(image_data, crop_window, channels=3)

    if self.resize_size:
      image = Resize([self.resize_size, self.resize_size])({
          "image": image
      })["image"]

    features[self.key_result or self.key] = image
    return features


@dataclasses.dataclass
class RandomCrop:
  """Performs a random crop of a given size.

  Attributes:
    crop_size: Either an integer H, where H is both the height and width of the
      random crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the random crop respectively.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
    rng_key: Key of the random number used for
      `tf.image.stateless_sample_distorted_bounding_box`.
  """

  crop_size: int
  key: str = "image"
  key_result: Optional[str] = None
  rng_key: str = "rng"

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    rng = features[self.rng_key]
    crop_size = _maybe_repeat(self.crop_size, 2)
    cropped_image = tf.image.stateless_random_crop(
        image, [crop_size[0], crop_size[1], image.shape[-1]], seed=rng)  # pytype: disable=attribute-error  # allow-recursive-types
    features[self.key_result or self.key] = cropped_image
    return features


@dataclasses.dataclass
class CentralCrop:
  """Performs a central crop of a given size.

  Attributes:
    crop_size: Either an integer H, where H is both the height and width of the
      central crop, or a list or tuple [H, W] of integers, where H and W are
      height and width of the central crop respectively.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
  """

  crop_size: int
  key: str = "image"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    crop_size = _maybe_repeat(self.crop_size, 2)
    h, w = crop_size[0], crop_size[1]
    dy = (tf.shape(image)[0] - h) // 2
    dx = (tf.shape(image)[1] - w) // 2
    cropped_image = tf.image.crop_to_bounding_box(image, dy, dx, h, w)
    features[self.key_result or self.key] = cropped_image
    return features


@dataclasses.dataclass
class FlipLr:
  """Flips an image horizontally with probability 50%.

  Attributes:
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
    rng_key: Key of the random number used for
      `tf.image.stateless_sample_distorted_bounding_box`.
  """

  key: str = "image"
  key_result: Optional[str] = None
  rng_key: str = "rng"

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    rng = features[self.rng_key]
    flipped_image = tf.image.stateless_random_flip_left_right(image, seed=rng)
    features[self.key_result or self.key] = flipped_image
    return features


@dataclasses.dataclass
class ValueRange:
  """Transforms a [in_min, in_max] image to [vmin, vmax] range.

  Input ranges in_min/in_max can be equal-size lists to rescale the invidudal
  channels independently.

  Attributes:
    vmin: A scalar. Output max value.
    vmax: A scalar. Output min value.
    in_min: A scalar or a list of input min values to scale. If a list, the
      length should match to the number of channels in the image.
    in_max: A scalar or a list of input max values to scale. If a list, the
      length should match to the number of channels in the image.
    clip_values: Whether to clip the output values to the provided ranges.
    key: Key of the data to be processed.
    key_result: Key under which to store the result (same as `key` if None).
  """

  vmin: float = -1
  vmax: float = 1
  in_min: float = 0
  in_max: float = 255.0
  clip_values: bool = False
  key: str = "image"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    image = features[self.key]
    in_min_t = tf.constant(self.in_min, tf.float32)
    in_max_t = tf.constant(self.in_max, tf.float32)
    image = tf.cast(image, tf.float32)
    image = (image - in_min_t) / (in_max_t - in_min_t)
    image = self.vmin + image * (self.vmax - self.vmin)
    if self.clip_values:
      image = tf.clip_by_value(image, self.vmin, self.vmax)
    features[self.key_result or self.key] = image
    return features


@dataclasses.dataclass
class Onehot:
  """One-hot encodes the input.

  Attributes:
    depth: Length of the one-hot vector (how many classes).
    multi: If there are multiple labels, whether to merge them into the same
      "multi-hot" vector (True) or keep them as an extra dimension (False).
    on: Value to fill in for the positive label (default: 1).
    off: Value to fill in for negative labels (default: 0).
    key: Key of the data to be one-hot encoded.
    key_result: Key under which to store the result (same as `key` if None).
  """

  depth: int
  multi: bool = True
  on: float = 1.0
  off: float = 0.0
  key: str = "labels"
  key_result: Optional[str] = None

  def __call__(self, features: Features) -> Features:
    # When there's more than one label, this is significantly more efficient
    # than using tf.one_hot followed by tf.reduce_max; we tested.
    labels = features[self.key]
    if labels.shape.rank > 0 and self.multi:  # pytype: disable=attribute-error  # allow-recursive-types
      x = tf.scatter_nd(labels[:, None], tf.ones(tf.shape(labels)[0]),
                        (self.depth,))
      x = tf.clip_by_value(x, 0, 1) * (self.on - self.off) + self.off
    else:
      x = tf.one_hot(labels, self.depth, on_value=self.on, off_value=self.off)
    features[self.key_result or self.key] = x
    return features


@dataclasses.dataclass
class Keep:
  """Keeps only the given keys.

  Attributes:
    keys: List of string keys to keep.
  """

  keys: List[str]

  def __call__(self, features: Features) -> Features:
    return {k: v for k, v in features.items() if k in self.keys}
