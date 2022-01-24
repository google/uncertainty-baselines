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

# Copyright 2021 The TensorFlow Datasets Authors.
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
"""https://www.kaggle.com/c/diabetic-retinopathy-detection/data."""

import io

from absl import logging
import numpy as np
import tensorflow_datasets as tfds

_CITATION = """\
@ONLINE {kaggle-diabetic-retinopathy,
    author = "Kaggle and EyePacs",
    title  = "Kaggle Diabetic Retinopathy Detection",
    month  = "jul",
    year   = "2015",
    url    = "https://www.kaggle.com/c/diabetic-retinopathy-detection/data"
}
"""
_URL_TEST_LABELS = (
    "https://storage.googleapis.com/kaggle-forum-message-attachments/"
    "90528/2877/retinopathy_solution.csv")
_BTGRAHAM_DESCRIPTION_PATTERN = (
    "Images have been preprocessed as the winner of the Kaggle competition did "
    "in 2015: first they are resized so that the radius of an eyeball is "
    "{} pixels, then they are cropped to 90% of the radius, and finally they "
    "are encoded with 72 JPEG quality.")
_BLUR_BTGRAHAM_DESCRIPTION_PATTERN = (
    "A variant of the processing method used by the winner of the 2015 Kaggle "
    "competition: images are resized so that the radius of an eyeball is "
    "{} pixels, then receive a Gaussian blur-based normalization with Kernel "
    "standard deviation along the X-axis of {}. Then they are cropped to 90% "
    "of the radius, and finally they are encoded with 72 JPEG quality.")


def _resize_image_if_necessary(image_fobj, target_pixels=None):
  """Resize an image to have (roughly) the given number of target pixels.

  Args:
    image_fobj: File object containing the original image.
    target_pixels: If given, number of pixels that the image must have.

  Returns:
    A file object.
  """
  if target_pixels is None:
    return image_fobj

  cv2 = tfds.core.lazy_imports.cv2
  # Decode image using OpenCV2.
  image = cv2.imdecode(
      np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
  # Get image height and width.
  height, width, _ = image.shape
  actual_pixels = height * width
  if actual_pixels > target_pixels:
    factor = np.sqrt(target_pixels / actual_pixels)
    image = cv2.resize(image, dsize=None, fx=factor, fy=factor)
  # Encode the image with quality=72 and store it in a BytesIO object.
  _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
  return io.BytesIO(buff.tobytes())


def _btgraham_processing(image_fobj,
                         filepath,
                         target_pixels,
                         blur_constant=30,
                         crop_to_radius=False):
  """Process an image as the winner of the 2015 Kaggle competition.

  Args:
    image_fobj: File object containing the original image.
    filepath: Filepath of the image, for logging purposes only.
    target_pixels: The number of target pixels for the radius of the image.
    blur_constant: Constant used to vary the Kernel standard deviation in
      smoothing the image with Gaussian blur.
    crop_to_radius: If True, crop the borders of the image to remove gray areas.

  Returns:
    A file object.
  """
  cv2 = tfds.core.lazy_imports.cv2
  # Decode image using OpenCV2.
  image = cv2.imdecode(
      np.frombuffer(image_fobj.read(), dtype=np.uint8), flags=3)
  # Process the image.
  image = _scale_radius_size(image, filepath, target_radius_size=target_pixels)
  image = _subtract_local_average(
      image, target_radius_size=target_pixels, blur_constant=blur_constant)
  image = _mask_and_crop_to_radius(
      image,
      target_radius_size=target_pixels,
      radius_mask_ratio=0.9,
      crop_to_radius=crop_to_radius)
  # Encode the image with quality=72 and store it in a BytesIO object.
  _, buff = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
  return io.BytesIO(buff.tobytes())


def _scale_radius_size(image, filepath, target_radius_size):
  """Scale the input image so that the radius of the eyeball is the given."""
  cv2 = tfds.core.lazy_imports.cv2
  x = image[image.shape[0] // 2, :, :].sum(axis=1)
  r = (x > x.mean() / 10.0).sum() / 2.0
  if r < 1.0:
    # Some images in the dataset are corrupted, causing the radius heuristic to
    # fail. In these cases, just assume that the radius is the height of the
    # original image.
    logging.info("Radius of image \"%s\" could not be determined.", filepath)
    r = image.shape[0] / 2.0
  s = target_radius_size / r
  return cv2.resize(image, dsize=None, fx=s, fy=s)


def _subtract_local_average(image, target_radius_size, blur_constant=30):
  cv2 = tfds.core.lazy_imports.cv2
  image_blurred = cv2.GaussianBlur(image, (0, 0),
                                   target_radius_size / blur_constant)
  image = cv2.addWeighted(image, 4, image_blurred, -4, 128)
  return image


def _mask_and_crop_to_radius(image,
                             target_radius_size,
                             radius_mask_ratio=0.9,
                             crop_to_radius=False):
  """Mask and crop image to the given radius ratio."""
  cv2 = tfds.core.lazy_imports.cv2
  mask = np.zeros(image.shape)
  center = (image.shape[1] // 2, image.shape[0] // 2)
  radius = int(target_radius_size * radius_mask_ratio)
  cv2.circle(mask, center=center, radius=radius, color=(1, 1, 1), thickness=-1)
  image = image * mask + (1 - mask) * 128
  if crop_to_radius:
    x_max = min(image.shape[1] // 2 + radius, image.shape[1])
    x_min = max(image.shape[1] // 2 - radius, 0)
    y_max = min(image.shape[0] // 2 + radius, image.shape[0])
    y_min = max(image.shape[0] // 2 - radius, 0)
    image = image[y_min:y_max, x_min:x_max, :]
  return image
