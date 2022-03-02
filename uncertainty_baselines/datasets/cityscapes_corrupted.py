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

"""Cityscapes Corrupted builder.

We have an option to use a percent of the training dataset as a validation set,
and treat the original validation set as the test set. This is because the test
set in cityscapes only includes a subset of the classes per image.
"""

import os
import re

import tensorflow as tf
import tensorflow_datasets.public_api as tfds

from tensorflow_datasets.image.cityscapes import _get_left_image_id
from typing import Dict, Optional

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base

_CITATION = """\
@inproceedings{Cordts2016Cityscapes,
  title={The Cityscapes Dataset for Semantic Urban Scene Understanding},
  author={Cordts, Marius and Omran, Mohamed and Ramos, Sebastian and Rehfeld, Timo and Enzweiler, Markus and Benenson, Rodrigo and Franke, Uwe and Roth, Stefan and Schiele, Bernt},
  booktitle={Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2016}
}
@inproceedings{
  hendrycks2018benchmarking,
  title={Benchmarking Neural Network Robustness to Common Corruptions and Perturbations},
  author={Dan Hendrycks and Thomas Dietterich},
  booktitle={International Conference on Learning Representations},
  year={2019},
  url={https://openreview.net/forum?id=HJz6tiCqYm},
}
@article{michaelis2019dragon,
  title={Benchmarking Robustness in Object Detection: 
    Autonomous Driving when Winter is Coming},
  author={Michaelis, Claudio and Mitzkus, Benjamin and 
    Geirhos, Robert and Rusak, Evgenia and 
    Bringmann, Oliver and Ecker, Alexander S. and 
    Bethge, Matthias and Brendel, Wieland},
  journal={arXiv preprint arXiv:1907.07484},
  year={2019}
}
"""

_DESCRIPTION = """\
Cityscapes Corrupted
"""

_DOWNLOAD_URL = "gs://ub-ekb/cityscapes_corrupted/raw_data/v.0.0"


_CORRUPTIONS = [
    'gaussian_noise',
]

class CityscapesCorruptedConfig(tfds.core.BuilderConfig):
  """BuilderConfig for Cityscapes corrupted.
    Args:
      corruption_type (str): name of corruption.
      severity (int): level of corruption.
      right_images (bool): Enables right images for stereo image tasks.
      segmentation_labels (bool): Enables image segmentation labels.
      disparity_maps (bool): Enables disparity maps.
      train_extra_split (bool): Enables train_extra split. This automatically
        enables coarse grain segmentations, if segmentation labels are used.
  """

  def __init__(self,
               *,
               corruption_type,
               severity,
               right_images=False,
               segmentation_labels=True,
               disparity_maps=False,
               train_extra_split=False,
               **kwargs):
    super(CityscapesCorruptedConfig, self).__init__(version='1.0.0', **kwargs)

    self.corruption = corruption_type
    self.severity = severity

    self.right_images = right_images
    self.segmentation_labels = segmentation_labels
    self.disparity_maps = disparity_maps
    self.train_extra_split = train_extra_split

    self.ignored_ids = set()

    # Setup required zips and their root dir names
    self.zip_root = {}
    self.zip_root['images_left'] = ('leftImg8bit_trainvaltest_{}-{}.zip'.format(corruption_type,
                                                                                severity),
                                    'leftImg8bit')

    if self.train_extra_split:
      raise NotImplementedError("train_extra_split")

    if self.right_images:
      raise NotImplementedError("right_images")

    if self.segmentation_labels:
      if not self.train_extra_split:
        self.zip_root['segmentation_labels'] = ('gtFine_trainvaltest.zip',
                                                'gtFine')
        self.label_suffix = 'gtFine_labelIds'
      else:
        # The 'train extra' split only has coarse labels unlike train and val.
        # Therefore, for consistency across splits, we also enable coarse labels
        # using the train_extra_split flag.
        raise NotImplementedError("train_extra_split for segmentation_labels")

    if self.disparity_maps:
      raise NotImplementedError("disparity_maps")


def _make_builder_configs():
  """Construct a list of BuilderConfigs.
  Construct a list of 95 Cifar10CorruptedConfig objects, corresponding to
  the 15 corruption types + 4 extra corruptions and 5 severities.
  Returns:
    A list of CityscapesCorruptedConfig objects.
  """
  config_list = []
  for corruption in _CORRUPTIONS:
    for severity in range(1, 6):
      config_list.append(
          CityscapesCorruptedConfig(
              corruption_type=corruption,
              severity=severity,
              name="semantic_segmentation_{}_{}".format(corruption, str(severity)),
              description='Cityscapes semantic segmentation dataset. Corruption method: ' + corruption +
              ', severity level: ' + str(severity),
              right_images=False,
              segmentation_labels=True,
              disparity_maps=False,
              train_extra_split=False,
          ))
  return config_list


class CityscapesCorrupted(tfds.core.GeneratorBasedBuilder):
  """Base class for Cityscapes datasets."""

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  Download files from _DOWNLOAD_URL and place them in the manual directory
  """

  BUILDER_CONFIGS = _make_builder_configs()
  RELEASE_NOTES = {
      '0.0.0': 'Cityscapes-C corruptions',
  }
  def _info(self):
    # Enable features as necessary
    features = {}
    features['image_id'] = tfds.features.Text()
    features['image_left'] = tfds.features.Image(
        shape=(1024, 2048, 3), encoding_format='png')

    if self.builder_config.right_images:
      raise NotImplementedError("right_images")

    if self.builder_config.segmentation_labels:
      features['segmentation_label'] = tfds.features.Image(
          shape=(1024, 2048, 1), encoding_format='png', use_colormap=True)

    if self.builder_config.disparity_maps:
      raise NotImplementedError("disparity_maps")


    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict(features),
        homepage='https://www.cityscapes-dataset.com',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    paths = {}
    for split, (zip_file, _) in self.builder_config.zip_root.items():
      paths[split] = os.path.join(dl_manager.manual_dir, zip_file)

    if any(not tf.io.gfile.exists(z) for z in paths.values()):
      msg = 'You must download the dataset files manually and place them in: '
      msg += ', '.join(paths.values())
      raise AssertionError(msg)

    for split, (_, zip_root) in self.builder_config.zip_root.items():
      paths[split] = os.path.join(dl_manager.extract(paths[split]), zip_root)

    splits = [
        tfds.core.SplitGenerator(
            name=tfds.Split.VALIDATION,
            gen_kwargs={
                feat_dir: os.path.join(path, 'val')
                for feat_dir, path in paths.items()
                if not feat_dir.endswith('/extra')
            },
        ),
    ]

    return splits

  def _generate_examples(self, **paths):
    left_imgs_root = paths['images_left']
    for city_id in tf.io.gfile.listdir(left_imgs_root):
      paths_city_root = {
          feat_dir: os.path.join(path, city_id)
          for feat_dir, path in paths.items()
      }

      left_city_root = paths_city_root['images_left']
      for left_img in tf.io.gfile.listdir(left_city_root):
        left_img_path = os.path.join(left_city_root, left_img)
        image_id = _get_left_image_id(left_img)

        if image_id in self.builder_config.ignored_ids:
          continue

        features = {
            'image_id': image_id,
            'image_left': left_img_path,
        }

        if self.builder_config.right_images:
          raise NotImplementedError("right_images")

        if self.builder_config.segmentation_labels:
          features['segmentation_label'] = os.path.join(
              paths_city_root['segmentation_labels'],
              '{}_{}.png'.format(image_id, self.builder_config.label_suffix))

        if self.builder_config.disparity_maps:
          raise NotImplementedError("disparity_maps")

        yield image_id, features


class CityscapesCorruptedDataset(base.BaseDataset):
  """Cityscapes dataset builder class."""

  def __init__(
      self,
      corruption_type: str,
      severity: int,
      split: str,
      validation_percent: float = 0.0,
      shuffle_buffer_size: Optional[int] = 1,
      num_parallel_parser_calls: int = 1,
      try_gcs: bool = False,
      download_data: bool = False,
      data_dir: Optional[str] = None,
      is_training: Optional[bool] = None,
      use_bfloat16: bool = False,
      normalize_input: bool = False,
      image_height: int = 1024,
      image_width: int = 2048,
      one_hot: bool = False,
      include_file_name: bool = False,
  ):
    """Create an Cityscapes tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      validation_percent: the percent of the training set to use as a validation
        set.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files.
      download_data: Whether or not to download data before loading.
      data_dir: Directory to read/write data, that is passed to the
        tfds dataset_builder as a data_dir parameter.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      use_bfloat16: Whether or not to use bfloat16 or float32 images.
      normalize_input: Whether or not to normalize images by the ImageNet mean
        and stddev.
      image_height: The height of the image in pixels.
      image_width: The height of the image in pixels.
      one_hot: whether or not to use one-hot labels.
      include_file_name: Whether or not to include a string file_name field in
        each example. Since this field is a string, it is not compatible with
        TPUs.
    """
    name = 'cityscapes_corrupted/semantic_segmentation_{}_{}'.format(corruption_type,str(severity))
    dataset_builder = tfds.builder(name, try_gcs=try_gcs, data_dir=data_dir)
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    new_split = base.get_validation_percent_split(
        dataset_builder,
        validation_percent,
        split,
        test_split=tfds.Split.VALIDATION)

    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=new_split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

    self._use_bfloat16 = use_bfloat16
    self._normalize_input = normalize_input
    self._image_height = image_height
    self._image_width = image_width
    self._one_hot = one_hot
    self._include_file_name = include_file_name

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Create a pre-process function to return images in [0, 1]."""

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """Preprocesses Cityscapes image Tensors."""
      image = example['image_left']

      if self._normalize_input:
        image = tf.cast(image, tf.float32) / 255.
      if self._use_bfloat16:
        image = tf.cast(image, tf.bfloat16)

      if self._one_hot:
        label = tf.one_hot(example['segmentation_label'], 34, dtype=tf.int32)
        label = tf.reshape(
            label, shape=[self._image_height, self._image_width, 34])

      else:
        label = tf.cast(example['segmentation_label'], tf.int32)

      parsed_example = {
          'features': image,
          'labels': label,
      }
      if self._include_file_name and 'file_name' in example:
        parsed_example['file_name'] = example['file_name']
      return parsed_example

    return _example_parser
