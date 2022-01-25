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

r"""Based on https://www.kaggle.com/c/diabetic-retinopathy-detection/data.

New split of the Kaggle Diabetic Retinopathy Detection data, in which we
perform classification on a mild threshold (i.e., {0} are negative
and {1, 2, 3, 4} are positive examples), and set aside examples with underlying
severity label \in {2, 3, 4} as out-of-distribution.

We call this a _severity shift_.

Note that this allows us to use examples with underlying labels \in {2, 3, 4}
 that are listed as 'train' in the tf.data partition in the evaluation set here.
"""
import csv
import os
from typing import Dict, Optional

from absl import logging
import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
from uncertainty_baselines.datasets import base
from uncertainty_baselines.datasets.diabetic_retinopathy_dataset_utils import _btgraham_processing
from uncertainty_baselines.datasets.diabetic_retinopathy_dataset_utils import _resize_image_if_necessary

_CITATION = """\
@ONLINE {kaggle-diabetic-retinopathy,
    author = 'Kaggle and EyePacs',
    title  = 'Kaggle Diabetic Retinopathy Detection',
    month  = 'jul',
    year   = '2015',
    url    = 'https://www.kaggle.com/c/diabetic-retinopathy-detection/data'
}
"""
_URL_TEST_LABELS = (
    'https://storage.googleapis.com/kaggle-forum-message-attachments/'
    '90528/2877/retinopathy_solution.csv')
_BTGRAHAM_DESCRIPTION_PATTERN = (
    'Images have been preprocessed as the winner of the Kaggle competition did '
    'in 2015: first they are resized so that the radius of an eyeball is '
    '{} pixels, then they are cropped to 90% of the radius, and finally they '
    'are encoded with 72 JPEG quality.')
_BLUR_BTGRAHAM_DESCRIPTION_PATTERN = (
    "A variant of the processing method used by the winner of the 2015 Kaggle "
    "competition: images are resized so that the radius of an eyeball is "
    "{} pixels, then receive a Gaussian blur-based normalization with Kernel "
    "standard deviation along the X-axis of {}. Then they are cropped to 90% "
    "of the radius, and finally they are encoded with 72 JPEG quality.")


class DiabeticRetinopathySeverityShiftMildDataset(base.BaseDataset):
  """Kaggle DiabeticRetinopathySeverityShift builder class."""

  def __init__(self,
               split: str,
               builder_config: str = (
                   'diabetic_retinopathy_severity_shift_mild/btgraham-300'),
               shuffle_buffer_size: Optional[int] = None,
               num_parallel_parser_calls: int = 64,
               data_dir: Optional[str] = None,
               download_data: bool = False,
               is_training: Optional[bool] = None,
               drop_remainder: bool = True,
               cache: bool = False):
    """Create a Kaggle diabetic retinopathy detection tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      builder_config: a builder config used by the
        DiabeticRetinopathySeverityShiftMild builder.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      data_dir: optional dir to save TFDS data to. If none then the local
        filesystem is used. Required for using TPUs on Cloud.
      download_data: Whether or not to download data before loading.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
      drop_remainder: Whether or not to drop the remaining partial batch.
      cache: Whether or not to cache the dataset in memory. Can lead to OOM
        errors in host memory.
    """
    if is_training is None:
      is_training = split in ['train', tfds.Split.TRAIN]
    logging.info(
        'Using Severity Shift (Mild decision threshold) builder config %s.',
        builder_config)
    dataset_builder = tfds.builder(builder_config, data_dir=data_dir)
    super(DiabeticRetinopathySeverityShiftMildDataset, self).__init__(
        name='diabetic_retinopathy_severity_shift_mild',
        dataset_builder=dataset_builder,
        split=split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data,
        drop_remainder=drop_remainder,
        cache=cache)
    logging.info(
        'Building Diabetic Retinopathy Severity Shift dataset with mild '
        'decision threshold.')
    if not drop_remainder:
      logging.info(
          'Not dropping the remainder (i.e., not truncating last batch).')

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      """A pre-process function to return images in [0, 1]."""
      image = example['image']
      image = tf.image.convert_image_dtype(image, tf.float32)
      image = tf.image.resize(image, size=(512, 512), method='bilinear')

      # We place the decision threshold between 0 and 1 in the 'mild' case
      highest_negative_class = 0

      # Binarize task.
      label = tf.cast(example['label'] > highest_negative_class, tf.int32)

      parsed_example = {
          'features': image,
          'labels': label,
          'name': example['name'],
      }
      return parsed_example

    return _example_parser


class DiabeticRetinopathySeverityShiftMildConfig(tfds.core.BuilderConfig):
  """BuilderConfig for DiabeticRetinopathySeverityShiftMild."""

  def __init__(self, target_pixels=None, blur_constant=None, **kwargs):
    """BuilderConfig for DiabeticRetinopathySeverityShiftMild.

    Args:
      target_pixels: If given, rescale the images so that the total number of
        pixels is roughly this value.
      blur_constant: Constant used to vary the Kernel standard deviation in
        smoothing the image with Gaussian blur.
      **kwargs: keyword arguments forward to super.
    """
    super(DiabeticRetinopathySeverityShiftMildConfig, self).__init__(
        version=tfds.core.Version('1.0.0'),
        release_notes={
            '1.0.0': 'Initial release',
        },
        **kwargs)
    self._target_pixels = target_pixels
    self._blur_constant = blur_constant

  @property
  def target_pixels(self):
    return self._target_pixels

  @property
  def blur_constant(self):
    return self._blur_constant


class DiabeticRetinopathySeverityShiftMild(tfds.core.GeneratorBasedBuilder):
  """A partitioning of the Kaggle/EyePACS Diabetic Retinopathy Detection.

  Allows for the formation of an out-of-distribution dataset with underlying
  severity labels unseen at training time.
  """

  MANUAL_DOWNLOAD_INSTRUCTIONS = """\
  You have to download this dataset from Kaggle.
  https://www.kaggle.com/c/diabetic-retinopathy-detection/data
  After downloading, unpack the test.zip file into test/ directory in manual_dir
  and sample.zip to sample/. Also unpack the sampleSubmissions.csv and
  trainLabels.csv.
  """

  BUILDER_CONFIGS = [
      DiabeticRetinopathySeverityShiftMildConfig(
          name='original',
          description='Images at their original resolution and quality.'),
      DiabeticRetinopathySeverityShiftMildConfig(
          name='1M',
          description='Images have roughly 1,000,000 pixels, at 72 quality.',
          target_pixels=1000000),
      DiabeticRetinopathySeverityShiftMildConfig(
          name='250K',
          description='Images have roughly 250,000 pixels, at 72 quality.',
          target_pixels=250000),
      DiabeticRetinopathySeverityShiftMildConfig(
          name='btgraham-300',
          description=_BTGRAHAM_DESCRIPTION_PATTERN.format(300),
          target_pixels=300),
      DiabeticRetinopathySeverityShiftMildConfig(
          name='blur-3-btgraham-300',
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 / 3),
          blur_constant=3,
          target_pixels=300),
      DiabeticRetinopathySeverityShiftMildConfig(
          name='blur-5-btgraham-300',
          description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 / 5),
          blur_constant=5,
          target_pixels=300),
      DiabeticRetinopathySeverityShiftMildConfig(
         name='blur-10-btgraham-300',
         description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 // 10),
         blur_constant=10,
         target_pixels=300),
      DiabeticRetinopathySeverityShiftMildConfig(
         name='blur-20-btgraham-300',
         description=_BLUR_BTGRAHAM_DESCRIPTION_PATTERN.format(300, 300 // 20),
         blur_constant=20,
         target_pixels=300)
  ]

  def _info(self):
    return tfds.core.DatasetInfo(
        builder=self,
        description='A large set of high-resolution retina images taken under '
        'a variety of imaging conditions. Partitioned to evaluate '
        'model generalization to clinical severity labels unseen at training '
        'time.',
        features=tfds.features.FeaturesDict({
            'name': tfds.features.Text(),  # patient ID + eye. eg: '4_left'.
            'image': tfds.features.Image(),
            # From 0 (no DR - saine) to 4 (Proliferative DR). -1 means no label.
            'label': tfds.features.ClassLabel(num_classes=5),
        }),
        homepage='https://www.kaggle.com/c/diabetic-retinopathy-detection/data',
        citation=_CITATION,
    )

  def _split_generators(self, dl_manager):
    # TODO(nband): implement download using kaggle API.
    # TODO(nband): implement extraction of multiple files archives.
    path = dl_manager.manual_dir
    test_labels_path = dl_manager.download(_URL_TEST_LABELS)
    if tf.io.gfile.isdir(test_labels_path):
      # While testing: download() returns the dir containing the tests files.
      test_labels_path = os.path.join(test_labels_path,
                                      'retinopathy_solution.csv')

    train_images_path = os.path.join(path, 'train')
    test_images_path = os.path.join(path, 'test')
    train_labels_path = os.path.join(path, 'trainLabels.csv')

    # Each dataset split is specified by a list of subsets of the data.
    # These subsets are specified by:
    # * images_dir_path, location of images
    # * is_in_domain, divides between the in-domain and OOD severity labels
    # * csv_path, location of labels and image names
    # * csv_usage, divides between validation and test
    return {
        'sample':
            self._generate_examples([(os.path.join(path, 'sample'), None, None,
                                      None)]),
        'train':
            self._generate_examples([(train_images_path, True,
                                      train_labels_path, None)]),
        'in_domain_validation':
            self._generate_examples([(test_images_path, True, test_labels_path,
                                      'Public')]),
        'ood_validation':
            self._generate_examples([(test_images_path, False, test_labels_path,
                                      'Public')]),
        'in_domain_test':
            self._generate_examples([(test_images_path, True, test_labels_path,
                                      'Private')]),
        # Note that we can use the OOD examples in the
        # original Kaggle train set below.
        'ood_test':
            self._generate_examples([
                (train_images_path, False, train_labels_path, None),
                (test_images_path, False, test_labels_path, 'Private')
            ]),
    }

  def _generate_examples(self, split_args):
    for args in split_args:
      generator = self._generate_examples_helper(
          images_dir_path=args[0],
          is_in_domain=args[1],
          csv_path=args[2],
          csv_usage=args[3])
      for name_and_record in generator:
        yield name_and_record

  def _generate_examples_helper(self,
                                images_dir_path,
                                is_in_domain: Optional[bool] = None,
                                csv_path=None,
                                csv_usage=None):
    """Yields example instances for a specified dataset split.

    Args:
      images_dir_path: path to dir in which images are stored.
      is_in_domain: Optional[bool], use in-domain examples wrt severity level
        (or OOD, if False)
      csv_path: optional, path to csv file with two columns: name of image and
        label. If not provided, just scan image directory, don't set labels.
      csv_usage: optional, subset of examples from the csv file to use based on
        the 'Usage' column from the csv.
    """
    if is_in_domain is not None and csv_path:
      in_domain_severity_levels = {0, 1}
      ood_severity_levels = {2, 3, 4}
      severity_level_set = (
          in_domain_severity_levels if is_in_domain else ood_severity_levels)

    if csv_path:
      with tf.io.gfile.GFile(csv_path) as csv_f:
        reader = csv.DictReader(csv_f)
        data = []
        for row in reader:
          level = int(row['level'])
          if level not in severity_level_set:
            continue

          if csv_usage is None or row['Usage'] == csv_usage:
            data.append((row['image'], level))
    else:
      data = [(fname[:-5], -1)
              for fname in tf.io.gfile.listdir(images_dir_path)
              if fname.endswith('.jpeg')]

    logging.info('Using BuilderConfig %s.', self.builder_config.name)

    for name, label in data:
      image_filepath = '%s/%s.jpeg' % (images_dir_path, name)
      record = {
          'name': name,
          'image': self._process_image(image_filepath),
          'label': label,
      }
      yield name, record

  def _process_image(self, filepath):
    with tf.io.gfile.GFile(filepath, mode='rb') as image_fobj:
      if self.builder_config.name.startswith('btgraham'):
        return _btgraham_processing(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=self.builder_config.target_pixels,
            blur_constant=30,
            crop_to_radius=True)
      elif self.builder_config.name.startswith("blur"):
        return _btgraham_processing(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            filepath=filepath,
            target_pixels=self.builder_config.target_pixels,
            blur_constant=self.builder_config.blur_constant,
            crop_to_radius=True)
      else:
        return _resize_image_if_necessary(  # pylint: disable=protected-access
            image_fobj=image_fobj,
            target_pixels=self.builder_config.target_pixels)
