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

"""Library of dataloaders to use in Introspective Active Sampling.

This file contains a library of dataloaders that return three features for each
example: example_id, input feature, and label. The example_id is a unique ID
that will be used to keep track of the bias label for that example. The input
feature will vary depending on the type of data (feature vector, image, etc.),
and the label is specific to the main task.
"""

import dataclasses
import json
import os
from typing import Any, Dict, Iterator, Optional, Tuple, List, Union

import tensorflow as tf
import tensorflow_datasets as tfds


DATASET_REGISTRY = {}
DATA_DIR = '/tmp/data'
_DEFAULT_PATH_CARDIOTOX_TRAIN_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TRAIN_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_VALIDATION_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_VALIDATION_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_TEST_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TEST_LABEL = ''
_DEFAULT_PATH_CARDIOTOX_TEST2_FEATURE = ''
_DEFAULT_PATH_CARDIOTOX_TEST2_LABEL = ''
_WATERBIRDS_DATA_DIR = ''
_WATERBIRDS_TRAIN_PATTERN = ''
_WATERBIRDS_VALIDATION_PATTERN = ''
# Smaller subsample for testing.
_WATERBIRDS_TRAIN_SAMPLE_PATTERN = ''
_WATERBIRDS_TEST_PATTERN = ''

RESNET_IMAGE_SIZE = 224
CROP_PADDING = 32


def register_dataset(name: str):
  """Provides decorator to register functions that return dataset."""
  def save(dataset_builder):
    DATASET_REGISTRY[name] = dataset_builder
    return dataset_builder

  return save


def get_dataset(name: str):
  """Retrieves dataset based on name."""
  if name not in DATASET_REGISTRY:
    raise ValueError(
        f'Unknown dataset: {name}\nPossible choices: {DATASET_REGISTRY.keys()}')
  return DATASET_REGISTRY[name]


@dataclasses.dataclass
class Dataloader:
  train_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  val_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  train_ds: tf.data.Dataset  # Dataset with all the train splits combined.
  train_sample_ds: Optional[tf.data.Dataset] = None  # Subsample of train set.
  eval_ds: Optional[Dict[
      str,
      tf.data.Dataset]] = None  # Validation and any additional test datasets.


def apply_batch(dataloader, batch_size):
  """Apply batching to dataloader."""
  dataloader.train_splits = [
      data.batch(batch_size) for data in dataloader.train_splits]
  dataloader.val_splits = [
      data.batch(batch_size) for data in dataloader.val_splits]
  num_splits = len(dataloader.train_splits)
  train_ds = gather_data_splits(list(range(num_splits)),
                                dataloader.train_splits)
  val_ds = gather_data_splits(list(range(num_splits)),
                              dataloader.val_splits)
  dataloader.train_ds = train_ds
  dataloader.eval_ds['val'] = val_ds
  for (k, v) in dataloader.eval_ds.items():
    if k != 'val':
      dataloader.eval_ds[k] = v.batch(batch_size)
  return dataloader


def gather_data_splits(
    slice_idx: List[int],
    dataset: Union[tf.data.Dataset, List[tf.data.Dataset]]) -> tf.data.Dataset:
  """Gathers slices of a split dataset based on passed indices."""
  data_slice = dataset[slice_idx[0]]
  for idx in slice_idx[1:]:
    data_slice = data_slice.concatenate(dataset[idx])
  return data_slice


def get_train_ids(dataloader: Dataloader):
  # Get example ids used for training
  ids_train_list = list(
      dataloader.train_ds.map(
          lambda feats, label, example_id: example_id).as_numpy_iterator())
  ids_train = []
  for ids in ids_train_list:
    ids_train += ids.tolist()
  return ids_train


class CardiotoxFingerprintDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for cardiotoxicity fingerprint dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'example_id': tfds.features.Text(),
            'feature': tfds.features.Tensor(shape=(1024,),
                                            dtype=tf.int64),
            'label': tfds.features.ClassLabel(num_classes=2),
        }),
        supervised_keys=('feature', 'label', 'example_id'),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Loads data and defines splits."""
    return {
        'train': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TRAIN_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TRAIN_LABEL),
        'validation': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_VALIDATION_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_VALIDATION_LABEL),
        'test': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TEST_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TEST_LABEL),
        'test2': self._generate_examples(
            features_path=_DEFAULT_PATH_CARDIOTOX_TEST2_FEATURE,
            label_path=_DEFAULT_PATH_CARDIOTOX_TEST2_LABEL),
    }

  def _generate_examples(self, features_path, label_path) -> Iterator[
      Tuple[str, Dict[str, Any]]]:
    """Generates examples for each split."""
    with tf.io.gfile.GFile(features_path, 'r') as features_file:
      x = json.load(features_file)
    with tf.io.gfile.GFile(label_path, 'r') as label_file:
      y = json.load(label_file)
    for idx, molecule_id in enumerate(x):
      key = '_'.join([molecule_id, str(idx)])
      yield key, {
          'example_id': molecule_id,
          'feature': x[molecule_id],
          'label': y[molecule_id]
      }


@register_dataset('cardiotoxicity')
def get_cardiotoxicity_dataset(
    num_splits: int,
    initial_sample_proportion: float,
    subgroup_ids: List[str], subgroup_proportions: List[float]

) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    initial_sample_proportion: Float for proportion of entire training
      dataset to sample initially before active sampling begins.
    subgroup_ids: List of strings of IDs indicating subgroups.
    subgroup_proportions: List of floats indicating proportion that each
      subgroup should take in initial training dataset.

  Returns:if subgroup_proportions:
      self.subgroup_proportions = subgroup_proportions
    else:
      self.subgroup_proportions = [1.] * len(subgroup_ids)
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  # No subgroups in this datset so ignored
  del subgroup_ids, subgroup_proportions
  split_size_in_pct = int(100 * initial_sample_proportion/ num_splits)
  val_splits = tfds.load(
      'cardiotox_fingerprint_dataset',
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, int(100 *
                                initial_sample_proportion), split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True)

  train_splits = tfds.load(
      'cardiotox_fingerprint_dataset',
      split=[
          f'train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, int(100 *
                                initial_sample_proportion), split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True)

  test_ds = tfds.load(
      'cardiotox_fingerprint_dataset',
      split='test',
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  test2_ds = tfds.load(
      'cardiotox_fingerprint_dataset',
      split='test2',
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
      'test2': test2_ds
  }
  return Dataloader(train_splits, val_splits, train_ds, eval_ds=eval_datasets)


class WaterbirdsDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Waterbirds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self,
               subgroup_ids: List[str],
               initial_sample_proportion: Optional[float] = None,
               subgroup_proportions: Optional[List[float]] = None,
               **kwargs):
    super(WaterbirdsDataset, self).__init__(**kwargs)
    self.subgroup_ids = subgroup_ids
    if initial_sample_proportion:
      self.initial_sample_proportion = initial_sample_proportion
    if subgroup_proportions:
      self.subgroup_proportions = subgroup_proportions
    else:
      self.subgroup_proportions = [1.] * len(subgroup_ids)

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'example_id': tfds.features.Text(),
            'subgroup_id': tfds.features.Text(),
            'subgroup_label': tfds.features.ClassLabel(num_classes=2),
            'feature': tfds.features.Image(shape=(224, 224, 3)),
            'label': tfds.features.ClassLabel(num_classes=2),
            'place': tfds.features.ClassLabel(num_classes=2),
            'image_filename': tfds.features.Text(),
            'place_filename': tfds.features.Text(),
        }),
        supervised_keys=('feature', 'label', 'example_id'),
    )

  def _decode_and_center_crop(self, image_bytes: tf.Tensor):
    """Crops to center of image with padding then scales RESNET_IMAGE_SIZE."""
    shape = tf.io.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        ((RESNET_IMAGE_SIZE / (RESNET_IMAGE_SIZE + CROP_PADDING)) *
         tf.cast(tf.math.minimum(image_height, image_width), tf.float32)),
        tf.int32)

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack([offset_height, offset_width,
                            padded_center_crop_size, padded_center_crop_size])
    image = tf.io.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    return image

  def _preprocess_image(self, image_bytes: tf.Tensor) -> tf.Tensor:
    """Preprocesses the given image for evaluation.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.

    Returns:
      A preprocessed image `Tensor`.
    """
    image = self._decode_and_center_crop(image_bytes)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.resize([image], [RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE],
                            method='nearest')[0]
    return image

  def _dataset_parser(self, value):
    """Parse a Waterbirds record from a serialized string Tensor."""
    keys_to_features = {
        'image/filename/raw': tf.io.FixedLenFeature([], tf.string, ''),
        'image/class/place': tf.io.FixedLenFeature([], tf.int64, -1),
        'image/encoded': tf.io.FixedLenFeature([], tf.string, ''),
        'image/filename/places': tf.io.FixedLenFeature([], tf.string, ''),
        'image/class/label': tf.io.FixedLenFeature([], tf.int64, -1),
    }

    parsed = tf.io.parse_single_example(value, keys_to_features)

    image = self._preprocess_image(image_bytes=parsed['image/encoded'])
    label = tf.cast(parsed['image/class/label'], dtype=tf.int32)
    place = tf.cast(parsed['image/class/place'], dtype=tf.int32)
    image_filename = tf.cast(parsed['image/filename/raw'], dtype=tf.string)
    place_filename = tf.cast(parsed['image/filename/places'], dtype=tf.string)

    return image, label, place, image_filename, place_filename

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    # _WATERBIRDS_DATA_DIR
    return {
        'train':
            self._generate_examples(
                os.path.join(_WATERBIRDS_DATA_DIR, _WATERBIRDS_TRAIN_PATTERN),
                is_training=True),
        'validation':
            self._generate_examples(
                os.path.join(_WATERBIRDS_DATA_DIR,
                             _WATERBIRDS_VALIDATION_PATTERN)),
        'train_sample':
            self._generate_examples(
                os.path.join(_WATERBIRDS_DATA_DIR,
                             _WATERBIRDS_TRAIN_SAMPLE_PATTERN)),
        'test':
            self._generate_examples(
                os.path.join(_WATERBIRDS_DATA_DIR, _WATERBIRDS_TEST_PATTERN)),
    }

  def _generate_examples(self,
                         file_pattern: str,
                         is_training: Optional[bool] = False
                        ) -> Iterator[Tuple[str, Dict[str, Any]]]:
    """Generator of examples for each split."""
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=is_training)

    def _fetch_dataset(filename):
      buffer_size = 8 * 1024 * 1024  # 8 MiB per file
      dataset = tf.data.TFRecordDataset(filename, buffer_size=buffer_size)
      return dataset

    # Reads the data from disk in parallel.
    dataset = dataset.interleave(
        _fetch_dataset,
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Parses and pre-processes the data in parallel.
    dataset = dataset.map(self._dataset_parser, num_parallel_calls=2)

    # Prefetches overlaps in-feed with training.
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    if is_training:
      options = tf.data.Options()
      options.experimental_deterministic = False
      dataset = dataset.with_options(options)

      # Prepare initial training set.
      # Pre-computed dataset size or large number >= estimated dataset size.
      dataset_size = 4780
      dataset = dataset.shuffle(dataset_size)
      sampled_datasets = []
      remaining_proportion = 1.
      for idx, subgroup_id in enumerate(self.subgroup_ids):

        def filter_fn_subgroup(image, label, place, image_filename,
                               place_filename):
          _ = image, image_filename, place_filename
          return tf.math.equal(
              tf.strings.join(
                  [tf.strings.as_string(label),
                   tf.strings.as_string(place)],
                  separator='_'), subgroup_id)  # pylint: disable=cell-var-from-loop

        subgroup_dataset = dataset.filter(filter_fn_subgroup)
        subgroup_sample_size = int(dataset_size *
                                   self.subgroup_proportions[idx] *
                                   self.initial_sample_proportion)
        subgroup_dataset = subgroup_dataset.take(subgroup_sample_size)
        sampled_datasets.append(subgroup_dataset)
        remaining_proportion -= self.subgroup_proportions[idx]

      def filter_fn_remaining(image, label, place, image_filename,
                              place_filename):
        _ = image, image_filename, place_filename
        return tf.reduce_all(
            tf.math.not_equal(
                tf.strings.join(
                    [tf.strings.as_string(label),
                     tf.strings.as_string(place)],
                    separator='_'), self.subgroup_ids))

      remaining_dataset = dataset.filter(filter_fn_remaining)
      remaining_sample_size = int(dataset_size * remaining_proportion *
                                  self.initial_sample_proportion)
      remaining_dataset = remaining_dataset.take(remaining_sample_size)
      sampled_datasets.append(remaining_dataset)

      dataset = sampled_datasets[0]
      for ds in sampled_datasets[1:]:
        dataset = dataset.concatenate(ds)
      dataset = dataset.shuffle(dataset_size)

    for example in dataset:
      image, label, place, image_filename, place_filename = example
      subgroup_id = str(label.numpy()) + '_' + str(place.numpy())
      subgroup_label = 1 if subgroup_id in self.subgroup_ids else 0
      yield image_filename.numpy(), {
          'example_id': image_filename.numpy(),
          'subgroup_id': subgroup_id,
          'subgroup_label': subgroup_label,
          'feature': image.numpy(),
          'label': label.numpy(),
          'place': place.numpy(),
          'image_filename': image_filename.numpy(),
          'place_filename': place_filename.numpy(),
      }


@register_dataset('waterbirds')
def get_waterbirds_dataset(
    num_splits: int, batch_size: int, initial_sample_proportion: float,
    subgroup_ids: List[str], subgroup_proportions: List[float]
) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    batch_size: Integer for number of examples per batch.
    initial_sample_proportion: Float for proportion of entire training
      dataset to sample initially before active sampling begins.
    subgroup_ids: List of strings of IDs indicating subgroups.
    subgroup_proportions: List of floats indicating proportion that each
      subgroup should take in initial training dataset.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  split_size_in_pct = int(100 / num_splits)
  builder_kwargs = {
      'initial_sample_proportion': initial_sample_proportion,
      'subgroup_ids': subgroup_ids,
      'subgroup_proportions': subgroup_proportions
  }
  val_splits = tfds.load(
      'waterbirds_dataset',
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      as_supervised=True)

  train_splits = tfds.load(
      'waterbirds_dataset',
      split=[
          f'train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      as_supervised=True)

  train_sample = tfds.load(
      'waterbirds_dataset',
      split='train_sample',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  test_ds = tfds.load(
      'waterbirds_dataset',
      split='test',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
  }
  return Dataloader(
      train_splits,
      val_splits,
      train_ds,
      train_sample_ds=train_sample,
      eval_ds=eval_datasets)


@register_dataset('celeb_a')
def get_celeba_dataset(
    num_splits: int, batch_size: int
) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    batch_size: Integer for number of examples per batch.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  read_config = tfds.ReadConfig()
  read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key
  split_size_in_pct = int(100 / num_splits)
  train_splits = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'train[:{k}%]+train[{k+split_size_in_pct}%:]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True
      )
  val_splits = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, 100, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True
      )
  train_sample = tfds.load(
      'celeb_a',
      split='train_sample',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  test_ds = tfds.load(
      'celeb_a',
      split='test',
      data_dir=DATA_DIR,
      batch_size=batch_size,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
  }
  return Dataloader(
      train_splits,
      val_splits,
      train_ds,
      train_sample_ds=train_sample,
      eval_ds=eval_datasets)
