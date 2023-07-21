# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

import collections
import dataclasses
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import uuid

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

DATASET_REGISTRY = {}
DATA_DIR = '/tmp/data'
_WATERBIRDS_DATA_DIR = ''
_WATERBIRDS10K_DATA_DIR = ''
_WATERBIRDS_TRAIN_PATTERN = ''
_WATERBIRDS_VALIDATION_PATTERN = ''
# Smaller subsample for testing.
_WATERBIRDS_TRAIN_SAMPLE_PATTERN = ''
_WATERBIRDS_TEST_PATTERN = ''
_WATERBIRDS_NUM_SUBGROUP = 4
_WATERBIRDS_TRAIN_SIZE = 4780
_WATERBIRDS10K_TRAIN_SIZE = 9549
_WATERBIRDS10K_SUPPORTED_CORR_STRENGTH = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9,
                                          0.95)

# TODO(dvij,martinstrobel): Set Celeb-A number of subgroups.
_CELEB_A_NUM_SUBGROUP = 2

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
  num_subgroups: int  # Number of subgroups in data.
  subgroup_sizes: Dict[str, int]  # Number of examples by subgroup.
  train_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  val_splits: tf.data.Dataset  # Result of tfds.load with 'split' arg.
  train_ds: tf.data.Dataset  # Dataset with all the train splits combined.
  num_train_examples: Optional[int] = 0  # Number of training examples.
  worst_group_label: Optional[int] = 2  # Label of the worst subgroup.
  train_sample_ds: Optional[tf.data.Dataset] = None  # Subsample of train set.
  eval_ds: Optional[Dict[
      str,
      tf.data.Dataset]] = None  # Validation and any additional test datasets.


def get_subgroup_sizes(dataloader: tf.data.Dataset) -> Dict[str, int]:
  """Gets the number examples of each subgroup."""
  subgroup_sizes = dict(
      collections.Counter(
          dataloader.map(lambda x: x['subgroup_label']).as_numpy_iterator()
      )
  )
  return {str(key): val for key, val in subgroup_sizes.items()}


def upsample_subgroup(
    dataset: tf.data.Dataset,
    lambda_value: int = 60,
    signal: str = 'subgroup_label',
    subgroup_sizes: Optional[Dict[str, int]] = None,
) -> tf.data.Dataset:
  """Creates dataset that has upsampled subgroup.

  Args:
    dataset: Dataset to be transformed.
    lambda_value: Number of times each example of the underrepresented group
      should be repeated in dataset.
    signal: String for the value that determines whether or not an example
      belongs to an underrepresented group.
    subgroup_sizes: Dictionary mapping subgroup index to size.

  Returns:
    Transformed dataset.
  """
  if signal != 'subgroup_label':
    raise ValueError(
        'Upsampling with signals other than subgroup_label is not supported.'
    )
  # In this case, we assume that the data has ground-truth subgroup labels.
  # Identify the group that is smallest and upsample it.
  if not subgroup_sizes:
    raise ValueError(
        'When using ground-truth subgroup label as upsampling signal,'
        ' dictionary of subgroup sizes must be available.'
    )
  examples_by_subgroup = {}
  smallest_subgroup_label = ''
  smallest_subgroup_size = -1
  for subgroup_label in subgroup_sizes.keys():

    def filter_subgroup(x, label=subgroup_label):
      return tf.math.equal(
          x['subgroup_label'], tf.strings.to_number(label, tf.int64)
      )

    examples_by_subgroup[subgroup_label] = dataset.filter(filter_subgroup)
    if smallest_subgroup_size == -1:
      smallest_subgroup_label = subgroup_label
      smallest_subgroup_size = subgroup_sizes[subgroup_label]
    elif subgroup_sizes[subgroup_label] < smallest_subgroup_size:
      smallest_subgroup_label = subgroup_label
      smallest_subgroup_size = subgroup_sizes[subgroup_label]
  examples_by_subgroup[smallest_subgroup_label] = examples_by_subgroup[
      smallest_subgroup_label
  ].repeat(lambda_value)
  subgroup_sizes[smallest_subgroup_label] *= lambda_value
  dataset_size = sum(subgroup_sizes.values())
  weights = [
      float(subgroup_sizes[subgroup_label]) / dataset_size
      for subgroup_label in subgroup_sizes
  ]
  upsampled_dataset = tf.data.Dataset.sample_from_datasets(
      examples_by_subgroup.values(),
      weights=weights,
      stop_on_empty_dataset=False,
  )
  return upsampled_dataset


def apply_batch(dataloader, batch_size):
  """Apply batching to dataloader."""
  dataloader.train_splits = [
      data.batch(batch_size) for data in dataloader.train_splits
  ]
  dataloader.val_splits = [
      data.batch(batch_size) for data in dataloader.val_splits
  ]
  num_splits = len(dataloader.train_splits)
  train_ds = gather_data_splits(
      list(range(num_splits)), dataloader.train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), dataloader.val_splits)
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


def get_ids_from_dataset(dataset: tf.data.Dataset) -> List[str]:
  """Gets example ids from dataset."""
  ids_list = list(dataset.map(lambda x: x['example_id']).as_numpy_iterator())
  if isinstance(ids_list[0], np.ndarray):
    new_ids_list = []
    for ids in ids_list:
      new_ids_list += ids.tolist()
    return new_ids_list
  else:
    return ids_list


def create_ids_table(dataloader: Dataloader,
                     initial_sample_proportion: float,
                     initial_sample_seed: int,
                     split_proportion: float,
                     split_num: int,
                     split_seed: int,
                     training: bool) -> tf.lookup.StaticHashTable:
  """Creates a hash table representing ids in each each split.

  Args:
    dataloader: Dataloader for the unfilterd dataset.
    initial_sample_proportion: Proportion of larger subset to initial dataset.
    initial_sample_seed: Seed to select the larger subset (identical for all
      splits.)
    split_proportion: Proportion of split to larger subset.
    split_num: Number of split. Used to set the sampling seed for the split
      subset.
    split_seed: Split seed second part of the sampling seed.
    training: Whether to create a training set or a validation set.

  Returns:
    A hash table mapping ids to membership in the filtered dataset.
  """
  ids = get_ids_from_dataset(dataloader.train_ds)
  initial_sample_size = int(len(ids) * initial_sample_proportion)
  np.random.seed(initial_sample_seed)
  subset_ids = np.random.choice(ids, initial_sample_size, replace=False)
  # ids_dir is populated by the sample_and_split_ids function above
  tf.compat.v1.logging.info('Seed number %d', split_num + split_seed)
  np.random.seed(split_num + split_seed)
  ids_i = np.random.choice(
      subset_ids, int(split_proportion * initial_sample_size), replace=False)
  tf.compat.v1.logging.info('Subset size %d', len(ids_i))
  if not training:
    ids_i = subset_ids[~np.isin(subset_ids, ids_i)]
  keys = tf.convert_to_tensor(ids_i, dtype=tf.string)
  values = tf.ones(shape=keys.shape, dtype=tf.int64)
  init = tf.lookup.KeyValueTensorInitializer(
      keys=keys,
      values=values,
      key_dtype=tf.string,
      value_dtype=tf.int64)
  return tf.lookup.StaticHashTable(init, default_value=0)


def filter_set(
    dataloader: Dataloader,
    initial_sample_proportion: float,
    initial_sample_seed: int,
    split_proportion: float,
    split_id: int,
    split_seed: int,
    training: bool,
) -> tf.data.Dataset:
  """Filters training set to create subsets of arbitrary size.

  First, a set of initial_sample_proportion is selected from which the different
  training an validation sets are sampled according to split_proportion. The
  indiviudal splits are used to train an ensemble of models where each model is
  trained on an equally sized training set.

  Args:
    dataloader: Dataloader for the unfilterd dataset.
    initial_sample_proportion: Proportion of larger subset to initial dataset.
    initial_sample_seed: Seed to select the larger subset (identical for all
      splits.)
    split_proportion: Proportion of split to larger subset.
    split_id: Number of split. Used to set the sampling seed for the split
      subset.
    split_seed: Split seed second part of the sampling seed.
    training:  Whether to create a training set or a validation set.

  Returns:
    A filtered dataset.
  """
  filter_table = create_ids_table(
      dataloader,
      initial_sample_proportion=initial_sample_proportion,
      initial_sample_seed=initial_sample_seed,
      split_proportion=split_proportion,
      split_num=split_id,
      split_seed=split_seed,
      training=training,
  )
  return dataloader.train_ds.filter(
      lambda datapoint: filter_table.lookup(datapoint['example_id']) == 1
  )


class WaterbirdsDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Waterbirds dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self,
               subgroup_ids: List[str],
               subgroup_proportions: Optional[List[float]] = None,
               train_dataset_size: int = _WATERBIRDS_TRAIN_SIZE,
               source_data_dir: str = _WATERBIRDS_DATA_DIR,
               include_train_sample: bool = True,
               **kwargs):
    super(WaterbirdsDataset, self).__init__(**kwargs)
    self.subgroup_ids = subgroup_ids
    self.train_dataset_size = train_dataset_size
    # Path to original TFRecords to sample data from.
    self.source_data_dir = source_data_dir
    self.include_train_sample = include_train_sample
    if subgroup_proportions:
      self.subgroup_proportions = subgroup_proportions
    else:
      self.subgroup_proportions = [1.] * len(subgroup_ids)

  def _info(self) -> tfds.core.DatasetInfo:
    """Dataset metadata (homepage, citation,...)."""
    return tfds.core.DatasetInfo(
        builder=self,
        features=tfds.features.FeaturesDict({
            'example_id':
                tfds.features.Text(),
            'subgroup_id':
                tfds.features.Text(),
            'subgroup_label':
                tfds.features.ClassLabel(num_classes=4),
            'input_feature':
                tfds.features.Image(
                    shape=(RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE, 3)),
            'label':
                tfds.features.ClassLabel(num_classes=2),
            'place':
                tfds.features.ClassLabel(num_classes=2),
            'image_filename':
                tfds.features.Text(),
            'place_filename':
                tfds.features.Text(),
        }),
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
    crop_window = tf.stack([
        offset_height, offset_width, padded_center_crop_size,
        padded_center_crop_size
    ])
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
    # No data augmentation, like in JTT paper.
    # image = tf.image.random_flip_left_right(image)
    image = tf.image.resize([image], [RESNET_IMAGE_SIZE, RESNET_IMAGE_SIZE])[0]
    return image

  def _get_subgroup_label(self, label: tf.Tensor,
                          place: tf.Tensor) -> tf.Tensor:
    """Determines subgroup label for given combination of label and place.

    0 for landbirds on land, 1 for waterbirds on water, 2 for landbirds
    on water, and 3 for waterbirds on land.

    Args:
      label: Class label (waterbird or landbird).
      place: Place label (water or land).

    Returns:
      TF Tensor containing subgroup label (integer).
    """
    if tf.math.equal(label, place):
      return label
    else:
      if tf.math.equal(label, 1):  # and place == 0, so waterbird on land
        return tf.constant(2, dtype=tf.int32)
      else:
        return tf.constant(3, dtype=tf.int32)

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
    subgroup_id = tf.strings.join(
        [tf.strings.as_string(label),
         tf.strings.as_string(place)],
        separator='_')
    subgroup_label = self._get_subgroup_label(label, place)

    return image_filename, {
        'example_id': image_filename,
        'label': label,
        'place': place,
        'input_feature': image,
        'image_filename': image_filename,
        'place_filename': place_filename,
        'subgroup_id': subgroup_id,
        'subgroup_label': subgroup_label
    }

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    """Download the data and define splits."""
    split_generators = {
        'train':
            self._generate_examples(
                os.path.join(self.source_data_dir, _WATERBIRDS_TRAIN_PATTERN),
                is_training=True),
        'validation':
            self._generate_examples(
                os.path.join(self.source_data_dir,
                             _WATERBIRDS_VALIDATION_PATTERN)),
        'test':
            self._generate_examples(
                os.path.join(self.source_data_dir, _WATERBIRDS_TEST_PATTERN)),
    }

    if self.include_train_sample:
      split_generators['train_sample'] = self._generate_examples(
          os.path.join(self.source_data_dir, _WATERBIRDS_TRAIN_SAMPLE_PATTERN))

    return split_generators

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
      dataset_size = self.train_dataset_size
      dataset = dataset.shuffle(dataset_size)
      sampled_datasets = []
      remaining_proportion = 1.
      for idx, subgroup_id in enumerate(self.subgroup_ids):

        def filter_fn_subgroup(image_filename, feats):
          _ = image_filename
          return tf.math.equal(feats['subgroup_id'], subgroup_id)  # pylint: disable=cell-var-from-loop

        subgroup_dataset = dataset.filter(filter_fn_subgroup)
        subgroup_sample_size = int(dataset_size *
                                   self.subgroup_proportions[idx])
        subgroup_dataset = subgroup_dataset.take(subgroup_sample_size)
        sampled_datasets.append(subgroup_dataset)
        remaining_proportion -= self.subgroup_proportions[idx]

      def filter_fn_remaining(image_filename, feats):
        _ = image_filename
        return tf.reduce_all(
            tf.math.not_equal(feats['subgroup_id'], self.subgroup_ids))

      remaining_dataset = dataset.filter(filter_fn_remaining)
      remaining_sample_size = int(dataset_size * remaining_proportion)
      remaining_dataset = remaining_dataset.take(remaining_sample_size)
      sampled_datasets.append(remaining_dataset)

      dataset = sampled_datasets[0]
      for ds in sampled_datasets[1:]:
        dataset = dataset.concatenate(ds)
      dataset = dataset.shuffle(dataset_size)

    return dataset.as_numpy_iterator()


class Waterbirds10kDataset(WaterbirdsDataset):
  """DatasetBuilder for Waterbirds10K dataset."""

  VERSION = tfds.core.Version('1.0.0')
  RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
  }

  def __init__(self,
               subgroup_ids: List[str],
               subgroup_proportions: Optional[List[float]] = None,
               corr_strength: float = 0.95,
               train_dataset_size: int = _WATERBIRDS10K_TRAIN_SIZE,
               source_data_parent_dir: str = _WATERBIRDS10K_DATA_DIR,
               include_train_sample: bool = False,
               **kwargs):
    if corr_strength not in _WATERBIRDS10K_SUPPORTED_CORR_STRENGTH:
      raise ValueError(
          f'corr_strength {corr_strength} not supported. '
          f'Should be one of: {_WATERBIRDS10K_SUPPORTED_CORR_STRENGTH}')

    # Makes the source data directory based on `corr_strength`.
    # The final data directory should follow the format
    # `{parent_dir}/corr_strength_{corr_strength}`.
    corr_strength_name = str(int(corr_strength * 100))
    source_data_folder_name = f'corr_strength_{corr_strength_name}'
    source_data_dir = os.path.join(source_data_parent_dir,
                                   source_data_folder_name)

    if not tf.io.gfile.exists(source_data_dir):
      raise ValueError(f'Required data dir `{source_data_dir}` not exist.')
    else:
      tf.compat.v1.logging.info(f'Loading from `{source_data_dir}`.')

    self.corr_strength = corr_strength
    super().__init__(subgroup_ids, subgroup_proportions, train_dataset_size,
                     source_data_dir, include_train_sample, **kwargs)


@dataclasses.dataclass
class SkaiDatasetConfig(tfds.core.BuilderConfig):
  """Configuration for SKAI datasets.

  Any of the attributes can be left blank if they don't exist.

  Attributes:
    name: Name of the dataset.
    labeled_train_pattern: Pattern for labeled training examples tfrecords.
    labeled_test_pattern: Pattern for labeled test examples tfrecords.
    unlabeled_pattern: Pattern for unlabeled examples tfrecords.
    use_post_disaster_only: Whether to use post-disaster imagery only rather
      than full 6-channel stacked image input.
  """
  labeled_train_pattern: str = ''
  labeled_test_pattern: str = ''
  unlabeled_pattern: str = ''
  use_post_disaster_only: bool = False
  image_size: int = RESNET_IMAGE_SIZE
  max_examples: int = 0
  load_small_images: bool = True


def _decode_and_resize_image(
    image_bytes: tf.Tensor, size: int) -> tf.Tensor:
  return tf.image.resize(
      tf.io.decode_image(
          image_bytes,
          channels=3,
          expand_animations=False,
          dtype=tf.float32,
      ),
      [size, size],
  )


class SkaiDataset(tfds.core.GeneratorBasedBuilder):
  """TFDS dataset for SKAI.

  Example usage:
    import tensorflow_datasets.public_api as tfds
    from skai import dataset

    ds = tfds.load('skai_dataset', builder_kwargs={
      'config': SkaiDatasetConfig(
          name='example',
          labeled_train_pattern='gs://path/to/train_labeled_examples.tfrecord',
          labeled_test_pattern='gs://path/to/test_labeled_examples.tfrecord',
          unlabeled_pattern='gs://path/to/unlabeled_examples-*.tfrecord')
    })
    labeled_train_dataset = ds['labeled_train']
    labeled_test_dataset = ds['labeled_test']
    unlabeled_test_dataset = ds['unlabeled']
  """

  VERSION = tfds.core.Version('1.0.0')


  def __init__(self,
               subgroup_ids: Optional[List[str]] = None,
               subgroup_proportions: Optional[List[float]] = None,
               include_train_sample: bool = True,
               **kwargs):
    super(SkaiDataset, self).__init__(**kwargs)
    self.subgroup_ids = subgroup_ids
    # Path to original TFRecords to sample data from.
    if self.subgroup_ids:
      if subgroup_proportions:
        self.subgroup_proportions = subgroup_proportions
      else:
        self.subgroup_proportions = [1.] * len(subgroup_ids)
    else:
      self.subgroup_proportions = None
    self.include_train_sample = include_train_sample

  def _info(self):
    # TODO(jihyeonlee): Change label and subgroup_label to
    #   tfds.features.ClassLabel.
    num_channels = 3 if self.builder_config.use_post_disaster_only else 6
    input_shape = (
        self.builder_config.image_size,
        self.builder_config.image_size,
        num_channels,
    )
    if self.builder_config.load_small_images:
      input_type = tfds.features.FeaturesDict({
          'large_image': tfds.features.Tensor(
              shape=input_shape, dtype=tf.float32
          ),
          'small_image': tfds.features.Tensor(
              shape=input_shape, dtype=tf.float32
          ),
      })
    else:
      input_type = tfds.features.FeaturesDict({
          'large_image': tfds.features.Tensor(
              shape=input_shape, dtype=tf.float32
          ),
      })
    return tfds.core.DatasetInfo(
        builder=self,
        description='Skai',
        features=tfds.features.FeaturesDict({
            'input_feature': input_type,
            'example_id': tfds.features.Text(),
            'coordinates': tfds.features.Tensor(shape=(2,), dtype=tf.float32),
            'label': tfds.features.Tensor(shape=(), dtype=tf.int64),
            'string_label': tfds.features.Text(),
            'subgroup_label': tfds.features.Tensor(shape=(), dtype=tf.int64),
        }),
    )

  def _split_generators(self, dl_manager: tfds.download.DownloadManager):
    splits = {}
    if self.builder_config.labeled_train_pattern:
      splits['labeled_train'] = self._generate_examples(
          self.builder_config.labeled_train_pattern
      )
    if self.builder_config.labeled_test_pattern:
      splits['labeled_test'] = self._generate_examples(
          self.builder_config.labeled_test_pattern
      )
    if self.builder_config.unlabeled_pattern:
      splits['unlabeled'] = self._generate_examples(
          self.builder_config.unlabeled_pattern
      )
    return splits

  def _decode_record(self, record_bytes):

    example = tf.io.parse_single_example(
        record_bytes,
        {
            'coordinates': tf.io.FixedLenFeature([2], dtype=tf.float32),
            'encoded_coordinates': tf.io.FixedLenFeature([], dtype=tf.string),
            'example_id': tf.io.FixedLenFeature([], dtype=tf.string),
            'pre_image_png_large': tf.io.FixedLenFeature([], dtype=tf.string),
            'pre_image_png': tf.io.FixedLenFeature([], dtype=tf.string),
            'post_image_png_large': tf.io.FixedLenFeature(
                [], dtype=tf.string
            ),
            'post_image_png': tf.io.FixedLenFeature([], dtype=tf.string),
            'label': tf.io.FixedLenFeature([], dtype=tf.float32),
            'string_label': tf.io.FixedLenFeature(
                [], dtype=tf.string, default_value=''
            ),
        },
    )

    features = {
        'input_feature': {}
    }
    large_image_concat = _decode_and_resize_image(
        example['post_image_png_large'], self.builder_config.image_size
    )
    small_image_concat = _decode_and_resize_image(
        example['post_image_png'], self.builder_config.image_size
    )

    if not self.builder_config.use_post_disaster_only:
      before_image = _decode_and_resize_image(
          example['pre_image_png_large'], self.builder_config.image_size
      )
      before_image_small = _decode_and_resize_image(
          example['pre_image_png'], self.builder_config.image_size
      )
      large_image_concat = tf.concat(
          [before_image, large_image_concat], axis=-1
      )
      small_image_concat = tf.concat(
          [before_image_small, small_image_concat], axis=-1
      )
    features['input_feature']['large_image'] = large_image_concat
    if self.builder_config.load_small_images:
      features['input_feature']['small_image'] = small_image_concat
    features['label'] = tf.cast(example['label'], tf.int64)
    features['example_id'] = example['example_id']
    features['subgroup_label'] = features['label']
    features['coordinates'] = example['coordinates']
    features['string_label'] = example['string_label']
    return features

  def _generate_examples(self, pattern: str):
    if not pattern:
      return
    paths = tf.io.gfile.glob(pattern)
    ds = tf.data.TFRecordDataset(paths).map(
        self._decode_record, num_parallel_calls=tf.data.AUTOTUNE)
    if self.builder_config.max_examples:
      ds = ds.take(self.builder_config.max_examples)
    for features in ds.as_numpy_iterator():
      yield uuid.uuid4().hex, features


@register_dataset('waterbirds')
def get_waterbirds_dataset(num_splits: int,
                           initial_sample_proportion: float,
                           subgroup_ids: List[str],
                           subgroup_proportions: List[float],
                           tfds_dataset_name: str = 'waterbirds_dataset',
                           include_train_sample: bool = True,
                           data_dir: str = DATA_DIR,
                           upsampling_lambda: int = 1,
                           upsampling_signal: str = 'subgroup_label',
                           **additional_builder_kwargs) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    initial_sample_proportion: Float for proportion of entire training dataset
      to sample initially before active sampling begins.
    subgroup_ids: List of strings of IDs indicating subgroups.
    subgroup_proportions: List of floats indicating proportion that each
      subgroup should take in initial training dataset.
    tfds_dataset_name: The name of the tfd dataset to load from.
    include_train_sample: Whether to include the `train_sample` split.
    data_dir: Default data directory to store the sampled waterbirds data.
    upsampling_lambda: Number of times subgroup examples should be repeated.
    upsampling_signal: Signal to use to determine subgroup to upsample.
    **additional_builder_kwargs: Additional keyword arguments to data builder.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  split_size_in_pct = int(100 * initial_sample_proportion / num_splits)
  reduced_datset_sz = int(100 * initial_sample_proportion)
  builder_kwargs = {
      'subgroup_ids': subgroup_ids,
      'subgroup_proportions': subgroup_proportions,
      'include_train_sample': include_train_sample,
      **additional_builder_kwargs
  }
  val_splits = tfds.load(
      tfds_dataset_name,
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, reduced_datset_sz, split_size_in_pct)
      ],
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False)

  train_splits = tfds.load(
      tfds_dataset_name,
      split=[
          f'train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, reduced_datset_sz, split_size_in_pct)
      ],
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False)

  test_ds = tfds.load(
      tfds_dataset_name,
      split='test',
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      with_info=False)

  train_sample = ()
  if include_train_sample:
    train_sample = tfds.load(
        tfds_dataset_name,
        split='train_sample',
        data_dir=data_dir,
        builder_kwargs=builder_kwargs,
        try_gcs=False,
        with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
  }
  subgroup_sizes = get_subgroup_sizes(train_ds)
  if upsampling_lambda > 1:
    train_ds = upsample_subgroup(
        train_ds, upsampling_lambda, upsampling_signal, subgroup_sizes
    )

  return Dataloader(
      _WATERBIRDS_NUM_SUBGROUP,
      subgroup_sizes,
      train_splits,
      val_splits,
      train_ds,
      num_train_examples=_WATERBIRDS_TRAIN_SIZE,
      worst_group_label=2,  # 1_0, waterbirds on land.
      train_sample_ds=train_sample,
      eval_ds=eval_datasets)


@register_dataset('waterbirds10k')
def get_waterbirds10k_dataset(
    num_splits: int,
    initial_sample_proportion: float,
    subgroup_ids: List[str],
    subgroup_proportions: List[float],
    corr_strength: float = 0.95,
    data_dir: str = DATA_DIR,
    upsampling_lambda: int = 1,
    upsampling_signal: str = 'subgroup_label',
) -> Dataloader:
  """Returns datasets for Waterbirds 10K."""
  # Create unique `waterbirds10k` directory for each correlation strength.
  data_folder_name = int(corr_strength * 100)
  data_folder_name = f'waterbirds10k_corr_strength_{data_folder_name}'
  data_dir = os.path.join(data_dir, data_folder_name)

  return get_waterbirds_dataset(
      num_splits,
      initial_sample_proportion,
      subgroup_ids,
      subgroup_proportions,
      tfds_dataset_name='waterbirds10k_dataset',
      include_train_sample=False,
      corr_strength=corr_strength,
      data_dir=data_dir,
      upsampling_lambda=upsampling_lambda,
      upsampling_signal=upsampling_signal)


@register_dataset('celeb_a')
def get_celeba_dataset(
    num_splits: int,
    initial_sample_proportion: float,
    subgroup_ids: List[str],
    subgroup_proportions: List[float],
    upsampling_lambda: int = 1,
    upsampling_signal: str = 'subgroup_label',
) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    initial_sample_proportion: Float for proportion of entire training dataset
      to sample initially before active sampling begins.
    subgroup_ids: List of strings of IDs indicating subgroups.
    subgroup_proportions: List of floats indicating proportion that each
      subgroup should take in initial training dataset.
    upsampling_lambda: Number of times subgroup examples should be repeated.
    upsampling_signal: Signal to use to determine subgroup to upsample.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """
  del subgroup_proportions, subgroup_ids
  read_config = tfds.ReadConfig()
  read_config.add_tfds_id = True  # Set `True` to return the 'tfds_id' key
  split_size_in_pct = int(100 * initial_sample_proportion / num_splits)
  reduced_dataset_sz = int(100 * initial_sample_proportion)
  train_splits = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'train[:{k}%]+train[{k+split_size_in_pct}%:]'
          for k in range(0, reduced_dataset_sz, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True)
  val_splits = tfds.load(
      'celeb_a',
      read_config=read_config,
      split=[
          f'validation[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, reduced_dataset_sz, split_size_in_pct)
      ],
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True)
  train_sample = tfds.load(
      'celeb_a',
      split='train_sample',
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  test_ds = tfds.load(
      'celeb_a',
      split='test',
      data_dir=DATA_DIR,
      try_gcs=False,
      as_supervised=True,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
  }
  subgroup_sizes = get_subgroup_sizes(train_ds)
  if upsampling_lambda > 1:
    train_ds = upsample_subgroup(
        train_ds, upsampling_lambda, upsampling_signal, subgroup_sizes
    )

  return Dataloader(
      _CELEB_A_NUM_SUBGROUP,
      subgroup_sizes,
      train_splits,
      val_splits,
      train_ds,
      train_sample_ds=train_sample,
      eval_ds=eval_datasets)


@register_dataset('skai')
def get_skai_dataset(num_splits: int,
                     initial_sample_proportion: float,
                     subgroup_ids: List[str],
                     subgroup_proportions: List[float],
                     tfds_dataset_name: str = 'skai_dataset',
                     data_dir: str = DATA_DIR,
                     include_train_sample: Optional[bool] = False,
                     labeled_train_pattern: str = '',
                     unlabeled_train_pattern: str = '',
                     validation_pattern: str = '',
                     use_post_disaster_only: Optional[bool] = False,
                     upsampling_lambda: int = 1,
                     upsampling_signal: str = 'subgroup_label',
                     load_small_images: bool = False,
                     **additional_builder_kwargs) -> Dataloader:
  """Returns datasets for training, validation, and possibly test sets.

  Args:
    num_splits: Integer for number of slices of the dataset.
    initial_sample_proportion: Float for proportion of entire training dataset
      to sample initially before active sampling begins.
    subgroup_ids: List of strings of IDs indicating subgroups.
    subgroup_proportions: List of floats indicating proportion that each
      subgroup should take in initial training dataset.
    tfds_dataset_name: The name of the tfd dataset to load from.
    data_dir: Default data directory to store the sampled data.
    include_train_sample: Whether to include the `train_sample` split.
    labeled_train_pattern: File pattern for labeled training data.
    unlabeled_train_pattern: File pattern for unlabeled training data.
    validation_pattern: File pattern for validation data.
    use_post_disaster_only: Whether to use post-disaster imagery only rather
      than full 6-channel stacked image input.
    upsampling_lambda: Number of times subgroup examples should be repeated.
    upsampling_signal: Signal to use to determine subgroup to upsample.
    load_small_images: A flag controls loading small images or not.
    **additional_builder_kwargs: Additional keyword arguments to data builder.

  Returns:
    A tuple containing the split training data, split validation data, the
    combined training dataset, and a dictionary mapping evaluation dataset names
    to their respective combined datasets.
  """

  builder_kwargs = {
      'subgroup_ids': subgroup_ids,
      'subgroup_proportions': subgroup_proportions,
      'include_train_sample': include_train_sample,
      **additional_builder_kwargs
  }
  if '/' not in tfds_dataset_name:
    # No named config variant specified, so provide the config explicitly.
    # pylint: disable=unexpected-keyword-arg
    builder_kwargs['config'] = SkaiDatasetConfig(
        name='skai_dataset',
        labeled_train_pattern=labeled_train_pattern,
        labeled_test_pattern=validation_pattern,
        unlabeled_pattern=unlabeled_train_pattern,
        use_post_disaster_only=use_post_disaster_only,
        load_small_images=load_small_images,
    )
    # pylint: enable=unexpected-keyword-arg
  split_size_in_pct = int(100 * initial_sample_proportion / num_splits)
  reduced_datset_sz = int(100 * initial_sample_proportion)

  val_splits = tfds.load(
      tfds_dataset_name,
      split=[
          f'labeled_test[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, reduced_datset_sz, split_size_in_pct)
      ],
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False)

  train_splits = tfds.load(
      tfds_dataset_name,
      split=[
          f'labeled_train[{k}%:{k+split_size_in_pct}%]'
          for k in range(0, reduced_datset_sz, split_size_in_pct)
      ],
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False)

  # TODO(jihyeonlee): Utilize unlabeled data.

  # No separate test set, so use validation for now.
  test_ds = tfds.load(
      tfds_dataset_name,
      split='labeled_test',
      data_dir=data_dir,
      builder_kwargs=builder_kwargs,
      try_gcs=False,
      with_info=False)

  train_ds = gather_data_splits(list(range(num_splits)), train_splits)
  val_ds = gather_data_splits(list(range(num_splits)), val_splits)
  eval_datasets = {
      'val': val_ds,
      'test': test_ds,
  }
  subgroup_sizes = get_subgroup_sizes(train_ds)
  if upsampling_lambda > 1:
    train_ds = upsample_subgroup(
        train_ds, upsampling_lambda, upsampling_signal, subgroup_sizes
    )
  return Dataloader(
      2,
      subgroup_sizes,
      train_splits,
      val_splits,
      train_ds,
      train_sample_ds=None,
      eval_ds=eval_datasets)
