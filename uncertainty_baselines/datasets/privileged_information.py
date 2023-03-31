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

"""Base dataset builders dealing with privileged_information."""

import abc
import contextlib
from typing import Any, Dict, Optional, Tuple, TypedDict, Union

from robustness_metrics.common import types
import tensorflow as tf
from uncertainty_baselines.datasets import base

EMPTY_KEY = -2
DELETED_KEY = -3

NoPIFeatures = types.Features


class PIFeatures(TypedDict):
  # All fields must have shape (num_annotators, ...) and dtype: tf.float32.
  annotator_labels: tf.Tensor
  annotator_ids: tf.Tensor
  # Possible additional fields.


class Example(TypedDict):
  pi_features: PIFeatures
  clean_labels: tf.Tensor  # shape: label_shape dtype: tf.float32.
  # Possible additional fields from NoPIFeatures.


def _estimate_correct_per_annotator(annotators_ids, is_correct):
  """Estimates the number of correct examples seen by the annotators in a batch.

  The function operates at the level of individual annotations, where
  each element corresponds to one annotation event given by (annotator_id,
  correctness). Usually, the input to this function will come from the
  flattening of the annotator and batch indices from a batch of examples.

  Args:
    annotators_ids: 1-D int32 Tensor containing a list of annotator_ids. The
      annotator_ids can be repeated.
    is_correct: 1-D int32 Tensor with the same length as annotators_ids,
      indicating whether the corresponding annotator was correct on a particular
      example.

  Returns:
    The set of unique ids in `annotators_ids` and two tensors of the same length
    indicating the number of examples seen by each annotator in the batch and
    the number of times the annotator was correct in the batch.
  """
  unique_ids, indices, counts = tf.unique_with_counts(annotators_ids)
  iterating_indices = tf.range(tf.shape(unique_ids)[0])

  def _sum_correct(idx):
    correct_count_indices = tf.where(tf.equal(indices, idx))
    return tf.reduce_sum(tf.gather(is_correct, correct_count_indices))

  return unique_ids, tf.map_fn(
      _sum_correct, iterating_indices, dtype=tf.int32), counts


def _gather_annotator_features(example: Example,
                               annotator_idx: tf.Tensor) -> Example:
  """Gathers the features from the annotators with index in `annotator_idx`.

  Args:
    example: An example with PI from a set of annotators.
    annotator_idx: An int32 tensor with shape (num_sampled_annotators, )
      indicating which indices to gather from the annotator features.

  Returns:
    The example with the PI from the gathered annotators.
  """
  annotator_features = example['pi_features']
  example['pi_features'] = {
      feature_key: tf.gather(feature_value, annotator_idx) for feature_key,
      feature_value in annotator_features.items()  # type: ignore
  }
  return example  # type: ignore


def _in_subrange(interval, valid_range):
  return interval[0] < interval[1] and interval[0] >= valid_range[
      0] and interval[0] < valid_range[1] and interval[1] > valid_range[
          0] and interval[1] <= valid_range[1]


class AnnotatorPIMixin(base.BaseDataset, abc.ABC):
  """Abstract mixin class to allow access to PI from a pool of annotators.

  The pattern of inheritance should follow:

  class SomeDataset(base.BaseDataset):
    ...

  class SomePIDataset(AnnotatorPIMixin, SomeDataset):
    def _load_pi_features(self, example: types.Features) -> AnnotatorPIFeatures:
      # Load pi_features of example.
      example['pi_features']['annotator_labels'] = ...
      example['pi_features']['annotator_ids'] = ...
      example['clean_labels'] = ...
      return example

    def pi_feature_length(self):
      return {'annotator_labels': ..., 'annotator_ids': 1, ...}
  """

  def __init__(self,
               num_annotators_per_example: Optional[int] = None,
               num_annotators_per_example_and_step: Optional[int] = None,
               reliability_interval: Optional[Tuple[float, float]] = (0.0, 1.0),
               artificial_id_increase_factor: Optional[int] = None,
               reliability_estimation_batch_size: int = 1,
               pi_seed: Optional[Union[int, tf.Tensor]] = None,
               num_adversarial_annotators_per_example: Optional[int] = 0,
               annotator_sampling_strategy: Optional[str] = 'uniform',
               random_pi_length: Optional[int] = None,
               disable_reliability_estimation: bool = False,
               **kwargs):
    """Creates AnnotatorPIDataset builder.

    Args:
      num_annotators_per_example: The number of annotators per example that are
        subsampled out of the total pool of annotators available for that
        example. If None, the maximum number of available annotators per example
        is returned.
      num_annotators_per_example_and_step: The number of annotators that are
        dynamically subsampled at each batch out of the
        `num_annotators_per_example` from that example. If None, no dynamic
        subsampling of annotators is performed.
      reliability_interval: Interval used to filter out annotators that fall
        outside of its range. It must be a subrange of [0.0, 1.0].
      artificial_id_increase_factor: Number of times the set of available ids
        will be artificially increased. It can be used to test the effect of
        altering the number of examples seen by each annotator, allowing to
        artifically decrease by this factor the number of examples seen by each
        annotator. If None, uses only the original annotators.
      reliability_estimation_batch_size: Number of examples to load at once when
        estimating the reliability of the annotators in the dataset. This is the
        batch size used to estimate the reliability once. It is only used once,
        prior to training. The reliability_estimation_batch_size would be mostly
        limited by CPU RAM and not accelerator memory, as the reliability
        estimation is done prior to training.
      pi_seed: Random seed controlling the randomness in the annotator sampling
        and processing.
      num_adversarial_annotators_per_example: Number of adversarial annotators
        to add to the pool of annotators of each example. An adversarial
        annotator is an annotator that assigns a label uniformly at random for
        each example.
      annotator_sampling_strategy: Strategy used to sample the desired
        `num_annotators_per_example`. To choose from `best`, `worst`, or
        `uniform`.
      random_pi_length: Length of an optional random PI field sampled from a
        normal distribution once for every example. If None the random PI is not
        loaded.
      disable_reliability_estimation: Whether to disable the expensive
        reliability estimation step when creating the dataset. This argument
        must be set to False if `num_adversarial_annotators_per_example > 0` or
        the `reliability_interval != (0., 1.)` as these options require
        computing precise statistics about each annotator to work.
      **kwargs: Any other dataset-specific configuration options.
    """

    # Dictionary of HashTables containing information about each annotator
    # indexed by `annotator_id`. It should contain the number of seen examples
    # by each annotator (i.e, `count`), the number of correctly annotated
    # examples (i.e. `correct_count`) and a flag indicating if that annotator's
    # reliability is within the `reliability_interval` (i.e., `accepted``).
    self._annotators_info = {
        'accepted':
            tf.lookup.experimental.DenseHashTable(
                key_dtype=tf.int32,
                value_dtype=tf.int32,
                default_value=1,
                empty_key=EMPTY_KEY,
                deleted_key=DELETED_KEY)
    }
    self._num_annotators_per_example = num_annotators_per_example
    self._num_annotators_per_example_and_step = (
        num_annotators_per_example_and_step
    )

    if not _in_subrange(reliability_interval, (0.0, 1.0)):
      raise ValueError('reliability_interval must fall within [0, 1].')

    # Stateless random ops require a (2,) shaped seed.
    if pi_seed is None:
      self._pi_seed = tf.random.uniform((2,), maxval=int(1e10), dtype=tf.int32)
    elif isinstance(pi_seed, int):
      self._pi_seed = (pi_seed, pi_seed + 1)
    elif isinstance(pi_seed, tf.Tensor) and tf.shape(pi_seed).shape == 0:
      self._pi_seed = tf.stack([pi_seed, pi_seed + 1])
    else:
      self._pi_seed = pi_seed

    self._reliability_interval = reliability_interval
    self._artificial_id_increase_factor = artificial_id_increase_factor
    self._reliability_estimation_buffer_size = reliability_estimation_batch_size
    self._num_adversarial_annotators_per_example = (
        num_adversarial_annotators_per_example
    )

    self._annotator_sampling_strategy = annotator_sampling_strategy

    self._random_pi_length = random_pi_length

    self._average_annotator_load = None
    self._num_effective_annotators = None

    super().__init__(**kwargs)

    if not disable_reliability_estimation:
      self._setup_annotator_tables()
    elif num_adversarial_annotators_per_example > 0:  # pytype: disable=unsupported-operands
      raise ValueError(
          '`num_adversarial_annotators_per_example > 0` requires setting'
          ' `disable_reliability_estimation=False`'
      )
    elif reliability_interval != (0.0, 1.0):
      raise ValueError(
          '`reliability_interval != (0.0, 1.0)` requires setting'
          ' `disable_reliability_estimation=False`'
      )

  @abc.abstractmethod
  def _process_pi_features_and_labels(
      self,
      example: NoPIFeatures,
      unprocessed_example: Optional[NoPIFeatures] = None) -> Example:
    """Abstract method defining the interface to load the metadata and asign clean labels of an example.
    """
    pass

  @property
  @abc.abstractmethod
  def pi_feature_length(self) -> Dict[str, int]:
    """Returns a dictionary with the same keys as the PI features defining the length of each pi_feature.
    """
    pass

  @property
  @abc.abstractmethod
  def num_dataset_annotators(self):
    """Returns the total number of annotators in the dataset before any processing.
    """
    pass

  def _setup_annotator_tables(self):
    # First, we pass over the data once without adding any adversarial
    # annotators, nor artificially increasing the ids, to obtain a first
    # estimate of the real annotators' reliability.
    with self._disable_dynamic_processing():
      self._compute_annotators_reliability()
    self._update_reliability_constants()

    if self._reliability_interval != (0.0, 1.0):
      self._filter_by_reliability()
      self._update_reliability_constants()

    # After filtering, we do a second pass to estimate the final reliability,
    # and collect the annotator_ids of all the annotators (real and artificial).
    with self._disable_per_step_sampling():
      self._compute_annotators_reliability()
    self._update_reliability_constants()

  @property
  @abc.abstractmethod
  def _max_annotators_per_example(self):
    """Returns the maximum number of available annotators per example, computed across all the examples in the dataset.
    """
    pass

  def _hash_fingerprint_int(self, fingerprint: Any) -> int:
    """Returns an integer hash code that uniquely identifies the fingerprint."""

    # The default behaviour in BaseDataset already gives that unique int code.
    return fingerprint

  def _remap_annotator_ids(self, within_chunk_annotator_ids: tf.Tensor,
                           example_idx: int, num_chunks: int,
                           num_new_annotators_per_chunk: int) -> tf.Tensor:
    """Transforms the `within_chunk_annotator_ids` to a set of new `annotator_ids` that do not collide among the `num_chunks` equal chunks of the dataset.

    This method splits the dataset into `num_chunks` unique chunks with the
    same number of examples, and transforms the `within_chunk_annotator_ids`
    which uniquely identifies the annotators within that chunk, and remaps them
    into a new set of `annotator_ids` which are unique to each chunk.
    This method is useful when one wants to artificially increase the
    `num_dataset_annotators`, equivalently reducing the effective number of
    examples seen by each annotator_id, e.g., its 'annotator_load'. To do this
    with O(1) memory complexity, it partitions the whole dataset into
    `num_chunks` chunks depending on `example_idx` and adds a constant shift to
    all the annotator ids in that chunk, so that they become unique.

    Args:
      within_chunk_annotator_ids: The `annotator_ids` that uniquely identify
        each annotator within a chunk. These `annotator_ids` are spread accross
        the dataset assigning them unique codes at each chunk.
      example_idx: A hash code or index that uniquely identifies this example
        within the BaseDatset.
      num_chunks: The number of times each id out of the `num_new_annotators`
        will be replicated across the dataset.
      num_new_annotators_per_chunk: The total number of unique annotators that
        will be added to the annotator pool. This number will be expanded by a
        factor `num_chunks` after its artificial increase.

    Returns:
      The new set of remapped annotators_ids.
    """
    example_chunk_idx = example_idx % num_chunks
    new_annotator_ids = within_chunk_annotator_ids + tf.cast(
        example_chunk_idx * num_new_annotators_per_chunk, tf.float32)
    return new_annotator_ids

  def _create_process_example_fn(self) -> base.PreProcessFn:
    """Creates a pre-process function wrapping the _create_process_example_fn from the parent BaseDataset class.

    Returns:
      example_parser: Function to process an example and load its
        privileged_information.
    """

    # super in this case calls the next sibling in the MRO.
    preproc_fn = super()._create_process_example_fn()

    def _example_parser(example: NoPIFeatures) -> Example:
      parsed_example = preproc_fn(example)
      pi_parsed_example = self._process_pi_features_and_labels(
          parsed_example, unprocessed_example=example)

      example_idx = self._hash_fingerprint_int(example[self._fingerprint_key])
      # Filter annotators by reliability.
      annotator_ids = tf.cast(pi_parsed_example['pi_features']['annotator_ids'],
                              tf.int32)[:, 0]  # Remove dummy feature dim.
      accepted_mask = self._annotators_info['accepted'][annotator_ids]
      accepted_idx = tf.reshape(tf.where(accepted_mask), [-1])
      pi_parsed_example = _gather_annotator_features(pi_parsed_example,
                                                     accepted_idx)

      per_example_seed = tf.random.experimental.stateless_fold_in(
          self._pi_seed, example_idx)
      if self._num_adversarial_annotators_per_example > 0:
        pi_parsed_example = self._append_adversarial_annotators(
            pi_parsed_example, per_example_seed=per_example_seed)

      if self._random_pi_length is not None:
        if self._random_pi_length <= 0:
          raise ValueError('random_pi_length must be greater than 0.')
        pi_parsed_example['pi_features'][  # type: ignore
            'random_pi'] = self._create_random_pi(
                pi_parsed_example, per_example_seed=per_example_seed)

      if self._artificial_id_increase_factor:
        # Artificially multiply the set of ids.
        annotator_ids = pi_parsed_example['pi_features']['annotator_ids']
        annotator_ids = self._remap_annotator_ids(
            within_chunk_annotator_ids=annotator_ids,
            example_idx=example_idx,
            num_chunks=self._artificial_id_increase_factor,
            num_new_annotators_per_chunk=self.num_dataset_annotators)
        pi_parsed_example['pi_features']['annotator_ids'] = annotator_ids

      # Subsample annotators.
      if self._num_annotators_per_example:
        per_example_seed = tf.random.experimental.stateless_fold_in(
            self._pi_seed, example_idx)
        pi_parsed_example = self._subsample_annotators(
            pi_parsed_example, per_example_seed,
            self._annotator_sampling_strategy, self._num_annotators_per_example)
      if self._num_annotators_per_example_and_step:
        per_example_step_seed = tf.random.experimental.stateless_fold_in(
            self._pi_seed, example[self._enumerate_id_key])
        pi_parsed_example = self._subsample_annotators(
            pi_parsed_example, per_example_step_seed, 'uniform',
            self._num_annotators_per_example_and_step)

      pi_parsed_example = self._pad_annotators(pi_parsed_example)

      return pi_parsed_example

    return _example_parser

  def _subsample_annotators(self, example: Example, seed: tf.Tensor,
                            strategy: str, num_max_annotators: int) -> Example:
    """Subsamples `num_max_annotators` out of all the annotators available in example['pi_features'] using the desired `strategy`.

    Possible strategies are: `uniform`, `best` and `worst`, where `uniform`
    represents sampling the `num_max_annotators` uniformly at random, and `best`
    and `worst` as many correct/incorrect annotators as possible for a given
    sample, respectively. Note, that correct in this case refers to the fact
    that the `annotator_label` matches the corresponding `clean_label` on this
    example, and does not take into account the overal reliability of the
    annotator across the dataset.`

    Args:
      example: The example with the pi_features to sample from.
      seed: Random tensorflow seed.
      strategy: Sampling strategy. To select from `uniform`, `best`, or `worst`.
      num_max_annotators: Number of annotators to sample.

    Returns:
      The example with the `pi_features` of the subsampled annotators.
    """
    annotator_features = example['pi_features']
    annotator_labels = tf.cast(annotator_features['annotator_labels'], tf.int32)
    num_available_annotators = tf.shape(annotator_labels)[0]

    # Repeat clean_labels to match number of annotator_labels.
    clean_labels = tf.expand_dims(
        tf.cast(example['clean_labels'], tf.int32), axis=0)
    clean_labels = tf.repeat(
        clean_labels, axis=0, repeats=[num_available_annotators])

    is_correct = tf.cast(
        tf.reduce_all(tf.math.equal(clean_labels, annotator_labels), axis=1),
        tf.int32)

    if strategy == 'uniform':
      selected_annotators = tf.random.experimental.stateless_shuffle(
          tf.range(num_available_annotators), seed)[:num_max_annotators]
    elif strategy == 'best':
      selected_annotators = tf.argsort(tf.reshape(is_correct,
                                                  [-1]))[-num_max_annotators:]
    elif strategy == 'worst':
      selected_annotators = tf.argsort(tf.reshape(is_correct,
                                                  [-1]))[:num_max_annotators]

    example = _gather_annotator_features(
        example=example, annotator_idx=selected_annotators)

    return example

  @property
  def num_global_adversarial_annotators(self):
    """Returns the number of adversarial annotators that are added to the global pool of annotators.
    """
    if (
        self._num_adversarial_annotators_per_example is None
        or self._num_adversarial_annotators_per_example == 0
    ):
      return 0
    else:
      return int(
          round(self._num_adversarial_annotators_per_example *
                self.num_examples / self.average_annotator_load))

  def _append_adversarial_annotators(self, example: Example,
                                     per_example_seed: tf.Tensor) -> Example:
    """It appends num_adversarial_annotators_per_example to the total number of annotators loaded on that example.
    """

    def _stack_adversarial_features(adversarial_value, pi_key):
      pi_features = example['pi_features'][pi_key]  # type: ignore
      return tf.concat([pi_features, adversarial_value], axis=0)

    adversarial_pi_features = self._set_adversarial_pi_features(
        example, per_example_seed)

    # We sample `num_adversarial_annotators_per_example` out of
    # `num_global_adversarial_annotators` to guarantee that the added
    # adversarial annotators have the same `average_annotator_load` as the rest
    # of the dataset.
    adversarial_ids = tf.random.experimental.stateless_shuffle(
        tf.range(
            self.num_dataset_annotators, self.num_dataset_annotators +
            self.num_global_adversarial_annotators),
        per_example_seed)[:self._num_adversarial_annotators_per_example]

    adversarial_pi_features['annotator_ids'] = tf.reshape(  # type: ignore
        tf.cast(adversarial_ids, tf.float32),
        [self._num_adversarial_annotators_per_example, 1])

    if adversarial_pi_features is not None:
      for pi_key in adversarial_pi_features.keys():
        example['pi_features'][
            pi_key] = _stack_adversarial_features(  # type: ignore
                adversarial_pi_features[pi_key], pi_key)  # type: ignore

      return example
    else:
      return example

  @abc.abstractmethod
  def _set_adversarial_pi_features(
      self, example: Example,
      per_example_seed: tf.Tensor) -> Union[PIFeatures, None]:
    """Assigns a value to the pi_features of the adversarial annotators except the `annotator_ids`.
    """
    pass

  def _pad_annotators(self, example: Example) -> Example:
    """Pads pi_features with -1 so that each example has features for exactly num_annotators_per_example_and_step.

    An annotator_id=-1 indicates that the corresponding pi_features are
    meaningless.

    Args:
      example: Example with PI features to pad.

    Returns:
      The example with the padded PI features.
    """
    num_available_annotators = tf.shape(
        example['pi_features']['annotator_ids']
    )[0]

    padding_length = (
        self.num_annotators_per_example_and_step - num_available_annotators
    )
    paddings = tf.convert_to_tensor([[0, padding_length], [0, 0]])

    def _pad_pi_feature(pi_value, pi_feature_length):
      padded_value = tf.pad(pi_value, paddings, 'CONSTANT', constant_values=-1)

      return tf.reshape(
          padded_value,
          [self.num_annotators_per_example_and_step, pi_feature_length])

    pi_features = {
        pi_key: _pad_pi_feature(pi_value, self.pi_feature_length[pi_key])
        for pi_key, pi_value in example['pi_features'].items()  # type: ignore
    }
    example['pi_features'] = pi_features  # type: ignore
    return example  # type: ignore

  def _create_random_pi(self, example: Example,
                        per_example_seed: tf.Tensor) -> tf.Tensor:
    """Creates a unique random PI feature vector of length `random_pi_length` sampled from a normal distribution.
    """
    num_available_annotators = tf.shape(
        example['pi_features']['annotator_ids'])[0]
    return tf.random.stateless_normal(
        (num_available_annotators, self._random_pi_length), per_example_seed)

  @property
  def num_effective_annotators(self):
    """Returns the number of effective annotators in the dataset.

    An effective annotator is any real annotator that has been accepted after
    filtering, as well as any annotator that has been artificially included
    afterwards, i.e., an adversarial annotator.
    """
    if self._num_effective_annotators is None:
      self._num_effective_annotators = self.annotators_info['count'].size()
    return self._num_effective_annotators

  @property
  def annotators_info(self):
    """Returns a dictionary with the total number of seen and correct examples per annotator.
    """
    if 'count' not in self._annotators_info.keys():
      self._compute_annotators_reliability()
    return self._annotators_info

  def _compute_average_annotator_load(self):
    """Computes the average load of the annotators in the dataset."""
    total_count = 0.
    annotator_ids = self.get_effective_annotators_ids()
    for n in range(self.num_effective_annotators):
      total_count += int(self.annotators_info['count'][annotator_ids[n]])
    return total_count / float(self.num_effective_annotators)

  @property
  def average_annotator_load(self):
    if self._average_annotator_load is None:
      self._average_annotator_load = self._compute_average_annotator_load()
    return self._average_annotator_load

  def get_effective_annotators_ids(self):
    """Returns a 1-D int32 Tensor listing all effective annotator_ids."""
    ids, _ = self.annotators_info['count'].export()
    unique_ids, _ = tf.unique(tf.reshape(ids, [-1]))

    # NOTE: The export function returns the indices of the empty buckets in the
    # DenseHashTable as well, so we need to remove them.
    non_empty_ids_indices = tf.where(
        tf.math.logical_and(
            tf.not_equal(unique_ids, EMPTY_KEY),
            tf.not_equal(unique_ids, DELETED_KEY)))
    return tf.gather(unique_ids, non_empty_ids_indices)

  @property
  def num_annotators_per_example(self):
    """Returns the number of annotators that are loaded per example."""
    if self._num_annotators_per_example is None:
      return self._max_annotators_per_example
    else:
      return self._num_annotators_per_example

  @property
  def num_annotators_per_example_and_step(self):
    """Returns the number of annotators that are subsampled per example at each step.
    """
    if self._num_annotators_per_example_and_step is None:
      return self.num_annotators_per_example
    else:
      return self._num_annotators_per_example_and_step

  @property
  def annotators_reliability(self):
    """Returns a dictionary of pairs (annotator_id, reliability)."""

    annotator_ids = self.get_effective_annotators_ids()

    return {
        int(annotator_ids[n]):
        self._estimate_annotator_reliability(annotator_ids[n])
        for n in range(self.num_effective_annotators)
    }

  def _estimate_annotator_reliability(self, annotator_id):
    return tf.math.divide_no_nan(
        tf.cast(self.annotators_info['correct_count'][annotator_id],
                tf.float32),
        tf.cast(self.annotators_info['count'][annotator_id], tf.float32),
    )

  @property
  def mean_reliability(self):
    """Estimates the average reliability of the annotators in the dataset."""
    total_count = 0.
    total_correct_count = 0.
    annotator_ids = self.get_effective_annotators_ids()
    for n in range(self.num_effective_annotators):
      total_count += int(self.annotators_info['count'][annotator_ids[n]])
      total_correct_count += int(
          self.annotators_info['correct_count'][annotator_ids[n]])
    return total_correct_count / float(total_count)

  @property
  def max_annotator_id(self):
    """Returns the value of the highest annotator id."""
    return (self.num_dataset_annotators + self.num_global_adversarial_annotators
           ) * self._artificial_id_increase_factor

  @contextlib.contextmanager
  def _disable_dynamic_processing(self):
    num_annotators_per_example = self._num_annotators_per_example
    num_annotators_per_example_and_step = (
        self._num_annotators_per_example_and_step
    )
    num_adversarial_annotators_per_example = (
        self._num_adversarial_annotators_per_example
    )
    artificial_id_increase_factor = self._artificial_id_increase_factor
    num_adversarial_annotators_per_example = (
        self._num_adversarial_annotators_per_example
    )
    drop_remainder = self._drop_remainder
    try:
      self._num_annotators_per_example = None
      self._num_annotators_per_example_and_step = None
      self._artificial_id_increase_factor = None
      self._num_adversarial_annotators_per_example = 0
      self._drop_remainder = False
      yield num_annotators_per_example, num_annotators_per_example_and_step, artificial_id_increase_factor, num_adversarial_annotators_per_example, drop_remainder
    finally:
      self._num_annotators_per_example = num_annotators_per_example
      self._num_annotators_per_example_and_step = (
          num_annotators_per_example_and_step
      )
      self._artificial_id_increase_factor = artificial_id_increase_factor
      self._num_adversarial_annotators_per_example = (
          num_adversarial_annotators_per_example
      )
      self._drop_remainder = drop_remainder

  @contextlib.contextmanager
  def _disable_per_step_sampling(self):
    num_annotators_per_example_and_step = (
        self._num_annotators_per_example_and_step
    )
    drop_remainder = self._drop_remainder
    try:
      self._drop_remainder = False
      self._num_annotators_per_example_and_step = None
      yield num_annotators_per_example_and_step, drop_remainder
    finally:
      self._num_annotators_per_example_and_step = (
          num_annotators_per_example_and_step
      )
      self._drop_remainder = drop_remainder

  def _compute_annotators_reliability(self):
    """Computes the reliability of each annotator in the dataset."""
    self._annotators_info['count'] = tf.lookup.experimental.DenseHashTable(
        key_dtype=tf.int32,
        value_dtype=tf.int32,
        default_value=0,
        empty_key=EMPTY_KEY,
        deleted_key=DELETED_KEY)
    self._annotators_info[
        'correct_count'] = tf.lookup.experimental.DenseHashTable(
            key_dtype=tf.int32,
            value_dtype=tf.int32,
            default_value=0,
            empty_key=EMPTY_KEY,
            deleted_key=DELETED_KEY)

    # NOTE: We bypass the default `process_batch_fn` of the dataset to prevent
    # that batch processing functions, such as mixup, influence the annotator
    # reliability estimation.
    dataset_iterator = iter(
        self.load(
            batch_size=self._reliability_estimation_buffer_size,
            process_batch_fn=lambda batch: batch))
    step_remainder = int(
        self.num_examples % self._reliability_estimation_buffer_size != 0
    )
    num_steps = (
        self.num_examples // self._reliability_estimation_buffer_size
        + step_remainder
    )
    for _ in range(num_steps):
      example = next(dataset_iterator)
      self._update_annotator_count(example)

  def _update_annotator_count(self, example: Example):
    """Updates the annotator's count in _annotators_info."""
    annotators_ids = tf.cast(
        tf.reshape(example['pi_features']['annotator_ids'], [-1]),
        dtype=tf.int32)

    # Repeat clean_labels to match number of annotator_labels.
    clean_labels = tf.expand_dims(example['clean_labels'], 1)
    label_length = tf.shape(clean_labels)[-1]  # clean_labels could be one_hot.
    clean_labels = tf.repeat(
        clean_labels, axis=1, repeats=[self.num_annotators_per_example])
    clean_labels = tf.cast(
        tf.reshape(clean_labels, [-1, label_length]), tf.int32)

    annotator_labels = tf.cast(
        tf.reshape(example['pi_features']['annotator_labels'],
                   [-1, label_length]), tf.int32)

    is_correct = tf.cast(
        tf.reduce_all(tf.math.equal(clean_labels, annotator_labels), axis=1),
        tf.int32)
    unique_ids, correct_counts, counts = _estimate_correct_per_annotator(
        annotators_ids, is_correct)

    annotators_count = self._annotators_info['count'][unique_ids] + counts
    annotators_correct_count = self._annotators_info['correct_count'][
        unique_ids] + correct_counts

    self._annotators_info['count'].insert(
        unique_ids, tf.cast(annotators_count, dtype=tf.int32))
    self._annotators_info['correct_count'].insert(
        unique_ids, tf.cast(annotators_correct_count, dtype=tf.int32))

    # Delete the entry of the padded annotators_ids
    self._annotators_info['count'].erase(tf.constant([-1]))
    self._annotators_info['correct_count'].erase(tf.constant([-1]))

  def _filter_by_reliability(self):
    """Filters the annotator set such that only those annotators with a reliability in `reliability_interval` are used.

    Note that only the real annotators are filtered. The adversarial annotators
    can have reliabilities outside of the `reliability_interval`.
    """

    for annotator_id, reliability in self.annotators_reliability.items():
      annotator_accepted = int(reliability >= self._reliability_interval[0] and
                               reliability <= self._reliability_interval[1])

      self._annotators_info['accepted'].insert(annotator_id, annotator_accepted)

      # Remove element from the annotators record.
      if not annotator_accepted:
        self._annotators_info['count'].erase(tf.constant([annotator_id]))
        self._annotators_info['correct_count'].erase(
            tf.constant([annotator_id]))

  def _update_reliability_constants(self):
    """Updates the attributes derived from the annotators_reliability.

    This method should be called every time the reliability is updated, through
    the removal or inclusion of annotators in the annotator pool. It allows to
    store a static copy of some derived attributes that are used in graph mode
    inside _exmaple_parser, and that we could not access otherwise.
    """
    self._num_effective_annotators = int(self.annotators_info['count'].size())
    self._average_annotator_load = self._compute_average_annotator_load()
