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

"""Util functions to handle privileged information in ImageNet-PI."""

from typing import Callable, Dict, Tuple

import tensorflow as tf
from uncertainty_baselines.datasets.privileged_information import Example

EncodingFn = Callable[[Example], tf.Tensor]


def get_privileged_information_fn(pi_subset: Tuple[str, ...],
                                  encoding_fn_dict: Dict[str, EncodingFn]):
  """Gets function that generates PI features by combining attributes from `pi_subset`.

  The returned `privileged_information_fn` concatenates the attributes in
  `pi_subset` by encoding them using the encoding functions in
  `encoding_fn_dict`. An `EncodingFn` takes an Example, including `pi_features`,
  and returns a derived tf.Tensor from one or more of its fields.

  Args:
    pi_subset: Set of keys that compose the privileged_information. Each key
      must have a matching encoding function in `encoding_fn_dict`.
    encoding_fn_dict: Dictionary of encoding functions for each possible
      pi_subset element. This function raises a `ValueError` if a key in
      `pi_subset` is not found in `encoding_fn_dict`.

  Returns:
    Function that generates PI.
  """

  encoding_fn_dict = encoding_fn_dict if encoding_fn_dict else {}

  def _pi_fn(example):
    """Function to generate PI."""
    any_pi_feature = list(example['pi_features'].values())[0]
    batch_size = tf.shape(any_pi_feature)[0]
    num_annotators = tf.shape(any_pi_feature)[1]

    pi_features = []
    for pi_key in pi_subset:
      if pi_key not in encoding_fn_dict:
        raise ValueError(
            f'{pi_key} not found in {encoding_fn_dict}. Each key in `pi_subset`'
            ' must have a corresponding encoding function.'
        )
      encoding_fn = encoding_fn_dict[pi_key]
      pi_feature = encoding_fn(example)

      pi_features.append(
          tf.cast(
              tf.reshape(pi_feature, [batch_size, num_annotators, -1]),
              tf.float32))

    return tf.concat(pi_features, axis=-1)

  return _pi_fn


def flatten_annotator_axis(tensor):
  """Flattens the annotator axis in a tensor, merging it with the batch axis.

  tensor: (batch_size, num_annotators, feature_length) ->
          (batch_size * num_annotators, feature_length)
  Args:
    tensor: A rank-3 Tensor for which to flatten the annotator axis.

  Returns:
    The flattened tensor.
  """
  return tf.reshape(tensor, [-1, tf.shape(tensor)[-1]])


def find_noisy_annotators(examples: Example, flatten_annotators: bool = True):
  """Returns a mask with entries in {0.0, 1.0} marking the annotators whose label does not agree with the clean label of that example.

  Args:
    examples: Set of examples with pi_features on which the noisy annotators are
      to be identified.
    flatten_annotators: Whether or not to flatten the final annotator axis. If
      True, the returned mask will have shape (batch_size *
      num_annotators_per_example), and (batch_size, num_annotators_per_example)
      otherwise.
  """
  annotator_labels = tf.cast(examples['pi_features']['annotator_labels'],
                             tf.int32)
  num_annotators_per_example = tf.shape(annotator_labels)[1]
  annotator_labels = flatten_annotator_axis(annotator_labels)

  clean_labels = tf.cast(examples['clean_labels'], tf.int32)
  clean_labels = repeat_across_annotators(
      clean_labels, num_annotators_per_example=num_annotators_per_example)
  clean_labels = flatten_annotator_axis(clean_labels)

  # The `pi_features` might come padded with dummy annotators to
  # guarantee an efficient batching of the `pi_features`.
  annotator_ids = tf.reshape(examples['pi_features']['annotator_ids'], [-1])
  non_dummy_annotator_indices = tf.where(annotator_ids != -1)

  annotator_labels = tf.gather_nd(annotator_labels, non_dummy_annotator_indices)

  clean_labels = tf.gather_nd(clean_labels, non_dummy_annotator_indices)

  noisy_idx = tf.cast(
      tf.math.reduce_any(
          tf.math.not_equal(annotator_labels, clean_labels), axis=1),
      tf.float32)
  if not flatten_annotators:
    noisy_idx = tf.reshape(noisy_idx, [-1, num_annotators_per_example])
  return noisy_idx


def repeat_across_annotators(per_example_tensor: tf.Tensor,
                             num_annotators_per_example: int):
  """Repeats the unique tensor value of each example to match the number of annotators of each example.

  That is, it transforms `per_example_tensor` as follows:
     (batch_size, )  -> (batch_size, num_annotators_per_example, 1)
  or
     (batch_size, feature_length) -> (batch_size, num_annotators_per_example,
     feature_length)

  Args:
    per_example_tensor: A tensor containing a unique feature for each example
      with shape (batch_size, feature_length) or (batch_size,)
    num_annotators_per_example: The number of annotators of each example in the
      batch.

  Returns:
    The repeated feature with shape (batch_size, num_annotators_per_example,
    feature_length).
  """

  # We reshape the per_example_tensor to make sure they are (batch_size,
  # feature_length).
  per_example_tensor = tf.expand_dims(per_example_tensor, axis=1)
  batch_size = tf.shape(per_example_tensor)[0]
  feature_length = tf.shape(per_example_tensor)[-1]
  per_example_tensor = tf.reshape(per_example_tensor,
                                  [batch_size, feature_length])

  # We then copy the `per_example_tensor` to match `num_annotators_per_example`.
  return tf.reshape(
      tf.repeat(
          per_example_tensor, repeats=[num_annotators_per_example], axis=0),
      [batch_size, num_annotators_per_example, feature_length])


def annotator_ids_encoding_fn(example, num_dataset_annotators):
  """Encodes the annotator ids as one-hot vectors."""
  return tf.one_hot(
      tf.cast(example['pi_features']['annotator_ids'], tf.int32),
      num_dataset_annotators,
      dtype=tf.int32)


def clean_labels_encoding_fn(example,
                             num_annotators_per_example,
                             label_encoding_fn=None):
  """Encodes clean labels and repeats them for each annotator in the batch."""
  if label_encoding_fn is not None:
    clean_labels = label_encoding_fn(example['clean_labels'])
  else:
    clean_labels = example['clean_labels']
  clean_labels = repeat_across_annotators(
      clean_labels, num_annotators_per_example=num_annotators_per_example)
  return clean_labels


def annotator_label_if_incorrect_encoding_fn(example,
                                             label_encoding_fn=None,
                                             is_label_one_hot=False):
  """Returns annotator_labels if an annotator is incorrect else zero-vector."""
  is_incorrect = find_noisy_annotators(example, flatten_annotators=False)

  annotator_labels = example['pi_features']['annotator_labels']

  if is_label_one_hot:
    complementary_labels = tf.zeros_like(annotator_labels)
  else:
    # A label of -1 will be mapped to an all-zero vector after `tf.one_hot`.
    complementary_labels = -1 * tf.ones_like(annotator_labels)

  # We cast `is_incorrect` to `tf.bool` and expand the last dimension so that
  # `is_incorrect` has shape (batch_size, num_annotators_per_example, 1) and
  # `tf.where` does not broadcast any dimensions.
  is_incorrect = tf.expand_dims(tf.cast(is_incorrect, tf.bool), axis=2)

  # If an annotator is incorrect, we provide its label as PI, otherwise we
  # return an all-zero vector.
  pi_feature = tf.where(is_incorrect, annotator_labels, complementary_labels)
  if label_encoding_fn is not None:
    pi_feature = label_encoding_fn(pi_feature)
  return pi_feature


def sample_pi_features(train_builder, privileged_information_fn,
                       num_mc_samples):
  """Samples `num_mc_samples` of PI from the training set."""
  mc_ds = train_builder.load(batch_size=1).take(num_mc_samples)
  pi_mc_samples = [privileged_information_fn(example) for example in mc_ds]
  pi_mc_samples = tf.concat(pi_mc_samples, axis=0)
  return pi_mc_samples
