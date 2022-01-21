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

"""Tests for the input pipeline utilities used in the ViT experiments."""

import os
import pathlib
import tempfile

from absl import logging
from absl.testing import parameterized
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import input_utils  # local file import from baselines.jft


class InputUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    # Go two directories up to the root of the UB directory.
    ub_root_dir = pathlib.Path(__file__).parents[2]
    data_dir = str(ub_root_dir) + "/.tfds/metadata"
    logging.info("data_dir contents: %s", os.listdir(data_dir))
    self.data_dir = data_dir

  def test_get_num_examples(self):
    dataset_name = "imagenet21k"
    split = "full[:10]+full[20:24]"

    process_count = 3
    process_batch_size = 1
    num_examples_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        process_batch_size=process_batch_size,
        drop_remainder=True,
        process_count=process_count,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_drop, 12)

    num_examples_no_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        process_batch_size=process_batch_size,
        drop_remainder=False,
        process_count=process_count,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_no_drop, 14)

    process_batch_size = 3
    num_examples_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        process_batch_size=process_batch_size,
        drop_remainder=True,
        process_count=process_count,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_drop, 9)

    num_examples_no_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        process_batch_size=process_batch_size,
        drop_remainder=False,
        process_count=process_count,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_no_drop, 14)

  # TODO(dusenberrymw): tfds.testing.mock_data ignores sub-splits. File a bug so
  # that sub-splits can be fully tested with mocked data.
  # NOTE: These numbers are simply being used to test for determinism.
  @parameterized.parameters(
      (0, 1, 575047232.0, 804.0, 191682400.0, 268.0),
      (0, 3, 191682400.0, 268.0, 191682416.0, 268.0),
      (1, 3, 191682400.0, 268.0, 191682416.0, 268.0),
  )
  def test_get_data(self, process_index, process_count, correct_train_image_sum,
                    correct_train_labels_sum, correct_val_image_sum,
                    correct_val_labels_sum):
    rng = jax.random.PRNGKey(42)

    dataset = "imagenet21k"
    train_split = "full[:10]"
    val_split = "full[:10]"
    num_classes = 21843
    batch_size = 3
    shuffle_buffer_size = 20

    def _get_num_examples(ds):

      def _reduce_fn(count, batch):
        x = tf.reshape(batch["image"], [-1, 224, 224, 3])
        if "mask" in batch:
          mask = tf.reshape(batch["mask"], [-1])
          x = tf.boolean_mask(x, mask)
        return count + tf.shape(x)[0]

      return int(ds.reduce(0, _reduce_fn))

    def preprocess_fn(example):
      image = tf.io.decode_image(
          example["image"], channels=3, expand_animations=False)
      image = tf.image.resize(image, [224, 224])
      labels = tf.reduce_max(
          tf.one_hot(example["labels"], depth=num_classes), axis=0)
      return {"image": image, "labels": labels}

    rng, train_rng = jax.random.split(rng)
    process_batch_size = batch_size // process_count
    with tfds.testing.mock_data(num_examples=10, data_dir=self.data_dir):
      train_ds = input_utils.get_data(
          dataset,
          split=train_split,
          rng=train_rng,
          process_batch_size=process_batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          data_dir=self.data_dir,
          process_index=process_index,
          process_count=process_count)

      train_ds_1_epoch = input_utils.get_data(
          dataset,
          split=train_split,
          rng=train_rng,
          process_batch_size=process_batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          num_epochs=1,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          data_dir=self.data_dir,
          process_index=process_index,
          process_count=process_count)

      val_ds = input_utils.get_data(
          dataset,
          split=val_split,
          rng=None,
          process_batch_size=process_batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          num_epochs=1,
          repeat_after_batching=True,
          shuffle=False,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          drop_remainder=False,
          data_dir=self.data_dir,
          process_index=process_index,
          process_count=process_count)

    batch_dims = (jax.local_device_count(),
                  process_batch_size // jax.local_device_count())
    train_batch = next(iter(train_ds))
    self.assertEqual(train_batch["image"].shape, batch_dims + (224, 224, 3))
    self.assertEqual(train_batch["labels"].shape, batch_dims + (num_classes,))

    # Check that examples are dropped or not.
    self.assertEqual(
        _get_num_examples(train_ds_1_epoch),
        input_utils.get_num_examples(
            dataset,
            split=train_split,
            process_batch_size=process_batch_size,
            data_dir=self.data_dir))
    self.assertEqual(
        _get_num_examples(val_ds),
        input_utils.get_num_examples(
            dataset,
            split=val_split,
            process_batch_size=process_batch_size,
            drop_remainder=False,
            data_dir=self.data_dir))

    # Test for determinism.
    def reduction_fn(state, batch):
      prev_image_sum, prev_labels_sum = state
      image_sum = tf.math.reduce_sum(batch["image"])
      labels_sum = tf.math.reduce_sum(batch["labels"])
      return (image_sum + prev_image_sum, labels_sum + prev_labels_sum)

    train_image_sum, train_labels_sum = train_ds.take(10).reduce((0., 0.),
                                                                 reduction_fn)
    val_image_sum, val_labels_sum = val_ds.take(10).reduce((0., 0.),
                                                           reduction_fn)
    logging.info(
        "(train_image_sum, train_labels_sum, val_image_sum, "
        "val_labels_sum) = %s, %s, %s, %s", float(train_image_sum),
        float(train_labels_sum), float(val_image_sum), float(val_labels_sum))

    self.assertAllClose(train_image_sum, correct_train_image_sum)
    self.assertAllClose(train_labels_sum, correct_train_labels_sum)
    self.assertAllClose(val_image_sum, correct_val_image_sum)
    self.assertAllClose(val_labels_sum, correct_val_labels_sum)


if __name__ == "__main__":
  tf.test.main()
