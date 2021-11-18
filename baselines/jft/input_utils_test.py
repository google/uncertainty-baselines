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

"""Tests for the input pipeline utilities used in the ViT experiments."""

import os
import pathlib
import tempfile

from absl import logging
from absl.testing import parameterized
import jax
import tensorflow as tf
import tensorflow_datasets as tfds
import input_utils  # local file import


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

    num_hosts = 3
    host_batch_size = 1
    num_examples_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        host_batch_size=host_batch_size,
        drop_remainder=True,
        num_hosts=num_hosts,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_drop, 12)

    num_examples_no_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        host_batch_size=host_batch_size,
        drop_remainder=False,
        num_hosts=num_hosts,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_no_drop, 14)

    host_batch_size = 3
    num_examples_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        host_batch_size=host_batch_size,
        drop_remainder=True,
        num_hosts=num_hosts,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_drop, 9)

    num_examples_no_drop = input_utils.get_num_examples(
        dataset_name,
        split=split,
        host_batch_size=host_batch_size,
        drop_remainder=False,
        num_hosts=num_hosts,
        data_dir=self.data_dir)
    self.assertEqual(num_examples_no_drop, 14)

  def test_get_data(self):
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
    with tfds.testing.mock_data(num_examples=10, data_dir=self.data_dir):
      train_ds = input_utils.get_data(
          dataset,
          split=train_split,
          rng=train_rng,
          host_batch_size=batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          data_dir=self.data_dir)
      train_ds_1_epoch = input_utils.get_data(
          dataset,
          split=train_split,
          rng=train_rng,
          host_batch_size=batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          num_epochs=1,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          data_dir=self.data_dir)
      val_ds = input_utils.get_data(
          dataset,
          split=val_split,
          rng=None,
          host_batch_size=batch_size,
          preprocess_fn=preprocess_fn,
          cache="loaded",
          num_epochs=1,
          repeat_after_batching=True,
          shuffle=False,
          shuffle_buffer_size=shuffle_buffer_size,
          prefetch_size=2,
          drop_remainder=False,
          data_dir=self.data_dir)

    local_batch_size = batch_size // jax.process_count()
    batch_dims = (jax.local_device_count(),
                  local_batch_size // jax.local_device_count())
    train_batch = next(iter(train_ds))
    self.assertEqual(train_batch["image"].shape, batch_dims + (224, 224, 3))
    self.assertEqual(train_batch["labels"].shape, batch_dims + (num_classes,))

    # Check that examples are dropped or not.
    self.assertEqual(
        _get_num_examples(train_ds_1_epoch),
        input_utils.get_num_examples(
            dataset,
            split=train_split,
            host_batch_size=local_batch_size,
            data_dir=self.data_dir))
    self.assertEqual(
        _get_num_examples(val_ds),
        input_utils.get_num_examples(
            dataset,
            split=val_split,
            host_batch_size=local_batch_size,
            drop_remainder=False,
            data_dir=self.data_dir))

    # Test for determinism.
    def reduction_fn(state, batch):
      prev_image_sum, prev_labels_sum = state
      image_sum = tf.math.reduce_sum(batch["image"])
      labels_sum = tf.math.reduce_sum(batch["labels"])
      return (image_sum + prev_image_sum, labels_sum + prev_labels_sum)

    image_sum, labels_sum = train_ds.take(10).reduce((0., 0.), reduction_fn)
    self.assertAllClose(image_sum, 575047232.0)
    self.assertAllClose(labels_sum, 804.)

    image_sum, labels_sum = train_ds.take(10).reduce((0., 0.), reduction_fn)
    self.assertAllClose(image_sum, 575047232.0)
    self.assertAllClose(labels_sum, 804.)


if __name__ == "__main__":
  tf.test.main()
