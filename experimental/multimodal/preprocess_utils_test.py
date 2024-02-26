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

"""Tests for preprocessing utilities."""

import io
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import preprocess_utils  # local file import from experimental.multimodal


class PreprocessUtilsTest(tf.test.TestCase):

  def get_data(self):
    return {
        "image": tf.random.uniform([640, 480, 3], 0, 255),
        "rng": tf.random.create_rng_state(42, 2)
    }

  def test_resize(self):
    data = self.get_data()
    data = preprocess_utils.Resize([120, 80])(data)
    self.assertEqual(data["image"].numpy().shape, (120, 80, 3))

  def test_resize_small(self):
    data = self.get_data()
    data = preprocess_utils.ResizeSmall(240)(data)
    self.assertEqual(data["image"].numpy().shape, (320, 240, 3))

  def test_inception_crop(self):
    data = self.get_data()
    data = preprocess_utils.InceptionCrop()(data)
    self.assertEqual(data["image"].numpy().shape[-1], 3)

  def test_decode_jpeg_and_inception_crop(self):
    f = io.BytesIO()
    plt.imsave(
        f,
        np.random.randint(0, 256, [224, 224, 3]).astype("uint8"),
        format="jpg")
    data = self.get_data()
    data["image"] = f.getvalue()
    data = preprocess_utils.DecodeJpegAndInceptionCrop()(data)
    self.assertEqual(data["image"].numpy().shape[-1], 3)

  def test_random_crop(self):
    data = self.get_data()
    data = preprocess_utils.RandomCrop([120, 80])(data)
    self.assertEqual(data["image"].numpy().shape, (120, 80, 3))

  def test_central_crop(self):
    data = self.get_data()
    data = preprocess_utils.CentralCrop([120, 80])(data)
    self.assertEqual(data["image"].numpy().shape, (120, 80, 3))

  def test_flip_lr(self):
    data = self.get_data()
    data_after_pp = preprocess_utils.FlipLr()(data)
    self.assertTrue(
        np.all(data["image"].numpy() == data_after_pp["image"].numpy()) or
        np.all(data["image"][:, ::-1].numpy() ==
               data_after_pp["image"].numpy()))

  def test_value_range(self):
    data = self.get_data()
    data = preprocess_utils.ValueRange(-0.5, 0.5)(data)

    self.assertLessEqual(np.max(data["image"].numpy()), 0.5)
    self.assertGreaterEqual(np.min(data["image"].numpy()), -0.5)

  def test_value_range_custom_input_range(self):
    data = self.get_data()
    data = preprocess_utils.ValueRange(-0.5, 0.5, -256, 255, True)(data)
    self.assertLessEqual(np.max(data["image"].numpy()), 0.5)
    self.assertGreaterEqual(np.min(data["image"].numpy()), 0.0)

  def test_keep(self):
    data = {"image": 1, "labels": 2, "something": 3}

    data_keep = preprocess_utils.Keep(["image", "labels"])(data)
    self.assertAllEqual(list(data_keep.keys()), ["image", "labels"])

  def test_onehot(self):
    data = {"labels": tf.constant(np.asarray(2), dtype=tf.int64)}
    output_data = preprocess_utils.Onehot(4, multi=True, key="labels")(data)
    self.assertAllClose(output_data["labels"].numpy(), np.asarray(
        [0., 0., 1., 0.], dtype=np.float32))

  def test_onehot_multi(self):
    data = {"labels": tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = preprocess_utils.Onehot(4, multi=False, key="labels")(data)
    self.assertAllClose(output_data["labels"].numpy(), np.asarray([
        [0., 0., 1., 0.],
        [0., 0., 0., 1.],
        [1., 0., 0., 0.]], dtype=np.float32))

    data = {"labels": tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = preprocess_utils.Onehot(4, multi=True, key="labels")(data)
    self.assertAllClose(output_data["labels"].numpy(),
                        np.asarray([1., 0., 1., 1.], dtype=np.float32))

  def test_onehot_smoothing(self):
    data = {"labels": tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = preprocess_utils.Onehot(
        4, multi=False, on=0.8, off=0.1, key="labels")(
            data)
    self.assertAllClose(output_data["labels"].numpy(), np.asarray([
        [0.1, 0.1, 0.8, 0.1],
        [0.1, 0.1, 0.1, 0.8],
        [0.8, 0.1, 0.1, 0.1]], dtype=np.float32))

    data = {"labels": tf.constant(np.asarray([2, 3, 0]), dtype=tf.int64)}
    output_data = preprocess_utils.Onehot(
        4, multi=True, on=0.8, off=0.1, key="labels")(
            data)
    self.assertAllClose(output_data["labels"].numpy(),
                        np.asarray([0.8, 0.1, 0.8, 0.8], dtype=np.float32))

  def test_get_coco_captions(self):
    data = {
        "captions": {
            "text": [tf.constant(x) for x in ["a", "b", "c", "d", "e"]]
        },
        "rng": tf.random.create_rng_state(42, 2)
    }

    output_data = preprocess_utils.GetCocoCaptions()(data)
    self.assertEqual(output_data["text"], tf.constant("b"))

  def test_clip_i1k_label_names(self):
    data = {"labels": tf.constant([0, 1])}
    output_data = preprocess_utils.ClipI1kLabelNames(key="labels")(data)
    labels = output_data["labels"].numpy().tolist()
    self.assertAllEqual(labels, ["tench", "goldfish"])

  def test_randaug(self):
    rng = tf.random.create_rng_state(42, 2)
    data = {
        "image": tf.cast(tf.clip_by_value(tf.random.stateless_uniform(
            [5, 5, 3], rng, 0, 255), 0.0, 255.0), tf.uint8),
        "rng": rng
    }
    output_data = preprocess_utils.Randaug()(data)
    expected_output = np.array(
        [[[136, 241, 255], [9, 38, 255], [255, 83, 60], [24, 255, 110],
          [229, 117, 255]],
         [[255, 255, 81], [252, 17, 0], [115, 224, 255], [255, 255, 89],
          [255, 60, 255]],
         [[146, 255, 255], [255, 100, 255], [255, 255, 255], [0, 255, 255],
          [255, 255, 255]],
         [[255, 255, 3], [255, 0, 255], [1, 0, 17], [197, 255, 255],
          [102, 58, 255]],
         [[136, 255, 68], [255, 255, 91], [255, 255, 93], [119, 184, 255],
          [255, 140, 218]]],
        dtype=np.uint8)
    self.assertAllClose(output_data["image"].numpy(), expected_output)

  def test_clip_tokenize(self):
    data = {
        "text": tf.constant("This is a test string.", dtype=tf.string),
    }

    output_data = preprocess_utils.ClipTokenize(
        key="text",
        max_len=10,
        bpe_path="third_party/py/uncertainty_baselines/experimental/multimodal/bpe_simple_vocab_16e6.txt.gz"
    )(data)
    expected_output = np.array(
        [49406, 589, 533, 320, 1628, 9696, 269, 49407, 0, 0], dtype=np.uint32)

    self.assertAllClose(output_data["text"].numpy(), expected_output)

  def test_normalize(self):
    data = {
        "image": tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.uint8),
    }

    output_data = preprocess_utils.Normalize(
        mean=[2.5, 3.5, 4.5],
        std=[0.5, 0.5, 0.5]
    )(data)
    expected_output = np.array([[-3., -3., -3.], [3., 3., 3.]],
                               dtype=np.float32)

    self.assertAllClose(output_data["image"].numpy(), expected_output)

  def test_shuffle_join(self):
    data = {
        "text": [tf.constant(x) for x in ["a", "b", "c", "d", "e"]],
        "rng": tf.random.create_rng_state(42, 2)
    }

    output_data = preprocess_utils.ShuffleJoin(
        key="text",
    )(data)
    print(output_data["text"])
    expected_output = tf.constant("e. d. c. a. b", dtype=tf.string)

    self.assertEqual(output_data["text"], expected_output)


if __name__ == "__main__":
  tf.test.main()
