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

"""Test for simple_tokenizer.py."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from simple_tokenizer import make_tokenize_fn  # local file import from experimental.multimodal
from simple_tokenizer import SimpleTokenizer  # local file import from experimental.multimodal


class SimpleTokenizerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="_short_string",
          text=tf.constant("This is a test string.", dtype=tf.string),
          tokens_expected=[49406, 589, 533, 320, 1628, 9696, 269, 49407, 0, 0]),
      dict(
          testcase_name="_longer_string",
          text=tf.constant(
              "This another test string that is longer.", dtype=tf.string),
          tokens_expected=[
              49406, 589, 1380, 1628, 9696, 682, 533, 5349, 269, 49407
          ]),
      dict(
          testcase_name="_much_longer_string",
          text=tf.constant(
              "This another test string that is longer. This another test string that is longer.",
              dtype=tf.string),
          tokens_expected=[
              49406, 589, 1380, 1628, 9696, 682, 533, 5349, 269, 589
          ]),
  )
  def test_simple(self, text, tokens_expected):
    tokenizer = SimpleTokenizer(
        bpe_path="third_party/py/uncertainty_baselines/experimental/multimodal/bpe_simple_vocab_16e6.txt.gz"
    )
    tokenize_fn = make_tokenize_fn(tokenizer, max_len=10)
    tokens = tokenize_fn(text)
    self.assertTrue(np.all(tokens == tokens_expected))


if __name__ == "__main__":
  tf.test.main()
