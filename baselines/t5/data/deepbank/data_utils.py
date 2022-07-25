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

"""Data utils.

Forked from the Ex2 codebase.
"""
import pickle
from typing import Any, Dict, Text
import tensorflow as tf


class Seq2SeqExample(object):
  """Represents a sequence-to-sequence training example."""

  def __init__(self, input_str: Text, output_str: Text, metadata: Any = None):
    """Constructs a Seq2SeqExample.

    Args:
      input_str: input text. Not tokenized into a sequence yet.
      output_str: output text. Not tokenized into a sequence yet.
      metadata: arbitrary metadata, must be pickle-able.
    """
    self.input_str = input_str
    self.output_str = output_str
    self.metadata = metadata

  def to_tf_example(self):
    """Converts this into TF Example format."""
    tf_example = tf.train.Example()
    add_text_feature("inputs", self.input_str, tf_example)
    add_text_feature("targets", self.output_str, tf_example)
    add_bytes_feature("metadata", pickle.dumps(self.metadata), tf_example)
    return tf_example


def add_bytes_feature(name: Text, feature: bytes, example: tf.train.Example):
  """Adds a bytes feature field with `name` to `example`."""
  example.features.feature[name].bytes_list.value.append(feature)


def add_text_feature(name: Text, feature: Text, example: tf.train.Example):
  """Adds a text feature field with `name` to `example`."""
  add_bytes_feature(name, feature.encode("utf-8"), example)


def get_byte_to_character_mapping(text: Text) -> Dict[int, int]:
  """Builds mapping from byte to character indices."""
  b2c = dict()
  bytes_offset = 0
  for i, c in enumerate(text):
    b2c[bytes_offset] = i
    bytes_offset += len(tf.compat.as_bytes(c))
  b2c[bytes_offset] = len(text)
  return b2c
