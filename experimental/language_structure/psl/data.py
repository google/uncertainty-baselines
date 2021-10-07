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

# Lint as: python3
"""Utilities for dialog dataset manipulation.

File consists of:
- Basic data loading and parsing
- Adding padding to data and labels
"""

import copy
import json

from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf

Utterance = List[int]
UserSystem = List[Utterance]
Dialog = List[UserSystem]


def _annotate_if_contains_words(utterance: List[int], key_words: List[str],
                                vocab_mapping: Dict[str, int], word_index: int,
                                excludes_word: int,
                                includes_word: int) -> List[int]:
  """Annotates an utterance if it contains at least one word from a list.

  Args:
    utterance: list of integers of length max utterance size representing a
      sentence.
    key_words: list of strings representing the words being looked for.
    vocab_mapping: a dictionary that maps vocab to integers.
    word_index: an integer representing the index an annotation will be placed
      in the utterance.
    excludes_word: an integer indicating an utterance does not contain any key
      words.
    includes_word: an integer indicating an utterance does contain a key word.

  Returns:
    A utterance of length max utterance size annotated with weither it
    includes/excludes at least one key word.
  """
  utterance[word_index] = excludes_word
  for word in key_words:
    if vocab_mapping[word] in utterance:
      utterance[word_index] = includes_word
      break

  return utterance


def add_features(dialogs: List[Dialog], vocab_mapping: Dict[str, int],
                 end_words: List[str], greet_words: List[str],
                 includes_word: int, excludes_word: int, greet_index: int,
                 end_index: int, utterance_mask: int, pad_utterance_mask: int,
                 last_utterance_mask: int, mask_index: int) -> List[Dialog]:
  """Makes a copy of dialogs and annotates if it contains any special tokens.

  This function adds the following features:
    - Padding Mask (indicator if an utterance is a padding)
    - Has Greet Word (indicator if an utterance contains a known greet word)
    - Has End Word (indicator if an utterance contains a known end word)

  Args:
    dialogs: list of dialogs being annotated with special tokens.
    vocab_mapping: a dictionary that maps a vocab to integers.
    end_words: list of strings representing the known end words.
    greet_words: list of strings representing the known greet words.
    includes_word: an integer indicating an utterance does contain a key word.
    excludes_word: an integer indicating an utterance does not contain a key
      word.
    greet_index: an integer representing the index a greet annotation will be
      placed in the utterance.
    end_index: an integer representing the index an end annotation will be
      placed in the utterance.
    utterance_mask: an integer indicating if it is not a padded utterance.
    pad_utterance_mask: an integer indicating if it is a padded utterance.
    last_utterance_mask: an integer indicating if it is the final utterance
      before padding.
    mask_index: an integer representing the index the utterance mask will be
      placed in the utterance.

  Returns:
    A copy of dialogs with annotations for special tokens included.
  """
  dialogs_copy = copy.deepcopy(dialogs)
  for index_i in range(len(dialogs)):
    first_padding = True
    for index_j in range(len(dialogs[index_i])):
      # Add null values for features.
      utterance = [0, 0, 0] + dialogs_copy[index_i][index_j][0]

      # Checks if the utternace in dialog is a padded utterance.
      utterance[mask_index] = utterance_mask
      if all(word == 0 for word in utterance):
        utterance[mask_index] = pad_utterance_mask

        # Checks if this is the first padding.
        if first_padding and index_j != 0:
          # Sets previous statement to last utterance in dialog.
          dialogs_copy[index_i][index_j -
                                1][0][mask_index] = last_utterance_mask
          first_padding = False
      # Check edge case where last utterance is not padding.
      elif first_padding and index_j == len(
          dialogs_copy[index_i][index_j][0]) - 1:
        utterance[mask_index] = last_utterance_mask

      # Checks if utterance in dialog contains a known end word.
      utterance = _annotate_if_contains_words(utterance, end_words,
                                              vocab_mapping, end_index,
                                              excludes_word, includes_word)

      # Checks if utterance in dialog contains a known greet word.
      utterance = _annotate_if_contains_words(utterance, greet_words,
                                              vocab_mapping, greet_index,
                                              excludes_word, includes_word)

      # Sets utterance with new features.
      dialogs_copy[index_i][index_j][0] = utterance

  return dialogs_copy


def pad_utterance(utterance: Utterance,
                  max_utterance_size: int) -> Tuple[Utterance, List[int]]:
  """Pads utterance up to the max utterance size."""
  utt = utterance + [0] * (max_utterance_size - len(utterance))
  mask = [1] * len(utterance) + [0] * (max_utterance_size - len(utterance))
  return utt, mask


def pad_dialog(
    dialog: Dialog, max_dialog_size: int, max_utterance_size: int
) -> Tuple[List[List[int]], List[List[int]], List[List[int]], List[List[int]]]:
  """Pads utterances in a dialog up to max dialog sizes."""

  dialog_usr_input, dialog_usr_mask, dialog_sys_input, dialog_sys_mask = [], [], [], []

  for turn in dialog:
    pad_utt, mask = pad_utterance(turn[0], max_utterance_size)
    dialog_usr_input.append(pad_utt)
    dialog_usr_mask.append(mask)

    pad_utt, mask = pad_utterance(turn[1], max_utterance_size)
    dialog_sys_input.append(pad_utt)
    dialog_sys_mask.append(mask)

  for _ in range(max_dialog_size - len(dialog)):
    pad_utt, mask = pad_utterance([], max_utterance_size)
    dialog_usr_input.append(pad_utt)
    dialog_usr_mask.append(mask)

    pad_utt, mask = pad_utterance([], max_utterance_size)
    dialog_sys_input.append(pad_utt)
    dialog_sys_mask.append(mask)

  return dialog_usr_input, dialog_usr_mask, dialog_sys_input, dialog_sys_mask


def pad_dialogs(
    dialogs: List[Dialog], max_dialog_size: int, max_utterance_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Pads all dialogs and utterances."""
  usr_input_sent, usr_input_mask, sys_input_sent, sys_input_mask = [], [], [], []

  for dialog in dialogs:
    usr_input, usr_mask, sys_input, sys_mask = pad_dialog(
        dialog, max_dialog_size, max_utterance_size)

    usr_input_sent.append(usr_input)
    usr_input_mask.append(usr_mask)
    sys_input_sent.append(sys_input)
    sys_input_mask.append(sys_mask)

  return np.array(usr_input_sent), np.array(usr_input_mask), np.array(
      sys_input_sent), np.array(sys_input_mask),


def one_hot_string_encoding(labels: List[List[str]],
                            mapping: Dict[str, int]) -> List[List[List[int]]]:
  """Converts string labels into one hot encodings."""
  one_hot_labels = []

  for dialog in labels:
    one_hot_labels.append([])
    for utterance in dialog:
      one_hot_labels[-1].append([0] * len(mapping))
      one_hot_labels[-1][-1][mapping[utterance]] = 1

  return one_hot_labels


def pad_one_hot_labels(
    labels: List[List[List[int]]], max_dialog_size: int,
    mapping: Dict[str, int]) -> Tuple[List[List[List[int]]], List[List[int]]]:
  """Pads one hot encoded lables."""
  pad_labels = []
  pad_mask = []

  for dialog in labels:
    pad_labels.append(dialog)
    pad_mask.append([1] * len(dialog) + [0] * (max_dialog_size - len(dialog)))

    for _ in range(max_dialog_size - len(dialog)):
      pad_labels[-1].append([0] * len(mapping))

  return pad_labels, pad_mask


def list_to_dataset(data: List[Dialog], labels: List[List[List[int]]],
                    shuffle: bool, batch_size: int) -> tf.data.Dataset:
  """Converts list into tensorflow dataset."""
  ds = tf.data.Dataset.from_tensor_slices((data, labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(data))
  ds = ds.batch(batch_size)
  return ds


def load_json(path: str):
  with tf.gfile.GFile(path, 'r') as json_file:
    return json.load(json_file)
