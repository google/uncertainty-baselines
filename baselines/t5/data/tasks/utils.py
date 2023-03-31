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

"""Utility function for specifying seqio tasks."""

import seqio
import t5.data

_DEFAULT_VOCAB = t5.data.get_default_vocabulary()


def get_default_vocab():
  """Defines default vocabulary."""
  return _DEFAULT_VOCAB


def get_output_features_text2text(vocab=_DEFAULT_VOCAB):
  """Defines output feature specifications for text2text task."""
  return {
      'inputs': seqio.Feature(vocabulary=vocab, add_eos=True, required=False),
      'targets': seqio.Feature(vocabulary=vocab, add_eos=True)
  }


def get_output_features_classification(vocab=_DEFAULT_VOCAB):
  """Defines output feature specifications for classification task."""
  # Notice that we do not append add_eos to the target tokens.
  return {
      'inputs': seqio.Feature(vocabulary=vocab, add_eos=True, required=False),
      'targets': seqio.Feature(vocabulary=vocab, add_eos=False)
  }
