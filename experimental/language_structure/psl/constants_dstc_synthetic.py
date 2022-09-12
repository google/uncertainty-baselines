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

"""Constants used for the DSTC8 psl model."""

# Specify test data here.
DEFAULT_DATA_PATH = ""

RULE_WEIGHTS = [1.0] * 1
RULE_NAMES = [f"rule_{i}" for i in range(1, len(RULE_WEIGHTS) + 1)]

DATA_CONFIG = {
    "num_batches": 10,
    "batch_size": 256,
    "max_dialog_size": 24,
    "max_utterance_size": 76,
    "num_labels": 39,
    "includes_word": -1,
    "excludes_word": -2,
    "utterance_mask": -1,
    "last_utterance_mask": -2,
    "pad_utterance_mask": -3,
    "mask_index": 0,
    "hard_pseudo_label": True,
    "word_weights": None,
}
