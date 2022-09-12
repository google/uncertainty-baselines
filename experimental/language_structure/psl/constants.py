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

"""Constants used in gradient based inference with psl constraints."""

# Specify test data here.
DEFAULT_DATA_PATH = ''

MULTIWOZ_RULE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
MULTIWOZ_CONFIG = {
    'data_path': DEFAULT_DATA_PATH,
    'default_seed': 5,
    'batch_size': 128,
    'max_dialog_size': 10,
    'max_utterance_size': 40,
    'greet_words': ['hello', 'hi'],
    'end_words': ['thank', 'thanks'],
    'class_map': {
        'accept': 0,
        'cancel': 1,
        'end': 2,
        'greet': 3,
        'info_question': 4,
        'init_request': 5,
        'insist': 6,
        'second_request': 7,
        'slot_question': 8,
    },
    'contains_word': -1,
    'excludes_word': -2,
    'greet_index': 0,
    'end_index': 1,
}

