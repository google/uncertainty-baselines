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

# Lint as: python3
"""Constants used for the DSTC8 psl model."""

# Specify test data here.
DEFAULT_DATA_PATH = ''

RULE_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
RULE_NAMES = ('rule_1', 'rule_2', 'rule_3', 'rule_4', 'rule_5', 'rule_6')

DATA_CONFIG = {
    'num_batches': 10,
    'batch_size': 256,
    'max_dialog_size': 24,
    'max_utterance_size': 76,
    'num_labels': 39,
    'includes_word': -1,
    'excludes_word': -2,
    'utterance_mask': -1,
    'last_utterance_mask': -2,
    'pad_utterance_mask': -3,
    'mask_index': 0,
    'state_transitions': [[1, 3], [1, 7], [1, 1],
                          [2, 10], [2, 9], [2, 8], [3, 5], [3, 12], [3, 15],
                          [4, 2], [4, 9], [4, 10], [5, 11], [5, 14], [5, 17],
                          [6, 1], [6, 18], [6, 7], [7, 5], [7, 12], [7, 15],
                          [8, 6], [8, 19], [8, 20], [9, 1], [9, 3], [9, 7],
                          [10, 1], [10, 3], [10, 7], [11, 8], [11, 13], [11, 2],
                          [12, 8], [12, 13], [12, 2], [13, 6], [13,
                                                                19], [13, 20],
                          [14, 2], [14, 9], [14, 10], [15, 2], [15,
                                                                9], [15, 10],
                          [16, 2], [16, 9], [16, 10], [17, 4], [18, 4], [19, 6],
                          [19, 20], [19, 19], [20, 4], [21, 5], [21, 12],
                          [21, 17], [22, 4], [23, 5], [23, 12], [23,
                                                                 17], [24, 9],
                          [24, 2], [24, 10], [25, 4], [26, 2], [26,
                                                                9], [26, 10],
                          [27, 4], [28, 17], [28, 11], [28, 8], [29, 14],
                          [29, 35], [29, 37], [30, 6], [30, 32], [30, 19],
                          [31, 17], [31, 14], [32, 14], [33, 6], [33, 19],
                          [33, 20], [34, 30], [34, 33], [34, 20], [35, 6],
                          [35, 20], [35, 29], [36, 30], [36, 33], [37, 6],
                          [37, 19], [37, 29], [38, 9], [38, 2], [38, 21]],
    'words': {
        '1': {
            'usr': {
                'index': 1,
                'words': ['leaving', 'planning', 'leave'],
            },
            'sys': {
                'index': 2,
                'words': ['leaving', 'traveling', 'march'],
            },
        },
        '2': {
            'usr': {
                'index': 3,
                'words': ['arrives', 'international', 'unfortunately'],
            },
            'sys': {
                'index': 4,
                'words': ['does', 'address', 'arrive'],
            },
        },
        '3': {
            'usr': {
                'index': 5,
                'words': ['search', 'city', 'watch'],
            },
            'sys': {
                'index': 6,
                'words': ['search', 'round', 'trip'],
            },
        },
        '4': {
            'usr': {
                'index': 7,
                'words': ['buy', 'tickets', 'want'],
            },
            'sys': {
                'index': 8,
                'words': ['sounds', 'good', 'great'],
            },
        },
        '5': {
            'usr': {
                'index': 9,
                'words': ['day', 'good', 'great'],
            },
            'sys': {
                'index': 10,
                'words': ['thank', 'thanks', 'help'],
            },
        },
        '6': {
            'usr': {
                'index': 11,
                'words': ['confirm', 'details', 'following'],
            },
            'sys': {
                'index': 12,
                'words': ['check', 'tickets', 'people'],
            },
        },
        '7': {
            'usr': {
                'index': 13,
                'words': ['plan', 'leave', 'city'],
            },
            'sys': {
                'index': 14,
                'words': ['search', 'round', 'trip'],
            },
        },
        '8': {
            'usr': {
                'index': 15,
                'words': ['successful', 'confirmed', 'fun'],
            },
            'sys': {
                'index': 16,
                'words': ['yes', 'works', 'correct'],
            },
        },
        '9': {
            'usr': {
                'index': 17,
                'words': ['outbound', 'price', 'flight'],
            },
            'sys': {
                'index': 18,
                'words': ['account', 'savings', 'leaving'],
            },
        },
        '10': {
            'usr': {
                'index': 19,
                'words': ['flights', 'buses', 'convenient'],
            },
            'sys': {
                'index': 20,
                'words': ['leaving', 'bed', 'looking'],
            },
        },
        '11': {
            'usr': {
                'index': 21,
                'words': ['help', 'today', 'shall'],
            },
            'sys': {
                'index': 22,
                'words': ['thank', 'thanks', 'help'],
            },
        },
        '12': {
            'usr': {
                'index': 23,
                'words': ['day', 'great', 'good'],
            },
            'sys': {
                'index': 24,
                'words': ['need', 'thanks', 'lot'],
            },
        },
        '13': {
            'usr': {
                'index': 25,
                'words': ['confirmed', 'reservation', 'booked'],
            },
            'sys': {
                'index': 26,
                'words': ['does', 'works', 'address'],
            },
        },
        '14': {
            'usr': {
                'index': 27,
                'words': ['help', 'shall', 'day'],
            },
            'sys': {
                'index': 28,
                'words': ['ok', 'sounds', 'good'],
            },
        },
        '15': {
            'usr': {
                'index': 29,
                'words': ['day', 'great', 'good'],
            },
            'sys': {
                'index': 30,
                'words': ['need', 'bye', 'ok'],
            },
        },
        '16': {
            'usr': {
                'index': 31,
                'words': ['nice', 'located', 'care'],
            },
            'sys': {
                'index': 32,
                'words': ['available', 'events', 'buses'],
            },
        },
        '17': {
            'usr': {
                'index': 33,
                'words': ['shall', 'help', 'today'],
            },
            'sys': {
                'index': 34,
                'words': ['maybe', 'purchase', 'later'],
            },
        },
        '18': {
            'usr': {
                'index': 35,
                'words': ['tickets', 'time', 'need'],
            },
            'sys': {
                'index': 36,
                'words': ['sure', 'like', 'book'],
            },
        },
        '19': {
            'usr': {
                'index': 37,
                'words': ['confirm', 'following', 'details'],
            },
            'sys': {
                'index': 38,
                'words': ['subtitles', 'people', 'group'],
            },
        },
        '20': {
            'usr': {
                'index': 39,
                'words': ['following', 'details', 'confirm'],
            },
            'sys': {
                'index': 40,
                'words': ['play', 'like', 'sure'],
            },
        },
        '21': {
            'usr': {
                'index': 41,
                'words': ['balance', 'slot', 'ends'],
            },
            'sys': {
                'index': 42,
                'words': ['balance', 'account', 'savings'],
            },
        },
        '22': {
            'usr': {
                'index': 43,
                'words': ['goes', 'departure', 'tv'],
            },
            'sys': {
                'index': 44,
                'words': ['vehicle', 'rent', 'ticket'],
            },
        },
        '23': {
            'usr': {
                'index': 45,
                'words': ['songs', 'dentists', 'salons'],
            },
            'sys': {
                'index': 46,
                'words': ['songs', 'searching', 'album'],
            },
        },
        '24': {
            'usr': {
                'index': 47,
                'words': ['slot', 'ends', 'starting'],
            },
            'sys': {
                'index': 48,
                'words': ['available', 'buses', 'tell'],
            },
        },
        '25': {
            'usr': {
                'index': 49,
                'words': ['great', 'day', 'wonderful'],
            },
            'sys': {
                'index': 50,
                'words': ['right', 'buy', 'need'],
            },
        },
        '26': {
            'usr': {
                'index': 51,
                'words': ['buses', 'cars', 'flights'],
            },
            'sys': {
                'index': 52,
                'words': ['buses', 'available', 'higher'],
            },
        },
        '27': {
            'usr': {
                'index': 53,
                'words': ['time', 'expected', 'works'],
            },
            'sys': {
                'index': 54,
                'words': ['calendar', 'add', 'make'],
            },
        },
        '28': {
            'usr': {
                'index': 55,
                'words': ['balance', 'account', 'savings'],
            },
            'sys': {
                'index': 56,
                'words': ['balance', 'songs', 'way'],
            },
        },
        '29': {
            'usr': {
                'index': 57,
                'words': ['subtitles', 'booking', 'details'],
            },
            'sys': {
                'index': 58,
                'words': ['try', 'booking', 'continue'],
            },
        },
        '30': {
            'usr': {
                'index': 59,
                'words': ['sorry', 'originates', 'unable'],
            },
            'sys': {
                'index': 60,
                'words': ['yes', 'good', 'sounds'],
            },
        },
        '31': {
            'usr': {
                'index': 61,
                'words': ['suitable', 'flights', 'songs'],
            },
            'sys': {
                'index': 62,
                'words': ['way', 'interested', 'finding'],
            },
        },
        '32': {
            'usr': {
                'index': 63,
                'words': ['picking', 'drop', 'pickup'],
            },
            'sys': {
                'index': 64,
                'words': ['reserve', 'car', 'song'],
            },
        },
        '33': {
            'usr': {
                'index': 65,
                'words': ['sorry', 'try', 'unable'],
            },
            'sys': {
                'index': 66,
                'words': ['contact', 'number', 'serve'],
            },
        },
        '34': {
            'usr': {
                'index': 67,
                'words': ['help', 'shall', 'day'],
            },
            'sys': {
                'index': 68,
                'words': ['good', 'yes', 'thanks'],
            },
        },
        '35': {
            'usr': {
                'index': 69,
                'words': ['unable', 'sorry', 'requested'],
            },
            'sys': {
                'index': 70,
                'words': ['yes', 'good', 'sounds'],
            },
        },
        '36': {
            'usr': {
                'index': 71,
                'words': ['thank', 'day', 'pleasant'],
            },
            'sys': {
                'index': 72,
                'words': ['need', 'yes', 'good'],
            },
        },
        '37': {
            'usr': {
                'index': 73,
                'words': ['sorry', 'unable', 'requested'],
            },
            'sys': {
                'index': 74,
                'words': ['contact', 'number', 'services'],
            },
        },
        '38': {
            'usr': {
                'index': 75,
                'words': ['failed', 'preferences', 'dates'],
            },
            'sys': {
                'index': 76,
                'words': ['suggest', 'events', 'dates'],
            },
        },
        '39': {
            'usr': {
                'index': 77,
                'words': ['building', 'bulldogs', 'bulls'],
            },
            'sys': {
                'index': 78,
                'words': ['flying', 'fly', 'flights'],
            },
        }
    },
}
