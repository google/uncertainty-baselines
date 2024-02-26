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

"""Utils for evaluation."""


def average_list_of_dicts(list_of_dicts):
  """Takes the average of a list of dicts with identical keys."""
  ret_dict = {}
  for cur_dict in list_of_dicts:
    for key, value in cur_dict.items():
      aggregated_value = ret_dict.get(key, 0)
      aggregated_value += value * 1.0 / len(list_of_dicts)
      ret_dict[key] = aggregated_value
  return ret_dict

