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

"""Diversity metrics and utils."""

import dpp_negative_logdet  # local file import
import pairwise_cosine_similarity  # local file import
import pairwise_euclidean_distances  # local file import
import ExponentialDecay  # local file import
import fast_weights_similarity  # local file import
import LinearAnnealing  # local file import
import outputs_similarity  # local file import
import scaled_similarity_loss  # local file import
