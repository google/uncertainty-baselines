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

"""Add Mixtures to the registry.

This module contains different mixtures for training T5 models.
"""
import seqio

import uncertainty_baselines.baselines.t5.data.tasks  # pylint: disable=unused-import
import deepbank as deepbank_config  # local file import from baselines.t5.data.tasks
import mnli as mnli_config  # local file import from baselines.t5.data.tasks

MixtureRegistry = seqio.MixtureRegistry

# ========================== Toxic Comments ================================
# TODO(jerelu): Add toxic comments covert mixtures.

# A mixture that trains on wikipedia_talk and
# evaluates on the validation splits of civil_comments.
MixtureRegistry.add(
    'toxic_comments_with_ood_eval',
    tasks=[
        'wikipedia_talk', 'wikipedia_talk_eval_only',
        'civil_comments_eval_only', 'civil_comments_identity_eval_only'
    ],
    default_rate=1.)

MixtureRegistry.add(
    'toxic_comments_infer',
    tasks=[
        'wikipedia_talk_eval_only',
        'civil_comments_eval_only',
        'civil_comments_identity_eval_only',
    ],
    default_rate=1.)

# ========================== NLI ===========================================
main_mnli_tasks = [
    'mnli',
    'mnli_mismatched',
    'hans_all',
]
hans_subpopulation_tasks = [
    f'hans_{subpopulation_type}'
    for subpopulation_type in mnli_config.get_hans_subpopulation_types()
]

MixtureRegistry.add('mnli', tasks=main_mnli_tasks, default_rate=1.)

MixtureRegistry.add('mnli_subpopulation',
                    tasks=main_mnli_tasks + hans_subpopulation_tasks,
                    default_rate=1.)

# ================ Compositional Intent Understanding (NaLUE) ==============
MixtureRegistry.add(
    'nalue',
    tasks=[
        'nalue',
        'nalue_standard_oos',
        'nalue_near_oos',
        'nalue_tail_intent',
        'nalue_ind_and_standard_oos',
        'nalue_ind_and_near_oos',
    ],
    default_rate=1.)

# ========================== Graph Semantic Parsing ============================
deepbank_subpopulation_tasks = ['deepbank_ood_valency', 'deepbank_ood_lang10']
deepbank_ood_tasks = [
    f'deepbank_ood_{data_name}'
    for data_name in deepbank_config.ood_config_patterns.keys()
]
deepbank_ood_aug_tasks = [
    f'deepbank_ood_aug_{data_name}'
    for data_name in deepbank_config.ood_aug_config_patterns.keys()
]

MixtureRegistry.add(
    'deepbank', tasks=[
        'deepbank',
    ], default_rate=1.)
MixtureRegistry.add(
    'deepbank_1.1', tasks=[
        'deepbank_1.1',
    ], default_rate=1.)
MixtureRegistry.add(
    'deepbank_1.1_aug', tasks=[
        'deepbank_1.1_aug',
    ], default_rate=1.)

MixtureRegistry.add(
    'deepbank_infer',
    tasks=['deepbank'] + deepbank_subpopulation_tasks + deepbank_ood_tasks,
    default_rate=1.)

MixtureRegistry.add(
    'deepbank_1.1_infer',
    tasks=['deepbank_1.1'] + deepbank_subpopulation_tasks + deepbank_ood_tasks,
    default_rate=1.)

MixtureRegistry.add(
    'deepbank_1.1_aug_infer',
    tasks=['deepbank_1.1_aug'] + deepbank_ood_aug_tasks,
    default_rate=1.)

# ====================== Dataflow Semantic Parsing =============================
MixtureRegistry.add('smcalflow', tasks=['smcalflow'], default_rate=1.)

MixtureRegistry.add('multiwoz', tasks=['multiwoz'], default_rate=1.)

MixtureRegistry.add('mtop', tasks=['mtop'], default_rate=1.)
MixtureRegistry.add('mtop_penman', tasks=['mtop_penman'], default_rate=1.)

MixtureRegistry.add('snips', tasks=['snips'], default_rate=1.)
MixtureRegistry.add('snips_penman', tasks=['snips_penman'], default_rate=1.)
