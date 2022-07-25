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

"""Utility functions for metrics computation."""
import re
from typing import Any, Dict, List, Text

from absl import logging
import numpy as np

# Argument edge types shared by DeepBank 1.0 and DeepBank 1.1.
MISC = ['-of']

ARG_EDGES = [
    ':ARG', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':BV', ':carg', ':L-INDEX',
    ':R-INDEX', ':L-HNDL', ':R-HNDL'
]

FUNC_NODES = {
    'v0': [  # 87 function nodes used by DeepBank 1.0.
        'every_q', 'ellipsis', 'interval_p_end', 'unknown', 'place_n',
        'meas_np', 'implicit_conj', 'little-few_a', 'number_q', 'property',
        'loc_nonsp', 'timezone_p', 'relative_mod', 'generic_entity', 'holiday',
        'unspec_adj', 'comp_enough', 'much-many_a', 'dofw', 'reason', 'measure',
        'temp', 'neg', 'dofm', 'ellipsis_ref', 'thing', 'refl_mod', 'excl',
        'id', 'manner', 'free_relative_q', 'addressee', 'fraction', 'v_event',
        'times', 'eventuality', 'comp_less', 'compound', 'num_seq', 'of_p',
        'with_p', 'interval', 'subord', 'idiom_q_i', 'proper_q', 'named_n',
        'cop_id', 'generic_verb', 'superl', 'comp_too', 'ellipsis_expl',
        'comp_equal', 'ord', 'nominalization', 'polite', 'abstr_deg',
        'elliptical_n', 'udef_q', 'recip_pro', 'card', 'yofc', 'discourse',
        'plus', 'numbered_hour', 'interval_p_start', 'year_range', 'pronoun_q',
        'def_implicit_q', 'season', 'appos', 'fw_seq', 'not_x_deg',
        'unspec_manner', 'def_explicit_q', 'parenthetical', 'comp_so', 'time_n',
        'prpstn_to_prop', 'mofy', 'person', 'named', 'comp', 'comp_not+so',
        'pron', 'poss', 'part_of', 'temp_loc_x'
    ],
    'v1': [  # 88 function nodes used by DeepBank 1.0.
        'udef_q', 'compound', 'named', 'proper_q', 'card', 'pronoun_q', 'pron',
        'def_explicit_q', 'poss', 'parg_d', 'focus_d', 'loc_nonsp',
        'nominalization', 'times', 'appos', 'def_implicit_q', 'nn_u_unknown',
        'generic_entity', 'comp', 'neg', 'implicit_conj', 'subord', 'named_n',
        'mofy', 'time_n', 'yofc', 'nns_u_unknown', 'part_of', 'number_q',
        'much-many_a', 'jj_u_unknown', 'ord', 'unknown', 'superl', 'place_n',
        'of_p', 'dofw', 'which_q', 'dofm', 'thing', 'comp_equal', 'measure',
        'fraction', 'plus', 'eventuality', 'with_p', 'idiom_q_i',
        'little-few_a', 'parenthetical', 'person', 'ellipsis_ref',
        'elliptical_n', 'interval', 'interval_p_end', 'interval_p_start',
        'season', 'rb_u_unknown', 'comp_so', 'free_relative_q', 'id',
        'vbn_u_unknown', 'temp_loc_x', 'vb_u_unknown', 'comp_too',
        'unspec_manner', 'manner', 'discourse', 'excl', 'vbg_u_unknown',
        'year_range', 'every_q', 'vbd_u_unknown', 'numbered_hour', 'abstr_deg',
        'temp', 'comp_less', 'reason', 'fw_u_unknown', 'comp_enough', 'holiday',
        'ellipsis', 'vbz_u_unknown', 'fw_seq', 'recip_pro',
        'free_relative_ever_q', 'vbp_u_unknown', 'timezone_p', 'refl_mod'
    ]
}

# Content node postfixes shared by DeepBank 1.0 and DeepBank 1.1.
CONTENT_NODE_POSTFIX_NAMES = [
    'n_1', 'q', 'n_of', 'v_1', 'p', 'a_1', 'c', 'v_to', 'v_modal', 'x_deg',
    'v_id', 'x', 'q_dem', 'v_cause', 'p_temp', 'a_for', 'p_state', 'x_subord',
    'a_of', 'v_for', 'p_per', 'a_to', 'x_then', 'n_temp', 'n_to', 'v_from',
    'n_in', 'v_with', 'v_there', 'v_2', 'v_of', 'v_up', 'p_namely', 'v_as',
    'v_on', 'v_qmodal', 'p_dir', 'v_in', 'a_at-for-of', 'x_h', 'v_at',
    'p_means', 'n_of-on', 'n_of-n', 'n_2', 'v_out', 'n_for', 'v_state', 'v_prd',
    'a_again', 'v_into', 'a_with', 'v_about', 'v_off', 'n_about', 'n_of-to',
    'a_as', 'v_nv', 'n_of-about', 'a_rvrs', 'n_cur', 'v_name', 'v_down', 'a_on',
    'a_about', 'a_from', 'v_from-to', 'n_on-about', 'n_i', 'p_nbar', 'p_time',
    'a_at', 'a_ante', 'a_than-from', 'a_in', 'v_back', 'x_prd', 'a_also',
    'v_over', 'a_thus', 'n_item', 'q_indiv', 'n_of-for', 'v_of-i', 'a_disc',
    'v_by', 'v_to-about', 'n_at', 'v_away', 'c_btwn', 'n_with', 'v_unspec',
    'v_on-upon', 'v_seem-to', 'v_seem-about', 'n_on', 'v_i', 'v_itcleft',
    'p_except', 'v_out-to', 'a_of-about', 'v_so', 'p_ind', 'v_with-for',
    'a_error', 'v_for-as', 'v_to-with', 'v_buy', 'n_of-as', 'p_1',
    'a_with-about-of', 'a_2', 'v_through', 'v_transfer', 'x_cond', 'v_around',
    'n_from', 'v_mental', 'n_meas', 'v_dir', 'v_like', 'v_coll', 'v_cause-on',
    'a', 'v_against', 'a_expl-for', 'a_same-as', 'v_together', 'a_to-for',
    'v_aside', 'v_do', 'v_be', 'a_of-for', 'p_comp', 'v_along', 'x_cause',
    'v_adv', 'a_true', 'v_onto', 'c_from', 'a_with-at', 'v_towards',
    'v_out-aim', 'v_ahead', 'a_at-by-in', 'v_even', 'n_abb', 'v_up-to',
    'v_up-for', 'v_up-of', 'a_at-by-with', 'v_yet', 'n_num', 'n_money',
    'v_cope', 'a_former', 'a_of-to', 'v_ing', 'n_against', 'c_not', 'v_open',
    'v_behind', 'v_x', 'v_after', 'v_forward-to', 'v_seem', 'a_for-as', 'v_loc',
    'v_upon', 'v_home', 'a_accept', 'v_across', 'n_do-be', 'v_away-with',
    'x_preph', 'v_to-i', 'v_sound', 'v_apart', 'v_up2', 'v_cause-to', 'v_yield',
    'v_suffice', 'v_it', 'n_pair', 'v_x-off', 'v_forth', 'v_out-of', 'c_mod',
    'v_without', 'x_1', 'v_out+of', 'v_up-with', 'p_time-on', 'v_up-cause',
    'v_cause-into', 'p_place-in', 'n_at-with', 'n_into', 'v_seem+to'
]

# Special SentencePiece tokens to be used to represent content-node postfixes.
# (total number = 398).
SPECIAL_SENTENCEPIECE_TOKENS = [
    'au\u00dfergew\u00f6hnlich', 'ver\u00f6ffentlicht', 'responsabilit\u00e9',
    'ausschlie\u00dflich', 'suppl\u00e9mentaire', 'fonctionnalit\u00e9',
    'repr\u00e9sentation', 'compl\u00e9mentaire', 'pr\u00e9sidentielle',
    'ber\u00fccksichtigt', 'Pers\u00f6nlichkeit', 't\u00e9l\u00e9chargement',
    'r\u00e9glementation', 'd\u00e9veloppement', 'M\u00f6glichkeiten',
    'Unterst\u00fctzung', 'dumneavoastr\u0103', 'r\u00e9guli\u00e8rement',
    'grunds\u00e4tzlich', 'interna\u021bional', 'interna\u0163ional',
    'imm\u00e9diatement', 'gew\u00e4hrleisten', 'B\u00fcrgermeister',
    'reprezentan\u021bi', 'probl\u00e9matique', 'disponibilit\u00e9',
    'gew\u00e4hrleistet', 'haupts\u00e4chlich', 'Einschr\u00e4nkung',
    'Besch\u00e4ftigung', 'collectivit\u00e9s', 'compr\u00e9hension',
    'Grunds\u00e4tzlich', 'Interna\u021bional', 'unterst\u00fctzen',
    'durchgef\u00fchrt', 'anschlie\u00dfend', '\u00e9lectronique',
    'pr\u00e9sentation', 'personnalis\u00e9', 'wundersch\u00f6ne',
    'particuli\u00e8re', 'compl\u00e8tement', 'ausdr\u00fccklich',
    'urspr\u00fcnglich', 'pr\u00e4sentieren', 'biblioth\u00e8que',
    'repr\u00e9sentant', 'Durchf\u00fchrung', 'participan\u021bi',
    'personnalit\u00e9', 'profesional\u0103', '\u00dcberraschung',
    'besch\u00e4ftigen', 't\u00e9l\u00e9phonique', 'g\u00e9ographique',
    '\u00eenregistrare', 'op\u00e9rationnel', 'M\u00f6glichkeit',
    'diff\u00e9rentes', 'possibilit\u00e9', 'd\u00e9partement',
    'unterst\u00fctzt', 'tats\u00e4chlich', 'int\u00e9ressant',
    'Universit\u00e4t', 'r\u00e9alisation', 'comp\u00e9tences',
    'temp\u00e9rature', 'pers\u00f6nliche', 'zus\u00e4tzliche',
    'n\u00e9cessaires', '\u00eenregistrat', 'vollst\u00e4ndig',
    'erm\u00f6glichen', 'cons\u00e9quence', 'pr\u00e4sentiert',
    'Ver\u00e4nderung', 'pr\u00e9paration', 'ind\u00e9pendant',
    'Bed\u00fcrfnisse', 'enti\u00e8rement', 'zuverl\u00e4ssig',
    '\u00eentotdeauna', 'besch\u00e4ftigt', 'sp\u00e9cialiste',
    'Bev\u00f6lkerung', 'd\u00e9placement', 'comp\u00e9tition',
    'd\u00e9claration', 'Aktivit\u00e4ten', '\u00e9nerg\u00e9tique',
    '\u00e9quipements', 'ausf\u00fchrlich', 'r\u00e9servation',
    't\u00e9l\u00e9charger', 'langj\u00e4hrige', 'Verst\u00e4ndnis',
    'autorit\u0103\u021bil', 'strat\u00e9gique', 'F\u00e4higkeiten',
    '\u00fcberraschen', 'autorit\u0103\u0163il', 'k\u00f6rperliche',
    'str\u0103in\u0103tate', 'tradi\u021bional', '\u00dcbersetzung',
    'pr\u00e9cis\u00e9ment', 'p\u00e9dagogique', 'Kreativit\u00e4t',
    'litt\u00e9rature', 'diff\u00e9rents', '\u00e9conomique', 'n\u00e9cessaire',
    'exp\u00e9rience', 'regelm\u00e4\u00dfig', 'reprezint\u0103',
    'zus\u00e4tzlich', '\u00e9lectrique', 'sp\u00e9cialis\u00e9',
    'erm\u00f6glicht', 'sp\u00e9cifique', 'communaut\u00e9', 'informa\u021bii',
    'europ\u00e9enne', 'd\u00e9velopper', 'diff\u00e9rence', 'd\u00e9couverte',
    'activit\u0103\u021bi', 'pers\u00f6nlich', '\u00f6ffentlich',
    'repr\u00e9sente', 'unabh\u00e4ngig', 'b\u00e9n\u00e9ficier',
    'conf\u00e9rence', 'accompagn\u00e9', 'financi\u00e8re', 'erh\u00e4ltlich',
    'activit\u0103\u0163i', 'Oberfl\u00e4che', 'R\u00e9publique',
    'm\u00e9dicament', 'informa\u0163ii', 'ausgew\u00e4hlt', '\u00fcbernehmen',
    'D\u00fcsseldorf', 'd\u00e9coration', 'r\u00e9paration', 'produc\u0103tor',
    'Zus\u00e4tzlich', '\u00fcberzeugen', 'modific\u0103ri', 'vielf\u00e4ltig',
    'gew\u00fcnschte', 'Atmosph\u00e4re', '\u00e9v\u00e9nements',
    '\u00e9videmment', 'r\u00e9volution', 'experien\u021b\u0103',
    'r\u00e9sistance', 'communiqu\u00e9', '\u00fcbertragen', 'Grundst\u00fcck',
    'enregistr\u00e9', 'institu\u021bii', 't\u00e9l\u00e9vision',
    'recommand\u00e9', 'caract\u00e9ris', '\u00e9cologique', 'Einf\u00fchrung',
    'cons\u00e9quent', 'pr\u00e9vention', '\u00fcbernommen', 'r\u00e9solution',
    'r\u00e9novation', 'institu\u0163ii', 'pre\u0219edinte',
    'rom\u00e2neasc\u0103', 'compl\u00e9ment', 'd\u00e9finitive',
    'd\u00e9finition', 'Ausf\u00fchrung', 'kilom\u00e8tres', 't\u00e9moignage',
    'zug\u00e4nglich', 'performan\u021b', 'd\u00e9terminer', 'm\u00e9tallique',
    'coordonn\u00e9e', 'comunit\u0103\u021bi', 'sorgf\u00e4ltig',
    'Angeh\u00f6rige', '\u00eenv\u0103\u021b\u0103m\u00e2nt',
    'pr\u00e9f\u00e9rence', '\u00eenchisoare', '\u00f6kologisch',
    'comp\u00e9tence', '\u00eenv\u0103\u0163\u0103m\u00e2nt',
    '\u00fcberpr\u00fcfen', 'investi\u021bii', 'solidarit\u00e9',
    'd\u00e9mocratie', 'cr\u00e9ativit\u00e9', 'Eigent\u00fcmer',
    'conduc\u0103tor', '\u00fcberrascht', 'r\u00e9compense', 'gew\u00f6hnlich',
    '\u00e9galement', 'nat\u00fcrlich', 'Verf\u00fcgung', 'd\u00e9couvrir',
    'Gesch\u00e4fts', 's\u00e9lection', 'activit\u00e9s', 'pr\u00e9sident',
    'fran\u00e7aise', 'gegen\u00fcber', 'r\u00e9sultats', 'v\u00e9ritable',
    'Bucure\u0219ti', 'r\u00e9f\u00e9rence', 'Bucure\u015fti', 'am\u00e9ricain',
    'num\u00e9rique', 't\u00e9l\u00e9phone', 'probl\u00e8mes', 'Nat\u00fcrlich',
    'cat\u00e9gorie', '\u00fcberhaupt', 'sup\u00e9rieur', 'pr\u00e9senter',
    'niciodat\u0103', 'pr\u00e9c\u00e9dent', 'strat\u00e9gie', 'conna\u00eetre',
    '\u00e9tudiants', 'caract\u00e8re', 'proc\u00e9dure', 'Ern\u00e4hrung',
    'proximit\u00e9', 'r\u00e9duction', 'propri\u00e9t\u00e9', 'D\u00e9couvrez',
    'Ma\u00dfnahmen', 'verf\u00fcgbar', 'repr\u00e9sent', 'gegr\u00fcndet',
    'diff\u00e9rent', 'gem\u00fctlich', 'func\u021biona',
    's\u0103pt\u0103m\u00e2ni', 'mat\u00e9riaux', 'r\u00e9flexion',
    '\u00fcberzeugt', 'ver\u00e4ndert', '\u00eenceputul', 'zust\u00e4ndig',
    'r\u00e9ception', 'Schl\u00fcssel', '\u00eencredere', 'Fr\u00fchst\u00fcck',
    'r\u00e9sidence', 'Pr\u00e9sident', 'privil\u00e9gi', 'gl\u00fccklich',
    'r\u00e8glement', '\u00eentreb\u0103ri', 'd\u00e9velopp\u00e9',
    'siguran\u021b\u0103', 'T\u00e4tigkeit', 'v\u00eatements', 'pr\u00e9alable',
    'F\u00f6rderung', 's\u0103pt\u0103m\u00e2n\u0103', 'minist\u00e8re',
    'forc\u00e9ment', 'judec\u0103tor', '\u00dcberblick', 'experien\u0163',
    'experien\u021b', 'urm\u0103toare', 'r\u00e9cemment', 'recomand\u0103',
    'consid\u00e9r\u00e9', 'gesch\u00fctzt', 'rencontr\u00e9', 'best\u00e4tigt',
    'passionn\u00e9', 's\u0103rb\u0103tori', 'Anspr\u00fcche', 'pl\u00f6tzlich',
    'fronti\u00e8re', '\u00dcbersicht', '\u00eembun\u0103t\u0103\u021b',
    'ph\u00e9nom\u00e8ne', 'reconna\u00eet', 'K\u00fcndigung', '\u00eentrebare',
    'cercet\u0103ri', 'pr\u00e9cision', 'client\u00e8le', 'd\u00e9pannage',
    'dispozi\u0163i', 'travaill\u00e9', 'Erkl\u00e4rung', 'ver\u00e4ndern',
    'd\u00e9couvert', '\u00eent\u00e2mplat', 'int\u00e9ress\u00e9',
    'n\u00e9cessit\u00e9', 'Timi\u0219oara', 'schimb\u0103ri', 'competi\u021bi',
    'diversit\u00e9', 'verst\u00e4rkt', 'Erf\u00fcllung',
    's\u0103pt\u0103m\u00e2na', 'Verk\u00e4ufer', 'appropri\u00e9',
    '\u00e9trang\u00e8re', 'societ\u0103\u0163i', 'm\u00e9canique',
    'nouveaut\u00e9', 'Sch\u00f6nheit', 'compliqu\u00e9', 'societ\u0103\u021bi',
    'chaudi\u00e8re', 'parall\u00e8le', 'gef\u00f6rdert', 'randonn\u00e9e',
    '\u00eent\u00e2lnire', 'r\u00e9daction', 'conseill\u00e9', 'comp\u00e9tent',
    'importan\u021b', 'compl\u00e9ter', 'Erg\u00e4nzung', 'c\u00e9r\u00e9monie',
    'Constan\u0163a', '\u00eent\u00e2lniri', 'Er\u00f6ffnung',
    'r\u00e9guli\u00e8re', 'Timi\u015foara', 'na\u0163ionale', 'consid\u00e8re',
    'func\u0163iona', 'Constan\u021ba', 'confrunt\u0103', 'gew\u00fcnscht',
    'inf\u00e9rieur', 'premi\u00e8re', 'derni\u00e8re', 'Qualit\u00e4t',
    'fran\u00e7ais', 's\u00e9curit\u00e9', 'cr\u00e9ation',
    'g\u00e9n\u00e9rale', 'pr\u00e9sente', 'Rom\u00e2niei', 'probl\u00e8me',
    'contr\u00f4le', '\u00eempreun\u0103', 'adev\u0103rat', 'r\u00e9aliser',
    'v\u00e9hicule', 'capacit\u00e9', '\u00eempotriv', 'd\u00e9cembre',
    'mat\u00e9riel', 'r\u00e9pondre', 'Gespr\u00e4ch', 'pr\u00e9sence',
    'ben\u00f6tigt', 'perioad\u0103', 'Na\u0163ional', 'r\u00e9sultat',
    'd\u00e9cision', 'na\u021bional', 'Na\u021bional', 'Fran\u00e7ais',
    'b\u00e9n\u00e9fici', 'K\u00fcnstler', 'b\u00e2timent', 'd\u00e9marche',
    'agr\u00e9able', 's\u0103n\u0103tate', 'compl\u00e8te', 'genie\u00dfen',
    'effectu\u00e9', 'condi\u0163ii', 'cre\u0219tere', 'pr\u00e9sent\u00e9',
    'europ\u00e9en', 'condi\u021bii', 'persoan\u0103', 'jum\u0103tate'
]

# Makes a mapping between graph element names and their corresponding tokens
# for both DeepBank 1.0 and DeepBank 1.1.
TOKEN_TO_NAME_MAPS = {}
NAME_TO_TOKEN_MAPS = {}
for version in ('v0', 'v1'):
  token_to_name_map = {}
  name_to_token_map = {}
  # Uses the 100 extra_ids for argument edges and function nodes.
  names = MISC + ARG_EDGES + FUNC_NODES[version]
  tokens = [f'<extra_id_{i}>' for i in range(len(names))]
  token_to_name_map = dict(zip(tokens, names))
  name_to_token_map = dict(zip(names, tokens))
  # Uses the special SentencePiece tokens for the content post-fixes.
  names = CONTENT_NODE_POSTFIX_NAMES
  tokens = SPECIAL_SENTENCEPIECE_TOKENS
  token_to_name_map.update(dict(zip(tokens[:len(names)], names)))
  name_to_token_map.update(dict(zip(names, tokens[:len(names)])))

  TOKEN_TO_NAME_MAPS[version] = token_to_name_map
  NAME_TO_TOKEN_MAPS[version] = name_to_token_map

# Argument edge types shared by SNIPS and MTOP.
SNIPS_MTOP_ARG_EDGES = [
    ':carg', ':ARG1', ':ARG2', ':ARG3', ':ARG4', ':ARG5', ':ARG6', ':ARG7',
    ':ARG8', ':ARG9', ':ARG10', ':ARG11', ':ARG12'
]

# Intent types for SNIPS and MTOP
SNIPS_INTENT = [
    'SL:SERVED_DISH', 'SL:CONDITION_TEMPERATURE', 'SL:CURRENT_LOCATION',
    'SL:GEOGRAPHIC_POI', 'IN:SEARCH_CREATIVE_WORK', 'SL:GENRE', 'SL:TRACK',
    'IN:ADD_TO_PLAYLIST', 'SL:BEST_RATING', 'SL:CUISINE', 'SL:ALBUM',
    'SL:RATING_VALUE', 'SL:OBJECT_PART_OF_SERIES_TYPE',
    'SL:CONDITION_DESCRIPTION', 'SL:MUSIC_ITEM', 'SL:POI', 'IN:GET_WEATHER',
    'SL:RESTAURANT_NAME', 'SL:SERVICE', 'IN:BOOK_RESTAURANT', 'SL:ENTITY_NAME',
    'SL:TIME_RANGE', 'SL:OBJECT_SELECT', 'SL:OBJECT_LOCATION_TYPE', 'SL:STATE',
    'SL:PLAYLIST', 'SL:OBJECT_NAME', 'SL:RATING_UNIT', 'SL:OBJECT_TYPE',
    'SL:LOCATION_NAME', 'SL:MOVIE_NAME', 'SL:ARTIST',
    'SL:PARTY_SIZE_DESCRIPTION', 'SL:CITY', 'SL:PARTY_SIZE_NUMBER',
    'SL:FACILITY', 'SL:RESTAURANT_TYPE', 'SL:SORT', 'IN:PLAY_MUSIC',
    'IN:SEARCH_SCREENING_EVENT', 'SL:YEAR', 'SL:SPATIAL_RELATION', 'SL:COUNTRY',
    'IN:RATE_BOOK', 'SL:PLAYLIST_OWNER', 'SL:MOVIE_TYPE'
]
MTOP_INTENT = [
    'SL:METHOD_RECIPES', 'SL:CONTACT', 'IN:GET_INFO_CONTACT', 'IN:DISPREFER',
    'SL:ALARM_NAME', 'IN:RESTART_TIMER', 'IN:IS_TRUE_RECIPES', 'IN:END_CALL',
    'IN:GET_CALL_CONTACT', 'IN:GET_EDUCATION_DEGREE', 'IN:GET_RECIPES',
    'IN:PAUSE_MUSIC', 'SL:MUSIC_TRACK_TITLE', 'SL:MAJOR', 'IN:DISLIKE_MUSIC',
    'SL:METHOD_MESSAGE', 'IN:SWITCH_CALL', 'SL:WEATHER_TEMPERATURE_UNIT',
    'SL:METHOD_RETRIEVAL_REMINDER', 'SL:MUTUAL_EMPLOYER', 'SL:GENDER',
    'SL:RECIPES_RATING', 'IN:GET_GENDER', 'IN:GET_EDUCATION_TIME',
    'IN:STOP_SHUFFLE_MUSIC', 'IN:GET_TRACK_INFO_MUSIC', 'SL:TITLE_EVENT',
    'IN:SET_RSVP_YES', 'IN:GET_MUTUAL_FRIENDS', 'IN:GET_TODO',
    'SL:MUSIC_ALBUM_MODIFIER', 'IN:REPLY_MESSAGE', 'IN:GET_REMINDER_DATE_TIME',
    'SL:TYPE_CONTENT', 'IN:GET_AGE', 'IN:DELETE_REMINDER',
    'IN:GET_LIFE_EVENT_TIME', 'IN:CREATE_ALARM', 'IN:CREATE_REMINDER',
    'SL:RECIPES_UNIT_MEASUREMENT', 'SL:MUSIC_PROVIDER_NAME',
    'SL:CONTACT_RELATED', 'IN:HELP_REMINDER', 'IN:GET_REMINDER_LOCATION',
    'SL:CONTACT_REMOVED', 'IN:GET_LYRICS_MUSIC', 'SL:METHOD_TIMER', 'SL:TODO',
    'IN:GET_JOB', 'SL:RECIPES_ATTRIBUTE', 'IN:CANCEL_CALL', 'SL:NEWS_CATEGORY',
    'SL:CONTACT_METHOD', 'SL:PHONE_NUMBER', 'IN:GET_EVENT',
    'IN:GET_MESSAGE_CONTACT', 'SL:NEWS_REFERENCE', 'IN:DELETE_PLAYLIST_MUSIC',
    'SL:EMPLOYER', 'IN:CREATE_CALL', 'IN:SET_AVAILABLE', 'SL:WEATHER_ATTRIBUTE',
    'IN:GET_MESSAGE', 'SL:CONTENT_EXACT', 'IN:GET_UNDERGRAD', 'IN:LIKE_MUSIC',
    'SL:ATTENDEE_EVENT', 'SL:CONTACT_ADDED', 'IN:SET_UNAVAILABLE',
    'IN:GET_EMPLOYMENT_TIME', 'IN:GET_EMPLOYER', 'IN:QUESTION_NEWS',
    'IN:SET_DEFAULT_PROVIDER_CALLING', 'IN:SNOOZE_ALARM',
    'IN:GET_CATEGORY_EVENT', 'IN:GET_ATTENDEE_EVENT', 'IN:RESUME_TIMER',
    'SL:RECIPES_UNIT_NUTRITION', 'IN:SET_RSVP_NO', 'SL:MUTUAL_SCHOOL',
    'SL:MUSIC_PLAYLIST_TITLE', 'IN:REMOVE_FROM_PLAYLIST_MUSIC',
    'IN:UPDATE_TIMER', 'IN:GET_EVENT_ATTENDEE', 'IN:DELETE_TIMER',
    'SL:MUSIC_GENRE', 'SL:MUSIC_TYPE', 'IN:UPDATE_ALARM', 'IN:RESUME_MUSIC',
    'SL:RECIPES_CUISINE', 'SL:ATTENDEE', 'SL:NAME_APP', 'SL:AGE', 'IN:PREFER',
    'SL:RECIPES_DISH', 'SL:RECIPES_COOKING_METHOD', 'IN:UPDATE_CALL',
    'SL:SIMILARITY', 'SL:EDUCATION_DEGREE', 'IN:GET_DATE_TIME_EVENT',
    'IN:GET_LANGUAGE', 'IN:GET_AIRQUALITY', 'IN:GET_LIFE_EVENT',
    'IN:RESUME_CALL', 'IN:GET_STORIES_NEWS', 'IN:GET_LOCATION',
    'SL:MUSIC_ARTIST_NAME', 'SL:RECIPES_EXCLUDED_INGREDIENT', 'IN:REPLAY_MUSIC',
    'IN:UPDATE_REMINDER', 'IN:GET_CALL', 'IN:GET_REMINDER_AMOUNT',
    'IN:GET_CALL_TIME', 'SL:PERIOD', 'IN:GET_WEATHER', 'IN:START_SHUFFLE_MUSIC',
    'SL:RECIPES_MEAL', 'IN:QUESTION_MUSIC', 'SL:RECIPES_SOURCE',
    'SL:RECIPES_TYPE', 'IN:ADD_TO_PLAYLIST_MUSIC', 'IN:GET_SUNSET',
    'IN:UPDATE_METHOD_CALL', 'IN:CREATE_PLAYLIST_MUSIC',
    'IN:GET_CONTACT_METHOD', 'IN:CANCEL_MESSAGE', 'IN:CREATE_TIMER',
    'SL:RECIPES_QUALIFIER_NUTRITION', 'SL:NEWS_SOURCE', 'IN:SEND_MESSAGE',
    'SL:SCHOOL', 'IN:UPDATE_REMINDER_TODO', 'IN:GET_ALARM',
    'SL:RECIPES_INCLUDED_INGREDIENT', 'IN:GET_DETAILS_NEWS', 'SL:AMOUNT',
    'IN:REPEAT_ALL_MUSIC', 'IN:FOLLOW_MUSIC', 'IN:GET_MAJOR', 'IN:STOP_MUSIC',
    'IN:SUBTRACT_TIME_TIMER', 'IN:GET_CONTACT', 'SL:DATE_TIME', 'SL:GROUP',
    'IN:REPEAT_ALL_OFF_MUSIC', 'SL:TYPE_RELATION', 'SL:MUSIC_RADIO_ID',
    'IN:HOLD_CALL', 'SL:PERSON_REMINDED', 'IN:PREVIOUS_TRACK_MUSIC',
    'SL:NEWS_TYPE', 'IN:LOOP_MUSIC', 'IN:UNLOOP_MUSIC', 'IN:GET_AVAILABILITY',
    'SL:RECIPES_DIET', 'SL:JOB', 'SL:MUSIC_PLAYLIST_MODIFIER', 'IN:GET_SUNRISE',
    'IN:GET_BIRTHDAY', 'IN:UPDATE_REMINDER_DATE_TIME', 'IN:DELETE_ALARM',
    'SL:USER_ATTENDEE_EVENT', 'SL:TYPE_CONTACT', 'IN:GET_INFO_RECIPES',
    'SL:ORDINAL', 'IN:IGNORE_CALL', 'IN:PLAY_MEDIA', 'SL:RECIPIENT',
    'SL:NEWS_TOPIC', 'IN:SET_RSVP_INTERESTED', 'SL:LIFE_EVENT',
    'SL:CATEGORY_EVENT', 'SL:MUSIC_ALBUM_TITLE', 'IN:SHARE_EVENT',
    'SL:LOCATION', 'SL:ATTRIBUTE_EVENT', 'SL:RECIPES_TYPE_NUTRITION',
    'SL:SENDER', 'IN:PLAY_MUSIC', 'IN:GET_GROUP', 'IN:UPDATE_REMINDER_LOCATION',
    'SL:MUSIC_REWIND_TIME', 'IN:MERGE_CALL', 'IN:GET_TIMER', 'IN:REWIND_MUSIC',
    'IN:PAUSE_TIMER', 'IN:FAST_FORWARD_MUSIC', 'IN:ANSWER_CALL',
    'IN:SKIP_TRACK_MUSIC', 'SL:TIMER_NAME', 'IN:SILENCE_ALARM',
    'IN:ADD_TIME_TIMER', 'IN:GET_REMINDER', 'IN:SET_DEFAULT_PROVIDER_MUSIC',
    'SL:RECIPES_TIME_PREPARATION'
]
INTENT_DICT = {'snips': SNIPS_INTENT, 'mtop': MTOP_INTENT}
SPECIAL_TOKENS = [f'<extra_id_{i}>' for i in range(100)
                 ] + SPECIAL_SENTENCEPIECE_TOKENS
for data_name, intent_list in INTENT_DICT.items():
  name_to_token_maps = {}
  token_to_name_maps = {}
  for idex, name in enumerate(SNIPS_MTOP_ARG_EDGES + intent_list):
    name_to_token_maps[name] = SPECIAL_TOKENS[idex]
    token_to_name_maps[SPECIAL_TOKENS[idex]] = name
  NAME_TO_TOKEN_MAPS[data_name] = name_to_token_maps
  TOKEN_TO_NAME_MAPS[data_name] = token_to_name_maps

# Nodes for SMCalflow, including 345 functional nodes, 15 special nodes,
# and 7 reference nodes.
SMCALFLOW_FUNC_NODES = [
    'NoneNode', 'adjustByPeriod', 'Constraint[Place]',
    'AttendeeListHasRecipientWithType', 'WeatherQueryApi', 'Midnight',
    'UpdateCommitEventWrapper', 'NeedsJacket', 'CreateCommitEventWrapper',
    'refer', 'NextWeekList', 'ReviseConstraint', 'toWeeks',
    'AlwaysTrueConstraint[LocationKeyphrase]',
    'AlwaysTrueConstraint[RespondComment]', 'inInches', 'inUsMilesPerHour',
    'Constraint[WeatherTablePlaces]', 'Constraint[Recipient]', 'FenceGibberish',
    'Constraint[Constraint[DateTime]]', 'LessThanFromStructDateTime',
    'WeatherQuantifier', 'adjustByDuration', 'RecipientFromRecipientConstraint',
    'DateTimeConstraint', 'HourMilitary', 'WeatherForEvent', 'IsWindy',
    'joinEventCommand', 'Now', 'Event', 'NextMonth', 'HourMinuteMilitary',
    'SeasonWinter', 'Acouple', 'IsSnowy', 'BottomResult',
    'CreatePreflightEventWrapper', 'WeatherProp', 'FenceScope',
    'Constraint[EventSpec]', 'FullYearofYear', 'SeasonSpring', 'NextPeriod',
    'AlwaysTrueConstraint[Holiday]', 'PlaceDescribableLocation',
    'alwaysTrueConstraintConstraint', 'GenericPleasantry', 'FindNumNextEvent',
    'ConfirmDeleteAndReturnAction', 'SetOtherOrganizer', 'Here', 'HolidayYear',
    'WillRain', 'Constraint[Constraint[Time]]', 'IsBusy', 'PersonName',
    'WeatherPleasantry', 'CurrentUser', 'FullMonthofPreviousMonth',
    'previousHoliday', 'FindTeamOf', 'inKilometersPerHour', 'not',
    'AlwaysTrueConstraint[String]', 'EventOnDateAfterTime', 'EarlyDateRange',
    'Today', 'NewClobber', 'Constraint[PleasantryCalendar]', 'RepeatAgent',
    'EventOnDateWithTimeRange', 'Early', 'TimeSinceEvent',
    'Constraint[Constraint[Recipient]]', 'toFourDigitYear',
    'ConfirmUpdateAndReturnActionIntension', 'EventDuringRangeTime',
    'callCreateCommitEventWrapper', 'previousMonthDay',
    'Constraint[Constraint[Event]]', 'FullMonthofMonth', 'nextMonthDay',
    'ChooseUpdateEventFromConstraint', 'UtteranceTemplate', 'let',
    'Constraint[Event]', 'AroundDateTime', 'toRecipient', 'WhenProperty',
    'HourMinuteAm', 'IsSunny', 'EventRescheduled', 'FindPlace', 'TopResult',
    'FenceTriviaQa', 'LateMorning', 'FenceRecurring',
    'DeleteCommitEventWrapper', 'AlwaysTrueConstraint[ShowAsStatus]',
    'subtractDurations', 'EventDuringRangeDateTime', 'AttendeesWithNotResponse',
    'IsHighUV', 'TimeToTime', 'ChooseCreateEventFromConstraint',
    'LastWeekendOfMonth', 'OnDateBeforeTime', 'toMonth', 'WeekendOfDate',
    'DoNotConfirm', 'EndOfWorkDay', 'AlwaysTrueConstraint[List[Attendee]]',
    'callCreatePreflightEventWrapper', 'toMonths', 'ConfirmAndReturnAction',
    'AlwaysFalseConstraint[LocationKeyphrase]', 'RecipientWithNameLike',
    'DateTimeAndConstraint', 'FenceConditional', 'exists',
    'AlwaysTrueConstraint[Duration]', 'Constraint[Time]', 'IsTeamsMeeting',
    'andConstraint', 'ChooseUpdateEvent', 'EventForRestOfToday', 'WillSleet',
    'NumberWeekOfMonth', 'RespondComment', 'Afew',
    'QueryEventIntensionConstraint', 'or', 'UpdatePreflightEventWrapper',
    'ShowAsStatus', 'Path', 'List[Path]', 'Yesterday', 'MDY', 'ThisWeekend',
    'TimeAround', 'TimeAfterDateTime', 'AlwaysTrueConstraint[WeatherTable]',
    'IsFree', 'List[Any]', 'FindReports', 'NextDOW', 'PlaceFeature',
    'NumberWeekFromEndOfMonth', 'Breakfast', 'append', 'Constraint[DateTime]',
    'PeriodDurationBeforeDateTime', 'ClosestMonthDayToDate',
    'Constraint[Constraint[Date]]', 'String', 'EventBeforeDateTime', 'MD',
    'FindPlaceAtHere', 'FenceTeams', 'PleasantryAnythingElseCombined', 'Night',
    'List[Recipient]', 'Afternoon', 'allows', 'Weekdays', 'Boolean', 'toYears',
    'PeriodDuration', 'ChoosePersonFromConstraint', 'FenceAggregation',
    'AttendeesWithResponse', 'NextWeekend', 'WeatherAggregate', 'nextHoliday',
    'DowOfWeekNew', 'DateAtTimeWithDefaults', 'Later', 'nextDayOfMonth',
    'NumberPM', 'SeasonSummer', 'inFahrenheit', 'roomRequest', 'Tomorrow',
    'toMinutes', 'WeekendOfMonth', 'LateAfternoon',
    'AlwaysFalseConstraint[ShowAsStatus]', 'singleton', 'EventAfterDateTime',
    'Noon', 'Weekend', 'nextDayOfWeek', 'EventDuringRange', 'buildUtterance',
    'EarlyTimeRange', 'SeasonFall', 'FenceReminder', 'DateTime',
    'ConfirmCreateAndReturnAction', 'MonthDayToDay', 'PersonFromRecipient',
    'FindPlaceMultiResults', 'numberToIndexPath', 'PeriodBeforeDate',
    'UserPauseResponse', 'PleasantryCalendar', 'EventOnDateFromTimeToTime',
    'LateDateRange', 'Constraint[PlaceSearchResponse]', 'negate', 'Time',
    'Number', 'AttendeeListHasPeople', 'Late', 'ConvertTimeToPM',
    'FenceWeather', 'cursorNext', 'EventSometimeOnDate', 'previousDayOfWeek',
    'DeletePreflightEventWrapper', 'LocationKeyphrase', 'intension',
    'LastPeriod', 'AlwaysTrueConstraint[ResponseStatusType]',
    'AlwaysTrueConstraint[Number]', 'PlaceHasFeature', 'DowToDowOfWeek', 'size',
    'OnDateAfterTime', 'ResponseStatusType', 'EventOnDateBeforeTime', 'take',
    'WeekOfDateNew', 'Yield', 'FullMonthofLastMonth', 'toFahrenheit',
    'DateTimeAndConstraintBetweenEvents', 'EventOnDate', 'Execute', 'minBy',
    'LateTimeRange', 'and', 'NumberAM', 'ChooseCreateEvent', 'Month',
    'DateAndConstraint', 'DateTimeFromDowConstraint',
    'AttendeeListHasPeopleAnyOf', 'ClosestDayOfWeek', 'FenceDateTime', 'Latest',
    'NextHolidayFromToday', 'FenceAttendee', 'LastYear', 'EventBetweenEvents',
    'EventAtTime', 'do', 'AttendeeResponseStatus', 'Holiday', 'NextYear',
    'DayOfWeek', 'orConstraint', 'IsStormy', 'update', 'Brunch', 'toHours',
    'EventOnDateTime', 'IsCloudy', 'ActionIntensionConstraint', 'inCelsius',
    'NextTime', 'AgentStillHere', 'Morning', 'LastWeekNew', 'HourMinutePm',
    'Lunch', 'Constraint[Date]', 'RecipientAvailability', 'ForwardEventWrapper',
    'AttendeeListHasRecipientConstraint', 'toDays', 'AttendeeListHasRecipient',
    'toSeconds', 'FindLastEvent', 'AlwaysFalseConstraint[List[Attendee]]',
    'addDurations', 'adjustByPeriodDuration', 'RespondShouldSend',
    'ConvertTimeToAM', 'PersonOnTeam', 'previousDayOfMonth', 'roleConstraint',
    'FenceSpecify', 'addPeriodDurations', 'CreatePersonWithEmail',
    'AlwaysTrueConstraint[EventId]', 'AlwaysTrueConstraint[Temperature]',
    'FenceConferenceRoom', 'EventAllDayStartingDateForPeriod', 'Date',
    'FenceNavigation', 'AttendeeListExcludesRecipient', 'FenceSwitchTabs',
    'AttendeeFromEvent', 'WillSnow', 'IsCold', 'FencePeopleQa',
    'NumberInDefaultTempUnits', 'Constraint[FenceScope]', 'Earliest', 'Evening',
    'get', 'TimeBeforeDateTime', 'FencePlaces', 'IsHot',
    'AlwaysTrueConstraint[Length]', 'AtPlace', 'Future', 'EventAttendance',
    'Dinner', 'extensionConstraint', 'IsRainy', 'EventAllDayOnDate',
    'GreaterThanFromStructDateTime', 'takeRight',
    'UpdateEventIntensionConstraint', 'FenceMultiAction', 'ThisWeek',
    'NextPeriodDuration', 'Constraint[CreateCommitEvent]', 'CancelScreen',
    'AlwaysTrueConstraint[RespondShouldSend]', 'ClosestDay',
    'FindEventWrapperWithDefaults', 'FindManager', 'listSize',
    'EventAllDayForDateRange', 'FenceOther', 'LastDayOfMonth', 'LastDuration',
    'send', 'cursorPrevious', 'AttendeeType'
]
SMCALFLOW_SPECIAL_NODES = [
    '_equal', '_add', '_q_equal', '_q_t_qual', '_less', '_greater', '_reader',
    '_l_equal', '_q_g_equal', '_minus', '_g_eq', '_q_greater', '_q_l_equal',
    '_empty_list', '_q_less'
]
SMCALFLOW_REFERENCE_NODES = ['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6']
SMCALFLOW_NODES = (
    SMCALFLOW_FUNC_NODES + SMCALFLOW_SPECIAL_NODES + SMCALFLOW_REFERENCE_NODES)

# Edges for SMCalflow, including 95 edges.
SMCALFLOW_ARG_EDGES = [
    ':NoneARG', ':carg', ':organizer', ':people', ':oldLocation', ':keyphrase',
    ':time1', ':property', ':period', ':name', ':recipient', ':emailAddress',
    ':event1', ':value', ':subject', ':event2', ':dayOfWeek', ':dateTime1',
    ':attendees', ':week', ':hour', ':update', ':intension', ':startDate',
    ':rating', ':formattedAddress', ':end', ':time', ':type', ':slotConstraint',
    ':radiusConstraint', ':wt', ':phoneNumber', ':duration', ':output',
    ':nonEmptyBase', ':dateTimeConstraint', ':isOneOnOne', ':url',
    ':eventConstraint', ':num1', ':month', ':num', ':timeRange', ':team',
    ':includingMe', ':rootLocation', ':day2', ':dateRange', ':date1', ':range',
    ':new', ':feature', ':recipients', ':attendee', ':person', ':email',
    ':start', ':number', ':data', ':place', ':showAs', ':day', ':response',
    ':minutes', ':officeLocation', ':index', ':price', ':day1',
    ':periodDuration', ':table', ':dow', ':item', ':event', ':year',
    ':isAllDay', ':num2', ':responseStatus', ':dateTime2', ':date',
    ':recipientConstraint', ':time2', ':constraint', ':hours', ':location',
    ':date2', ':eventCandidates', ':quantifier', ':sendResponse', ':dateTime',
    ':holiday', ':results', ':dowConstraint', ':id', ':comment'
]

token_to_name_map = {}
name_to_token_map = {}
for idex, name in enumerate(SMCALFLOW_ARG_EDGES + SMCALFLOW_NODES):
  name_to_token_maps[name] = SPECIAL_TOKENS[idex]
  token_to_name_maps[SPECIAL_TOKENS[idex]] = name
NAME_TO_TOKEN_MAPS['smcalflow'] = name_to_token_maps
TOKEN_TO_NAME_MAPS['smcalflow'] = token_to_name_maps


def single_token_transfer(token: Text, data_version: str = 'v0') -> Text:
  """Transfers single special tokens to their original tokens."""
  token_map = TOKEN_TO_NAME_MAPS[data_version]
  pattern = re.compile(r'<extra_id_[0-9]+>')
  search_result = re.search(pattern, token)
  retoken = token
  if search_result:
    # The token is in MISC + ARG_EDGES + FUNC_NODES.
    match_str = search_result.group(0)
    if match_str:
      retoken = token_map[match_str]
  elif token in token_map:
    retoken = token_map[token]
  return retoken


def token_transfer(graph_str: Text, data_version: str = 'v0') -> Text:
  """Transfers special tokens in a graph string to their original tokens."""
  token_map = TOKEN_TO_NAME_MAPS[data_version]
  new_graph_list = []
  for token in graph_str.split():
    search_result = re.search(re.compile(r'<extra_id_[0-9]+>'), token)
    if search_result:
      # The token is in MISC + ARG_EDGES + FUNC_NODES.
      match_str = search_result.group(0)
      if match_str:
        retoken = token_map[match_str]
        if retoken == '-of' and new_graph_list:
          retoken = new_graph_list.pop() + '-of'
      else:
        retoken = token
    elif token.endswith('_'):
      # The token here is a content node.
      postfix = '_'.join(token.split('_')[:-2])
      lemma = token.split('_')[-2]
      if postfix in token_map:
        retoken = '_' + lemma + '_' + token_map[postfix]
      elif lemma in token_map:
        retoken = '_' + '<nolemma>' + '_' + token_map[lemma]
      else:
        retoken = '_' + lemma + '_' + postfix
    else:
      retoken = token
    new_graph_list.append(retoken)
  graph_str = ' '.join(new_graph_list)
  return graph_str


def merge_token_prob(token_list: List[Text],
                     beam_scores: List[float],
                     data_version: str = 'v0') -> List[Dict[Text, Any]]:
  """Merges tokens to graph subgraphs (nodes/edges), and sums up beam scores.

  For example, for tokens ['p', '_', 'down', '_'],
  and beam scores [0.0, -1.1920928955078125e-07, 0.0, 0.0],
  we will merge tokens into 'p_down_', and compute the corresponding
  probability exp(sum([0.0, -1.1920928955078125e-07, 0.0, 0.0])).

  Args:
    token_list: a list of tokens to be merged.
    beam_scores: a list of beam scores for each token position, the length is
      equal to the length of `token_list`.
    data_version: DeepBank version.

  Returns:
    subgraph_infos: a list of dictionaries, which contiains the values and
      probaibilties.
  """
  subgraph_infos = []
  # If the node/edge name is not finished, store the previous tokens
  # in `node_stack`.
  node_stack = []
  # Records each token's start index and end index.
  start, end = 0, 0
  # Checks if the current token is in quotes.
  start_quote = False
  for i, token in enumerate(token_list):
    token = token.replace(' ', '')
    end_symbol_case = token and token[0] not in ['(', ')', '"', '*']
    edge_case = token not in ARG_EDGES
    piece_case1 = token not in FUNC_NODES[data_version] + [
        'polite', 'addressee']
    piece_case2 = i + 1 < len(token_list) and token_list[i + 1] == '_'
    piece_case3 = i > 0 and token_list[i - 1] == '_'
    func_node_case = piece_case1 or piece_case2 or piece_case3
    if not token:
      end += 1
      continue
    elif not start_quote and token == '-of':
      end += 1
      previous_info = subgraph_infos.pop()
      subgraph_infos.append({
          'value': previous_info['value'] + token,
          'prob': np.exp(sum(beam_scores[start - 1:end])),
          'align': '%s-%s' % (start - 1, end)
      })
      start = end
    elif token == '"':
      # The start or end of a double quote.
      if token_list[:i].count('"') % 2 == 0:
        start_quote = True
      else:
        start_quote = False
      if not start_quote and node_stack:
        subgraph_infos.append({
            'value': ''.join(node_stack),
            'prob': np.exp(sum(beam_scores[start:end])),
            'align': '%s-%s' % (start, end)
        })
        node_stack = []
        start = end
      end += 1
      # Adds subgraph info for quote, which is non-mergable symbol.
      subgraph_infos.append({
          'value': token,
          'prob': np.exp(sum(beam_scores[start:end])),
          'align': '%s-%s' % (start, end)
      })
      start = end
    elif start_quote or (end_symbol_case and edge_case and func_node_case):
      # Merges the pieces of node/attribute name in to a full name,
      # e.g., ['p', '_', 'down', '_'] into 'p_down_'.
      # `end_symbol_case`: the token is an end symbol (brackets,
      #   double quote or star)
      # `edge_case`: the token is an argument edge.
      # `func_node_case`: the token is a functional node. Ensures that
      #   pieces of node/attribute name are not included in the
      #   function nodes (func_node_case), e.g., 'comp' in 'compact'.
      node_stack.append(token)
      end += 1
    else:
      # Gets non-mergable symbol, first write the merged node from node_stack,
      # and then write the non-mergable symbol.
      # Example: for tokens, 'comp', 'act', ')', node_stack = ['comp', 'act'].
      # We first write node name 'compact', and then write non-mergable
      # symbol ')'.
      if node_stack:
        subgraph_infos.append({
            'value': ''.join(node_stack),
            'prob': np.exp(sum(beam_scores[start:end])),
            'align': '%s-%s' % (start, end)
        })
        node_stack = []
        start = end
      end += 1
      subgraph_infos.append({
          'value': token,
          'prob': np.exp(sum(beam_scores[start:end])),
          'align': '%s-%s' % (start, end)
      })
      start = end
  if node_stack:
    # The graph is incomplete and `node_stack` has something left.
    subgraph_infos.append({
        'value': ''.join(node_stack),
        'prob': np.exp(sum(beam_scores[start:end])),
        'align': '%s-%s' % (start, end)
    })
  return subgraph_infos


def assign_prob_to_penman(subgraph_infos: List[Dict[Text, Any]],
                          data_version: str = 'v0') -> Text:
  """Assigns the probability to each node/edge in the PENMAN string.

  Example input: ( unknown :ARG1 ( _look_v_1 ) )
  Example output: ( unknown_1.0 :ARG1_0.9999 ( _look_v_1_0.9987 ))

  Args:
    subgraph_infos: A list of dictionaries, which contiains the values and
      probaibilties.
    data_version: DeepBank data version.

  Returns:
    A list of graph string.
  """
  graph_str_list = []
  quote_count = 0
  for subgraph_info in subgraph_infos:
    token = subgraph_info['value']
    prob = subgraph_info['prob']
    if token in ARG_EDGES and token != ':carg':
      # The token here is an edge.
      token_prob = token + '_' + str(prob)
    elif '-of' in token and token.split(
        '-of')[0] in ARG_EDGES and quote_count % 2 == 0:
      # The token here is a reversed version of edge, e.g., 'ARG1-of'.
      token_prob = token[:-3] + '_' + str(prob) + '-of'
    elif token in FUNC_NODES[data_version] + ['polite', 'addressee'
                                             ] and quote_count % 2 == 0:
      # The token here is a functional node, e.g., 'pron'.
      token_prob = token + '_' + str(prob)
    elif token[-1] == '_' and quote_count % 2 == 0:
      # The token here is a surface node, e.g., 'v_1_look_'.
      # Here we need to reorder the node to '_look_v_1'.
      lemma = token.split('_')[-2]
      postfix = '_'.join(token.split('_')[:-2])
      token_prob = '_' + lemma + '_' + postfix + '_' + str(prob)
    elif '*' in token and quote_count % 2 == 0:
      previous_component = graph_str_list.pop()
      previous_token = '_'.join(previous_component.split('_')[:-1])
      try:
        previous_prob = float(previous_component.split('_')[-1])
        token_prob = previous_token + '*' * token.count('*') + '_' + str(
            previous_prob * prob)
      except ValueError:
        logging.warning('Unable to retrieve prob in previous '
                        'component %s.', previous_component)
        token_prob = previous_token + '*' * token.count('*') + '_' + str(prob)
    elif token in ['(', ')', ':carg']:
      # For those symbol, there is no need to assign probabilities.
      token_prob = token
    elif token == '"':
      quote_count += 1
      token_prob = token
    else:
      token_prob = token + '_' + str(prob)
    graph_str_list.append(token_prob)
  return ' '.join(graph_str_list)


def post_processing(graph_str: Text) -> Text:
  """Post-processing for generating variable-free linearized graphs."""
  # Merges the quote to the value of attributes.
  # Example: " John " to "John".
  graph_str = re.sub(r'" ([\S]*) "', r'"\1"', graph_str)
  # Handles peculiar tokens generated by the model, e.g., " ⁇ ".
  graph_str = graph_str.replace(' ⁇ ', '⁇')
  graph_str = graph_str.replace(' *', '*')

  if graph_str.split()[-1][0] in ['(', ':']:
    # The graph is incomplete, i.e., end with a left bracket or edge.
    # Example: '( unknown :ARG' or '( unknown :ARG ('.
    last_right_bracket_index = graph_str.rfind(')')
    graph_str = graph_str[:last_right_bracket_index + 1]

  # The number of left/right bracket is for matching the brackets.
  num_left_bracket, num_right_bracket = 0, 0
  # The `quote_count` is for check if current token is in quote
  # (attribute value). The bracket in quote does not count towards
  # total number of brackets.
  num_quote = 0
  new_graph_str = ''
  for x in graph_str:
    new_graph_str += x
    if x == '"':
      num_quote += 1
    if x == '(' and num_quote % 2 == 0:
      num_left_bracket += 1
    if x == ')' and num_quote % 2 == 0:
      num_right_bracket += 1
    if num_right_bracket == num_left_bracket:
      # If the number of right bracket has reached the number of
      # left brackets, the rest of the graph become illegal and
      # we just drop it.
      break
  if num_left_bracket > num_right_bracket:
    # After going through the whole graph string, if the number of left
    # brackets is greater than the number of right brackets,
    # we need to match the number of left brackets.
    new_graph_str += ' )' * (num_left_bracket - num_right_bracket)
  graph_str = new_graph_str
  return graph_str


def transfer_to_penman(graph_str: Text) -> Text:
  """Tranfers the variable-free linearized graph to penman style.

  Args:
    graph_str: variable-free linearized graph, e.g.,
      "( unknown :ARG ( _book_n_1 ) )".

  Returns:
    penman_graph_str: e.g., "( x0 / unknown :ARG ( x1 / _book_n_1 ) )".
  """
  graph_str = post_processing(graph_str)
  graph_str_list = []
  node_dict = {}
  count = 0
  for i, x in enumerate(graph_str.split()):
    if x[0] not in ['(', ')', ':', '"']:
      # x here is a node.
      if '*' in x:
        # Address coreference.
        # Example: replace 'unknown*' to 'x0' if previously
        # we defined '( x0 / unknown* )'.
        # There are two different versions of inputs,
        # [1] Without probabilities, e.g., unknown**.
        # [2] With probabilities, e.g., unknown**_1.0.
        # Here we need retrieve the node name 'unknown*'.
        last_star_index = x.rfind('*')
        node_name = x[:last_star_index + 1]
        if node_name not in node_dict:
          # The node name has not been defined previously.
          node_id = 'x' + str(count)
          node_dict[node_name] = node_id
          graph_str_list.append(node_id + ' / ' + x.replace('*', ''))
          count += 1
        else:
          # The node name has been defined previously, replace the
          # node name to its index.
          # Example '( unknown* )' -> 'x0'.
          graph_str_list.append(node_dict[node_name])
      else:
        graph_str_list.append('x' + str(count) + ' / ' + x)
        count += 1
    else:
      graph_str_list.append(x)
  graph_str = ' '.join(graph_str_list)

  # Addresses the duplicate coreference bracket issues.
  # Example: :ARG1 ( x0 ) -> :ARG1 x0.
  for _, v in node_dict.items():
    graph_str = graph_str.replace('( %s )' % v, v)

  # Addresses the duplicate coreference bracket issues.
  # Example: :ARG1 ( x0 :BV-of ( ... ) ) -> :ARG1 x0 :BV-of ( ... ).
  for _, v in node_dict.items():
    while '( %s :' % v in graph_str:
      index_left_bracket = graph_str.index('( %s :' % v)
      num_left_bracket, num_right_bracket = 0, 0
      for i in range(index_left_bracket, len(graph_str)):
        if graph_str[i] == '(':
          num_left_bracket += 1
        if graph_str[i] == ')':
          num_right_bracket += 1
        if num_left_bracket == num_right_bracket:
          # Removes the duplicate left bracket.
          graph_str = graph_str[:index_left_bracket] + graph_str[
              index_left_bracket + 2:]
          # Removes the duplicate right bracket.
          graph_str = graph_str[:i-2] + graph_str[i:]
          break
  return graph_str


def graph_to_nodeseq(graph_str: Text,
                     data_version: str = 'v0') -> Dict[Text, List[Text]]:
  """Extracting node sequence from a graph string."""
  nodeseq = []
  cont_nodeseq = []
  func_nodeseq = []
  entity_nodeseq = []
  graph_str_split = graph_str.split()
  for x in graph_str_split:
    if x in ['(', ')', '"']:
      continue
    if x[0] == ':':
      continue
    if x[0] == '*':
      node_name = nodeseq.pop()
      nodeseq.append(node_name + x)
      continue
    nodeseq.append(x)

  for x in nodeseq:
    if x[0] == '_':
      cont_nodeseq.append(x)
    elif x in FUNC_NODES[data_version]:
      func_nodeseq.append(x)
    else:
      entity_nodeseq.append(x)

  return dict(
      all=nodeseq, cont=cont_nodeseq, func=func_nodeseq, entity=entity_nodeseq)


def find_root(graph_str: Text) -> Text:
  """Find the root node of the graph."""
  root = ''
  for x in graph_str.split():
    if root == '' and x[0] not in ['(', ')', '"', '*', ':']:  # pylint:disable=g-explicit-bool-comparison
      root = x
      break
  return root
