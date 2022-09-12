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

"""Tests for data_preprocessor."""

from absl.testing import absltest
from absl.testing import parameterized
import more_itertools
import tensorflow as tf
from uncertainty_baselines.datasets import datasets
from vrnn import data_preprocessor  # local file import from experimental.language_structure
from vrnn import data_utils  # local file import from experimental.language_structure
from vrnn import utils  # local file import from experimental.language_structure

INPUT_ID_NAME = data_preprocessor.INPUT_ID_NAME
INPUT_MASK_NAME = data_preprocessor.INPUT_MASK_NAME
DIAL_TURN_ID_NAME = data_preprocessor.DIAL_TURN_ID_NAME
USR_UTT_NAME = data_preprocessor.USR_UTT_NAME
SYS_UTT_NAME = data_preprocessor.SYS_UTT_NAME
USR_UTT_RAW_NAME = data_preprocessor.USR_UTT_RAW_NAME
SYS_UTT_RAW_NAME = data_preprocessor.SYS_UTT_RAW_NAME


class CreateUtteranceFeatureTest(absltest.TestCase):

  def _create_input_tensor(self, dialog_length, seq_length) -> tf.Tensor:
    inputs = {
        USR_UTT_NAME:
            tf.keras.Input(shape=(dialog_length, seq_length)),
        SYS_UTT_NAME:
            tf.keras.Input(shape=(dialog_length, seq_length)),
        USR_UTT_RAW_NAME:
            tf.keras.Input(shape=(dialog_length,), dtype=tf.string),
        SYS_UTT_RAW_NAME:
            tf.keras.Input(shape=(dialog_length,), dtype=tf.string),
    }

    return inputs

  def test_feature_output_shape(self):
    dialog_length = 2
    seq_length = 5
    inputs = self._create_input_tensor(dialog_length, seq_length)

    outputs = data_preprocessor.create_utterance_features(inputs)

    self.assertLen(outputs, 2)
    for output in outputs:
      for key in [INPUT_ID_NAME, INPUT_MASK_NAME]:
        self.assertEqual([None, dialog_length, seq_length],
                         output[key].shape.as_list())

  def test_bert_feature_output_shape(self):
    dialog_length = 2
    seq_length = 5
    inputs = self._create_input_tensor(dialog_length, seq_length)

    preprocess_tfhub_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'
    bert_preprocess_model = utils.BertPreprocessor(preprocess_tfhub_url,
                                                   seq_length)

    outputs = data_preprocessor.create_bert_utterance_features_fn(
        bert_preprocess_model)(
            inputs)

    self.assertLen(outputs, 2)
    for output in outputs:
      for key in [INPUT_ID_NAME, INPUT_MASK_NAME]:
        self.assertEqual([None, dialog_length, seq_length],
                         output[key].shape.as_list())


class DataPreprocessorTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.batch_size = 2

  def _load_dataset(self, dataset_name):
    dataset_builder = datasets.get(
        dataset_name, split='test', add_dialog_turn_id=True)
    return dataset_builder.load(batch_size=self.batch_size).prefetch(1)

  @parameterized.named_parameters(('multiwoz_synth', 'multiwoz_synth'),
                                  ('simdial', 'simdial'),
                                  ('sgd_synth', 'sgd_synth'), ('sgd', 'sgd'),
                                  ('sgd_da', 'sgd_domain_adapation'))
  def test_output_shape(self, dataset_name):
    dataset = self._load_dataset(dataset_name)
    dialog_length = data_utils.get_dataset_max_dialog_length(dataset_name)
    seq_length = data_utils.get_dataset_max_seq_length(dataset_name)
    num_states = data_utils.get_dataset_num_latent_states(dataset_name)

    encoder_feature_fn = data_preprocessor.create_utterance_features
    decoder_feature_fn = data_preprocessor.create_utterance_features
    preprocessor = data_preprocessor.DataPreprocessor(encoder_feature_fn,
                                                      decoder_feature_fn,
                                                      num_states)
    dataset = dataset.map(preprocessor.create_feature_and_label)
    (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2, label,
     label_mask, initial_state, initial_sample, domains,
     _) = more_itertools.first(dataset)

    domain_label, _ = domains

    for inputs in [
        encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2
    ]:
      for key in [INPUT_ID_NAME, INPUT_MASK_NAME]:
        self.assertEqual([self.batch_size, dialog_length, seq_length],
                         inputs[key].shape.as_list())

    for inputs in [label, label_mask, domain_label]:
      self.assertEqual([self.batch_size, dialog_length], inputs.shape.as_list())

    for inputs in [initial_state, initial_sample]:
      self.assertEqual([self.batch_size, num_states], inputs.shape.as_list())

  @parameterized.named_parameters(('multiwoz_synth', 'multiwoz_synth'),
                                  ('simdial', 'simdial'),
                                  ('sgd_synth', 'sgd_synth'))
  def test_label_mask_by_dialog_turn_ids(self, dataset_name):
    dataset = self._load_dataset(dataset_name)
    inputs = more_itertools.first(dataset)
    dialog_turn_id_indices = [(0, 2), (1, 3), (1, 5)]
    dialog_turn_ids = tf.gather_nd(inputs[DIAL_TURN_ID_NAME],
                                   dialog_turn_id_indices)
    num_states = data_utils.get_dataset_num_latent_states(dataset_name)

    encoder_feature_fn = data_preprocessor.create_utterance_features
    decoder_feature_fn = data_preprocessor.create_utterance_features
    preprocessor = data_preprocessor.DataPreprocessor(
        encoder_feature_fn,
        decoder_feature_fn,
        num_states,
        labeled_dialog_turn_ids=dialog_turn_ids)
    dataset = dataset.map(preprocessor.create_feature_and_label)
    (_, _, _, _, _, label_mask, _, _, _, _) = more_itertools.first(dataset)

    for i, row in enumerate(label_mask.numpy()):
      for j, val in enumerate(row):
        if (i, j) in dialog_turn_id_indices:
          self.assertEqual(val, 1)
        else:
          self.assertEqual(val, 0)

  @parameterized.named_parameters(('sgd', 'sgd'))
  def test_in_domain_mask(self, dataset_name):
    in_domains = [2, 3]
    num_states = data_utils.get_dataset_num_latent_states(dataset_name)

    encoder_feature_fn = data_preprocessor.create_utterance_features
    decoder_feature_fn = data_preprocessor.create_utterance_features
    preprocessor = data_preprocessor.DataPreprocessor(
        encoder_feature_fn,
        decoder_feature_fn,
        num_states,
        in_domains=tf.constant(in_domains))

    dataset = self._load_dataset(dataset_name)
    dataset = dataset.map(preprocessor.create_feature_and_label)
    (_, _, _, _, _, _, _, _, domains, _) = more_itertools.first(dataset)

    domain_label_id, ind_mask = domains
    for domain_label, mask in zip(domain_label_id.numpy(), ind_mask.numpy()):
      for l, m in zip(domain_label, mask):
        if l in in_domains:
          self.assertEqual(m, 1)
        else:
          self.assertEqual(m, 0)


if __name__ == '__main__':
  absltest.main()
