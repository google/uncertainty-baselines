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

"""Default config for VRNN experiments."""

import os
from typing import Optional

from ml_collections import config_dict
import data_utils  # local file import from experimental.language_structure.vrnn
import model_config  # local file import from experimental.language_structure.vrnn

# Base directory containing model configs for different datasets.
# Method get_config_dir(dataset) returns the config directory of a specific
# dataset, where we place the following config files:
# - label sampling config defining the percentage/shots of each class of label
#   to sample in the few-shot training.
# - word-in-vocab weight file.
# - vocabulary file.
_CONFIG_BASE_DIR = './datasets'



def get_config_dir(dataset: str) -> str:
  """Returns the directory containing the config files."""
  return os.path.join(_CONFIG_BASE_DIR, dataset)


def _create_model_config(
    dataset: str,
    num_states: int,
    with_bow: bool,
    shared_bert_embedding: bool,
    bert_dir: Optional[str] = '',
    word_embedding_path: Optional[str] = ''
) -> model_config.VanillaLinearVRNNConfig:
  """Create model config with hyperparemeters overwritten by flag values."""
  data = dict(
      vae_cell=dict(
          max_seq_length=data_utils.get_dataset_max_seq_length(dataset),
          num_states=num_states,
          shared_bert_embedding=shared_bert_embedding,
          word_embedding_path=word_embedding_path,
      ),
      with_bow=with_bow,
      max_dialog_length=data_utils.get_dataset_max_dialog_length(dataset),
  )

  if shared_bert_embedding:
    data['vae_cell']['shared_bert_embedding_ckpt_dir'] = os.path.join(
        bert_dir, 'bert_model.ckpt')

  return model_config.vanilla_linear_vrnn_config(**data)


def get_config(dataset: str,
               num_states: Optional[int] = None,
               with_bow: Optional[bool] = True,
               shared_bert_embedding: Optional[bool] = False,
               bert_embedding_type: Optional[str] = 'base',
               bert_dir: Optional[str] = '') -> config_dict.ConfigDict:
  """Returns the configuration for this experiment.

  Args:
    dataset: dataset name.
    num_states: number of the latent dialog states of the model.
    with_bow: whether to enable BoW loss.
    shared_bert_embedding: whether to use BERT as the shared embedding layer.
    bert_embedding_type:  the type of Bert model for the embedding layer.
      See http://shortn/_PzBKxLRgDl for details.
    bert_dir: the directory contains pretrained BERT TF checkpoints.

  Returns:
    A ConfigDict containing all configs for the experiment.
  """
  config = config_dict.ConfigDict()
  config_dir = get_config_dir(dataset)

  # config.max_per_task_failures = -1
  # config.max_task_failures = 10

  config.platform = 'jf'
  config.tpu_topology = '2x2'

  config.seed = 8

  config.dataset = dataset
  config.dataset_dir = data_utils.get_dataset_dir(dataset)

  config.train_epochs = 10
  config.train_batch_size = 16
  config.eval_batch_size = 16
  # Batch size for inference. Predicting in batches in case of OOM.
  config.inference_batch_size = 300
  # Seed used to generate datasets for inference.
  config.inference_seed = 9527
  # Directory storing the saved model and model prediction outputs.
  config.model_base_dir = None
  # Maximum number of evaluation cycles with the primary metric worse than the
  # current best to tolerate before early stopping.
  # Disable it and run fixed epochs training by setting it to some value < 0
  # (e.g., -1)
  config.patience = 3
  # The minimal difference to be counted as improvement on the metric.
  config.min_delta = 0.01

  # Number of epochs between saving checkpoints. Use -1 to never save
  # checkpoints.
  config.checkpoint_interval = 5
  # Number of epochs between evaluation.
  config.evaluation_interval = 5

  config.base_learning_rate = 5e-4
  config.one_minus_momentum = 0.1

  config.classification_loss_weight = 0.
  config.kl_loss_weight = 1.
  config.bow_loss_weight = 0.5
  # Whether to use batch prior regularization in KL loss.
  config.with_bpr = True
  # Whether to use Bag-of-Word loss.
  config.with_bow = with_bow

  # Path to the JSON file defining the percentage/shots of each class of label
  # to be used to compute the classification loss. Defaults to 0. It should
  # also specify the sampling mode by setting {"mode": "ratios"|"shots"}.
  config.label_sampling_path = os.path.join(config_dir,
                                            'label_ratio_0_shots.json')

  config.shared_bert_embedding = shared_bert_embedding


  # The TF Hub url for preprocessing inputs for the embedding Bert.
  config.bert_embedding_preprocess_tfhub_url = (
      'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3')
  config.bert_dir = bert_dir
  # Path to the pre-trained embedding.
  word_embedding_path = data_utils.get_word_embedding_path(dataset)

  config.vocab_file_path = os.path.join(
      bert_dir, 'vocab.txt') if config.shared_bert_embedding else os.path.join(
          config_dir, 'vocab.txt')

  if not num_states:
    num_states = data_utils.get_dataset_num_latent_states(dataset)
  config.model = _create_model_config(dataset, num_states, config.with_bow,
                                      config.shared_bert_embedding, bert_dir,
                                      word_embedding_path)

  # Weight of the word weights from word_weights_path used to interpolate with
  # uniform weight (1 / vocab_size). It should be between 0 and 1. The final
  # word weight is w * word_weight + (1 - w) * 1 / vocab_size.
  config.word_weights_file_weight = 1.
  # Path to the word-in-vocab weight file.
  config.word_weights_path = os.path.join(
      config_dir, 'word_weights_bert_en_uncased_base.npy'
      if config.shared_bert_embedding else 'word_weights.npy')

  config.psl_constraint_learning_weight = 0.
  config.psl_constraint_inference_weight = 0.
  # Number of iterations we apply PSL model in the inference.
  config.psl_constraint_inference_steps = 0
  config.psl_constraint_rule_names = []
  config.psl_constraint_rule_weights = []
  config.psl_config_file = ''
  config.psl = {}

  config.hidden_state_model_learning_rate = 1e-3
  config.hidden_state_model_train_epochs = 10

  config.few_shots = [1, 5, 10]
  config.few_shots_trials = 100
  # L2 regularization weights for the few-shot regression.
  config.few_shots_l2_weights = [2.0**i for i in range(-10, 20)]

  return config
