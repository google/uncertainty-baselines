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

r"""Training script of VanillaLinearVRNN on dataset SimDial.

This script trains model VanillaLinearVRNN on SimDial data and report the
evaluation results.

"""

import collections
import json
import os
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from absl import app
from absl import flags
from absl import logging
from ml_collections import config_dict
import ml_collections.config_flags
import numpy as np
import tensorflow as tf
from uncertainty_baselines.datasets import datasets
import data_preprocessor as preprocessor  # local file import from experimental.language_structure.vrnn
import data_utils  # local file import from experimental.language_structure.vrnn
import linear_vrnn  # local file import from experimental.language_structure.vrnn
import model_config  # local file import from experimental.language_structure.vrnn
import psl_utils  # local file import from experimental.language_structure.vrnn
import train_lib  # local file import from experimental.language_structure.vrnn
import utils  # local file import from experimental.language_structure.vrnn


_DOMAIN_LABEL_NAME = preprocessor.DOMAIN_LABEL_NAME
_STATE_LABEL_NAME = preprocessor.STATE_LABEL_NAME
_DIAL_TURN_ID_NAME = preprocessor.DIAL_TURN_ID_NAME

_INPUT_ID_NAME = 'input_word_ids'
_INPUT_MASK_NAME = 'input_mask'

_LABEL_SAMPLE_MODE_KEY = 'mode'
_LABEL_RATIO_MODE = 'ratios'
_LABEL_SHOT_MODE = 'shots'

_TRAIN = 'train'
_TEST = 'test'
_SPLITS = [_TRAIN, _TEST]
_FEWSHOT_NAMESPACE = 'fewshot'

# The metric used for early stopping.
_PRIMARY_METRIC_KEY = f'{_TEST}/hidden_state_class_balanced_mixed_accuracy'
_PRIMARY_METRIC_SHOULD_DECREASE = False

FLAGS = flags.FLAGS

_CONFIG = ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)

_EXTRA_CONFIG = flags.DEFINE_string(
    'extra_config', '/tmp/vrnn/extra_config.json',
    'Json file storing configs to be overwritten.')
_OUTPUT_DIR = flags.DEFINE_string('output_dir', '/tmp/vrnn',
                                  'Output directory.')

_SEED = flags.DEFINE_integer('seed', 8, 'Random seed.')

# Accelerator flags.
_USE_GPU = flags.DEFINE_bool('use_gpu', False,
                             'Whether to run on GPU or otherwise TPU.')
_NUM_CORES = flags.DEFINE_integer('num_cores', 8,
                                  'Number of TPU cores or number of GPUs.')
_TPU = flags.DEFINE_string('tpu', None,
                           'Name of the TPU. Only used if use_gpu is False.')

_MetricMap = Dict[str, tf.keras.metrics.Metric]


def _label_count_map(labels) -> Dict[int, int]:
  unique_labels, counts = np.unique(labels, return_counts=True)
  return dict(zip(unique_labels, counts))


def _primary_metric_improved(metrics: _MetricMap, current_best: tf.Tensor,
                             min_delta: float) -> bool:
  """Returns whether the primary metric is improved."""
  if _PRIMARY_METRIC_SHOULD_DECREASE:
    return metrics[_PRIMARY_METRIC_KEY] + abs(min_delta) < current_best
  else:
    return metrics[_PRIMARY_METRIC_KEY] - abs(min_delta) > current_best


def _get_unmasked_dialog_turn_ids(labels: tf.Tensor, dialog_turn_ids: tf.Tensor,
                                  label_sample_map: Dict[int, float],
                                  label_sample_mode: str,
                                  seed: int) -> tf.Tensor:
  """Samples unmasked dialog turn ids from label_sample_map."""
  if label_sample_mode not in (_LABEL_RATIO_MODE, _LABEL_SHOT_MODE):
    raise NotImplementedError(
        'Only support label sample mode: %s, %s. Found %s.' %
        (_LABEL_RATIO_MODE, _LABEL_SHOT_MODE, label_sample_mode))

  rng = np.random.default_rng(seed=seed)
  labels = labels.numpy().flatten()
  dialog_turn_ids = dialog_turn_ids.numpy().flatten()

  if label_sample_mode == _LABEL_RATIO_MODE:
    # In ratio mode, we sample by the probabilities from ratios of examples to
    # be sampled per class.

    label_counts = _label_count_map(labels)
    label_sample_map = {
        label: ratio * float(label_counts[label])
        for label, ratio in label_sample_map.items()
    }

    # Compute the total number of samples and the sampling probabilities.
    total_samples = round(sum(label_sample_map.values()))
    label_sample_prob_map = {
        label: num_samples / (total_samples * float(label_counts[label]))
        for label, num_samples in label_sample_map.items()
    }

    sample_prob = np.zeros_like(labels, dtype=np.float32)
    for label, prob in label_sample_prob_map.items():
      sample_prob += prob * (labels == label).astype(np.float32)

    sample_dialog_turn_ids = rng.choice(
        dialog_turn_ids, total_samples, replace=False, p=sample_prob)
  else:
    # In shot mode, we sample separately for each class to ensure there are
    # exact number of examples to be sampled for each class.
    label_sample_map = {
        label: int(num_samples)
        for label, num_samples in label_sample_map.items()
    }

    # Summarize dialog turn ids for each class.
    label_dialog_turn_id_map = collections.defaultdict(list)
    for label, dialog_turn_id in zip(labels, dialog_turn_ids):
      label_dialog_turn_id_map[label].append(dialog_turn_id)

    # Sample given number of labeled dialog turns.
    sample_dialog_turn_ids = []
    for label in sorted(label_sample_map):
      if label_dialog_turn_id_map[label]:
        num_samples = min(
            len(label_dialog_turn_id_map[label]), label_sample_map[label])
        sample_dialog_turn_ids.append(
            rng.choice(
                label_dialog_turn_id_map[label], num_samples, replace=False))

    if sample_dialog_turn_ids:
      sample_dialog_turn_ids = np.concatenate(sample_dialog_turn_ids)

  return tf.constant(sample_dialog_turn_ids, dtype=tf.int32)


def _should_generate_labeled_dialog_turn_ids(with_label: bool,
                                             num_model_latent_states: int,
                                             num_latent_states: int,
                                             label_sampling_path: str) -> bool:
  """Determines whether to generate labeled dialog turn ids."""
  if not with_label:
    return False
  if num_model_latent_states != num_latent_states:
    raise ValueError(
        'Expected model num_states equal to the latent states of the'
        'dataset in semi-supervised mode, found {} vs {}'.format(
            num_model_latent_states, num_latent_states))
  return label_sampling_path is not None


def _generate_labeled_dialog_turn_ids(label_sampling_path: str,
                                      labels: tf.Tensor,
                                      dialog_turn_ids: tf.Tensor,
                                      seed: int) -> tf.Tensor:
  """Generates labeled dialog turn ids and saves them to `output_dir`."""
  with tf.io.gfile.GFile(label_sampling_path, 'r') as file:
    data = json.loads(file.read())

  label_sample_mode = data[_LABEL_SAMPLE_MODE_KEY]
  label_sample_map = {
      int(label): float(value)
      for label, value in data.items()
      if label != _LABEL_SAMPLE_MODE_KEY and value > 0
  }

  labeled_dialog_turn_ids = _get_unmasked_dialog_turn_ids(
      labels, dialog_turn_ids, label_sample_map, label_sample_mode, seed)

  return labeled_dialog_turn_ids


def _metric_namespace(split: str, in_domain: Optional[bool] = None) -> str:
  """Creates the metric namespace."""
  if in_domain is None:
    return split
  if split == _TRAIN:
    return split
  suffix = 'ind' if in_domain else 'ood'
  return f'{split}-{suffix}'


def _create_metrics_of_type(namespaces: Sequence[str],
                            metric_names: Sequence[str],
                            metric_type: Any) -> _MetricMap:
  """Creates a set of metrics of the same type under each of the namespace."""
  metrics = {}
  for namespace in namespaces:
    for metric_name in metric_names:
      metrics['{}/{}'.format(namespace, metric_name)] = metric_type()
  return metrics


# TODO(yquan): Create a class to manage metrics and re-organize namespaces.
def _create_metrics(
    namespaces: Sequence[str],
    psl_constraint_rule_names: Optional[Sequence[str]]) -> _MetricMap:
  """Creates metrics to be tracked in the training."""

  mean_type_metrics = [
      'total_loss',
      'rc_loss',
      'kl_loss',
      'bow_loss',
      'cls_loss',
      'elbo',
      'constraint_loss',
      'hidden_state_loss',
      'hidden_state_accuracy',
      'hidden_state_class_balanced_accuracy',
      'hidden_state_domain_loss',
      'hidden_state_domain_accuracy',
      'hidden_state_domain_class_balanced_accuracy',
      'hidden_state_class_balanced_mixed_accuracy',
      'adjusted_mutual_info',
      'cluster_purity',
      'unique_prediction_class_count',
  ]

  for rule_name in psl_constraint_rule_names:
    mean_type_metrics.append('constraint_loss_%s' % rule_name)

  accuracy_type_metrics = ['accuracy', 'class_balanced_accuracy']

  return {
      **_create_metrics_of_type(namespaces, mean_type_metrics,
                                tf.keras.metrics.Mean),
      **_create_metrics_of_type(namespaces, accuracy_type_metrics,
                                tf.keras.metrics.Accuracy),
  }




def _update_loss_metrics(metrics: _MetricMap, namespace: str,
                         losses: Sequence[Any],
                         psl_constraint_rule_names: Optional[Sequence[str]]):
  """Updates loss metrics."""

  (total_loss, rc_loss, kl_loss, bow_loss, classification_loss, constraint_loss,
   elbo, constraint_loss_per_rule) = losses
  metrics['{}/total_loss'.format(namespace)].update_state(total_loss)
  metrics['{}/elbo'.format(namespace)].update_state(elbo)
  metrics['{}/rc_loss'.format(namespace)].update_state(rc_loss)
  metrics['{}/kl_loss'.format(namespace)].update_state(kl_loss)
  metrics['{}/bow_loss'.format(namespace)].update_state(bow_loss)
  metrics['{}/cls_loss'.format(namespace)].update_state(classification_loss)
  metrics['{}/constraint_loss'.format(namespace)].update_state(constraint_loss)

  if constraint_loss_per_rule is not None:
    for rule_name, rule_loss in zip(psl_constraint_rule_names,
                                    constraint_loss_per_rule):
      metrics['{}/constraint_loss_{}'.format(namespace,
                                             rule_name)].update_state(rule_loss)


def _log_metric_results(metrics: _MetricMap, namespace: str):
  logging.info(
      '%s Accuracy: %.4f, Adjusted_Mutual_Information:'
      ' %.4f, Cluster_Purity: %.4f, Total Loss: %.4f, '
      'RC_Loss: %.4f, KL_Loss: %.4f, BOW_Loss: %.4f, CLS_loss: %.4f, '
      'PSL_Loss: %.4f, ELBO: %.4f, Hidden_State_Loss: %.4f, '
      'Hidden_State_Accuracy: %.4f, Hidden_State_Accuracy (balanced): %.4f, '
      'Hidden_State_Domain_Loss: %.4f, Hidden_State_Domain_Accuracy: %.4f, '
      'Hidden_State_Domain_Accuracy (balanced): %.4f', namespace,
      metrics['{}/accuracy'.format(namespace)].result(),
      metrics['{}/adjusted_mutual_info'.format(namespace)].result(),
      metrics['{}/cluster_purity'.format(namespace)].result(),
      metrics['{}/total_loss'.format(namespace)].result(),
      metrics['{}/rc_loss'.format(namespace)].result(),
      metrics['{}/kl_loss'.format(namespace)].result(),
      metrics['{}/bow_loss'.format(namespace)].result(),
      metrics['{}/cls_loss'.format(namespace)].result(),
      metrics['{}/constraint_loss'.format(namespace)].result(),
      metrics['{}/elbo'.format(namespace)].result(),
      metrics['{}/hidden_state_loss'.format(namespace)].result(),
      metrics['{}/hidden_state_accuracy'.format(namespace)].result(), metrics[
          '{}/hidden_state_class_balanced_accuracy'.format(namespace)].result(),
      metrics['{}/hidden_state_domain_loss'.format(namespace)].result(),
      metrics['{}/hidden_state_domain_accuracy'.format(namespace)].result(),
      metrics['{}/hidden_state_domain_class_balanced_accuracy'.format(
          namespace)].result())


def _load_data_from_files(config: config_dict.ConfigDict):
  """Update config by data read from files."""

  def _load_embedding_data_from_files(
      embedding_config: model_config.EmbeddingConfig):
    with tf.io.gfile.GFile(embedding_config.vocab_file_path, 'r') as f:
      embedding_config.vocab_size = len(f.read()[:-1].split('\n'))

    if (embedding_config.embedding_type == model_config.BERT_EMBED and
        embedding_config.bert_config_file):
      with tf.io.gfile.GFile(embedding_config.bert_config_file) as config_file:
        embedding_config.bert_config = json.load(config_file)

  _load_embedding_data_from_files(config.model.vae_cell.encoder_embedding)
  _load_embedding_data_from_files(config.model.vae_cell.decoder_embedding)

  if config.psl_config_file:
    with tf.io.gfile.GFile(config.psl_config_file, 'r') as file:
      config.psl = json.loads(file.read())
  if config.psl_word_weights_file and 'word_weights' in config.psl:
    with tf.io.gfile.GFile(config.psl_word_weights_file, 'rb') as file:
      config.psl['word_weights'] = np.load(file)


def _overwrite_configs_by_flags(config: config_dict.ConfigDict) -> str:
  """Overwrites configs by flag arguments."""
  message = ''
  if _EXTRA_CONFIG.value:
    with tf.io.gfile.GFile(_EXTRA_CONFIG.value, 'r') as f:
      params = json.load(f)
    message = os.path.basename(_EXTRA_CONFIG.value) + '\n'
    for key, value in params.items():
      setattr(config, key, value)
      message += f'{key}={value}\n'
  return message


def _save_model_results(outputs: Sequence[tf.Tensor], output_dir: str,
                        split: str):
  """Saves the model predictions, labels and latent state representations."""
  latent_state, label, prediction, domain_label = outputs

  for file_name, data in zip(
      ['label.npy', 'prediction.npy', 'latent_state.npy', 'domain_label.npy'],
      [label, prediction, latent_state, domain_label]):
    with tf.io.gfile.GFile(
        os.path.join(output_dir, '{}-{}'.format(split, file_name)), 'wb') as f:
      np.save(f, data.numpy())


def _update_hidden_state_model_metrics(
    metrics: _MetricMap, namespaces: Sequence[str],
    eval_results: Tuple[Sequence[Sequence[float]], Sequence[Sequence[float]]]):
  """Updates hidden state model specific metrics."""
  hidden_state_model_metrics = [
      'hidden_state_loss',
      'hidden_state_accuracy',
      'hidden_state_class_balanced_accuracy',
      'hidden_state_domain_loss',
      'hidden_state_domain_accuracy',
      'hidden_state_domain_class_balanced_accuracy',
  ]

  for namespace, state_eval_result, domain_eval_result in zip(
      namespaces, eval_results[0], eval_results[1]):
    for key, value in zip(hidden_state_model_metrics,
                          state_eval_result + domain_eval_result):
      metrics['{}/{}'.format(namespace, key)].update_state(value)
    metrics['{}/hidden_state_class_balanced_mixed_accuracy'.format(
        namespace)].update_state(
            (state_eval_result[2] + domain_eval_result[2]) / 2)


def _update_model_prediction_metrics(metrics: _MetricMap, namespace: str,
                                     label_id: tf.Tensor,
                                     prediction: tf.Tensor):
  """Updates metrics related to model prediction quality."""
  # Updates clustering related metrics.
  metrics['{}/adjusted_mutual_info'.format(namespace)].update_state(
      utils.adjusted_mutual_info(label_id, prediction))
  metrics['{}/cluster_purity'.format(namespace)].update_state(
      utils.cluster_purity(label_id, prediction))
  prediction_classes, _ = tf.unique(tf.reshape(prediction, shape=[-1]))
  metrics['{}/unique_prediction_class_count'.format(namespace)].update_state(
      tf.size(prediction_classes))
  # Updates accuracies.
  metrics['{}/accuracy'.format(namespace)].update_state(label_id, prediction,
                                                        tf.sign(label_id))
  class_balanced_weight = utils.create_rebalanced_sample_weights(label_id)
  metrics['{}/class_balanced_accuracy'.format(namespace)].update_state(
      label_id, prediction, class_balanced_weight)




def _transform_hidden_representation(
    inputs: tf.Tensor, labels: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Flatten the hidden representation and labels and filtering out paddings."""
  inputs = tf.reshape(inputs, [-1, inputs.shape[-1]])
  labels = tf.reshape(labels, [-1])

  padding_mask = labels > 0
  return tf.boolean_mask(inputs,
                         padding_mask), tf.boolean_mask(labels, padding_mask)


def _evaluate_hidden_state_model(input_size: int, num_classes: int,
                                 train_x: tf.Tensor, train_y: tf.Tensor,
                                 test_x: tf.Tensor, test_y: tf.Tensor,
                                 test_masks: Sequence[tf.Tensor],
                                 train_epochs: int, learning_rate: float):
  """Evaluates the hidden state representation."""
  train_x, train_y = _transform_hidden_representation(train_x, train_y)

  model = train_lib.build_hidden_state_model(input_size, num_classes,
                                             learning_rate)
  model.fit(train_x, train_y, epochs=train_epochs, verbose=0)

  results = []
  results.append(
      model.evaluate(
          train_x,
          train_y,
          sample_weight=utils.create_rebalanced_sample_weights(train_y),
          verbose=0))

  for mask in test_masks:
    test_x_transformed, test_y_transformed = _transform_hidden_representation(
        tf.boolean_mask(test_x, mask), tf.boolean_mask(test_y, mask))
    results.append(
        model.evaluate(
            test_x_transformed,
            test_y_transformed,
            sample_weight=utils.create_rebalanced_sample_weights(
                test_y_transformed),
            verbose=0))
  return results


def _load_class_map(file_path: str) -> Dict[int, str]:
  """Loads class {id, name} mapping from the given file."""
  with tf.io.gfile.GFile(file_path) as class_map_file:
    class_map = json.load(class_map_file)
    class_map = {int(key): value for key, value in class_map.items()}
  return class_map


def _create_fewshot_dataset_and_sample_weights(
    feautres: tf.Tensor, labels: tf.Tensor,
    repr_fn: Any) -> Tuple[tf.data.Dataset, tf.Tensor]:
  """Creates dataset for few-shot evaluation and the rebalanced sample weights."""
  _, label = repr_fn(feautres, labels)
  sample_weights = utils.create_rebalanced_sample_weights(label)
  dataset = tf.data.Dataset.from_tensor_slices((feautres, labels))
  dataset = dataset.batch(labels.shape[0]).repeat()
  return dataset, sample_weights


def _json_dump(config: config_dict.ConfigDict, filename: str):
  """Dumps the config into a json file."""
  with tf.io.gfile.GFile(filename, 'w') as f:
    json.dump(config.to_dict(), f)


def _build_data_processor(
    config: config_dict.ConfigDict,
    labeled_dialog_turn_ids: Optional[tf.Tensor] = None,
    in_domains: Optional[tf.Tensor] = None) -> preprocessor.DataPreprocessor:
  """Creates data processor for the dataset."""

  def _get_utterance_feature_fn(embedding_type: str):
    """Returns the utterance feature function by `embedding_type`."""
    if embedding_type == model_config.GLOVE_EMBED:
      return preprocessor.create_utterance_features

    bert_preprocess_model = utils.BertPreprocessor(
        config.bert_embedding_preprocess_tfhub_url,
        config.model.vae_cell.max_seq_length)
    return preprocessor.create_bert_utterance_features_fn(bert_preprocess_model)

  encoder_embedding = config.model.vae_cell.encoder_embedding
  decoder_embedding = config.model.vae_cell.decoder_embedding

  model_config.verify_embedding_configs(encoder_embedding, decoder_embedding,
                                        config.shared_embedding)

  encoder_process_fn = _get_utterance_feature_fn(
      encoder_embedding.embedding_type)
  decoder_process_fn = _get_utterance_feature_fn(
      decoder_embedding.embedding_type)

  return preprocessor.DataPreprocessor(encoder_process_fn, decoder_process_fn,
                                       config.model.num_states,
                                       labeled_dialog_turn_ids, in_domains)


def run_experiment(config: config_dict.ConfigDict, output_dir: str):
  """Runs training/evaluation experiment."""
  seed = config.get('seed', 0)


  _load_data_from_files(config)
  message = _overwrite_configs_by_flags(config)

  logging.info('Config: %s\n Overwritten: %s', config, message)

  tf.io.gfile.makedirs(output_dir)
  logging.info('Model checkpoint will be saved at %s', output_dir)
  tf.random.set_seed(seed)

  if config.model_base_dir:
    dir_name = os.path.basename(output_dir)
    model_dir = os.path.join(config.model_base_dir, dir_name)
    logging.info('Model outputs will be saved at %s', model_dir)
    tf.io.gfile.makedirs(model_dir)
    _json_dump(config, os.path.join(model_dir, 'config.json'))
    _json_dump(config.model, os.path.join(model_dir, 'model_config.json'))
  else:
    model_dir = None

  if _USE_GPU.value:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
    drop_remainder = False
  else:
    logging.info('Use TPU at %s',
                 _TPU.value if _TPU.value is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=_TPU.value)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)
    drop_remainder = True

  with_label = config.classification_loss_weight > 0

  # Create dataset builders
  train_dataset_builder = datasets.get(
      config.dataset,
      split=_TRAIN,
      data_dir=config.dataset_dir,
      shuffle_buffer_size=config.train_batch_size * 10,
      seed=seed,
      add_dialog_turn_id=with_label,
      drop_remainder=drop_remainder)
  test_dataset_builder = datasets.get(
      config.dataset,
      split=_TEST,
      data_dir=config.dataset_dir,
      shuffle_buffer_size=config.eval_batch_size * 10,
      drop_remainder=drop_remainder)

  test_datasets_builders = {_TEST: test_dataset_builder}

  inference_train_dataset_builder = datasets.get(
      config.dataset,
      split=_TRAIN,
      data_dir=config.dataset_dir,
      shuffle_buffer_size=config.inference_batch_size * 10,
      seed=config.inference_seed,
      is_training=False)
  inference_test_dataset_builder = datasets.get(
      config.dataset,
      split=_TEST,
      data_dir=config.dataset_dir,
      shuffle_buffer_size=config.inference_batch_size * 10,
      seed=config.inference_seed)

  # Choose labeled dialog turns.
  num_latent_states = data_utils.get_dataset_num_latent_states(config.dataset)
  if _should_generate_labeled_dialog_turn_ids(with_label,
                                              config.model.num_states,
                                              num_latent_states,
                                              config.label_sampling_path):
    inputs = preprocessor.get_full_dataset_outputs(train_dataset_builder)
    labeled_dialog_turn_ids = _generate_labeled_dialog_turn_ids(
        config.label_sampling_path, inputs[_STATE_LABEL_NAME],
        inputs[_DIAL_TURN_ID_NAME], seed)
    if model_dir:
      with tf.io.gfile.GFile(
          os.path.join(model_dir, 'labeled_dialog_turn_ids.txt'), 'w') as f:
        f.write('\n'.join(
            str(id) for id in labeled_dialog_turn_ids.numpy().tolist()))
  else:
    labeled_dialog_turn_ids = None

  if config.domain_adaptation:
    inputs = preprocessor.get_full_dataset_outputs(train_dataset_builder)
    # Notice domain label id 0 is also treated as in-domain: ood should have
    # a different id from it.
    in_domains, _ = tf.unique(tf.reshape(inputs[_DOMAIN_LABEL_NAME], [-1]))
    metric_namespaces = [
        _metric_namespace(_TRAIN),
        _metric_namespace(_TEST, True),
        _metric_namespace(_TEST, False)
    ]
    fewshot_metric_namespaces = [
        _metric_namespace(_FEWSHOT_NAMESPACE, True),
        _metric_namespace(_FEWSHOT_NAMESPACE, False)
    ]
  else:
    in_domains = None
    metric_namespaces = [_metric_namespace(split) for split in _SPLITS]
    fewshot_metric_namespaces = [_metric_namespace(_FEWSHOT_NAMESPACE)]

  data_preprocessor = _build_data_processor(config, labeled_dialog_turn_ids,
                                            in_domains)
  preprocess_fn = data_preprocessor.create_feature_and_label

  # Load PSL configs
  psl_learning = config.psl_constraint_learning_weight > 0
  psl_inference = config.psl_constraint_inference_weight > 0
  if psl_learning or psl_inference:
    with tf.io.gfile.GFile(
        config.model.vae_cell.encoder_embedding.vocab_file_path, 'r') as f:
      vocab = f.read()[:-1].split('\n')
    preprocess_fn = psl_utils.psl_feature_mixin(preprocess_fn, config.dataset,
                                                config.psl, vocab)

  # Load datasets
  # TODO(yquan): invesigate why distributed training fails in *fish TPU
  # Failure example: https://xm2a.corp.google.com/experiments/33275459
  distributed_training = False
  train_dataset = preprocessor.create_dataset(train_dataset_builder,
                                              config.train_batch_size,
                                              preprocess_fn, strategy,
                                              distributed_training)
  steps_per_epoch = train_dataset_builder.num_examples // config.train_batch_size

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in test_datasets_builders.items():
    steps_per_eval[dataset_name] = (
        dataset_builder.num_examples // config.eval_batch_size)
    test_datasets[dataset_name] = preprocessor.create_dataset(
        dataset_builder, config.eval_batch_size, preprocess_fn, strategy,
        distributed_training)

  distributed_inference = False
  inference_train_dataset = preprocessor.create_dataset(
      inference_train_dataset_builder, config.inference_batch_size,
      preprocess_fn, strategy, distributed_inference)
  num_inference_train_steps = (
      inference_train_dataset_builder.num_examples //
      config.inference_batch_size)
  inference_test_dataset = preprocessor.create_dataset(
      inference_test_dataset_builder, config.inference_batch_size,
      preprocess_fn, strategy, distributed_inference)
  num_inference_test_steps = (
      inference_test_dataset_builder.num_examples //
      config.inference_batch_size)

  # Initialize word weights.
  word_weights = np.ones((config.model.vae_cell.decoder_embedding.vocab_size),
                         dtype=np.float32)
  if config.word_weights_path:
    w = config.word_weights_file_weight
    if w > 1 or w < 0:
      raise ValueError(
          'Expected word_weights_file_weight between 0 and 1, found {}'.format(
              w))
    with tf.io.gfile.GFile(config.word_weights_path, 'rb') as word_weights_file:
      word_weights_from_file = np.load(word_weights_file)
    word_weights = w * word_weights_from_file + (1 - w) * word_weights

  _json_dump(config.model, os.path.join(output_dir, 'model_config.json'))

  with strategy.scope():
    model = linear_vrnn.VanillaLinearVRNN(config.model)

    optimizer = tf.keras.optimizers.Adam(
        config.base_learning_rate, beta_1=1.0 - config.one_minus_momentum)

    metrics = _create_metrics(metric_namespaces,
                              config.psl_constraint_rule_names)

    if psl_learning or psl_inference:
      psl_model = psl_utils.get_psl_model(
          config.dataset,
          config.psl_constraint_rule_names,
          config.psl_constraint_rule_weights,
          config=config.psl)
    else:
      psl_model = None

    if psl_inference:
      psl_optimizer = tf.keras.optimizers.Adam(
          config.base_learning_rate, beta_1=1.0 - config.one_minus_momentum)
    else:
      psl_optimizer = None

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # If we found a valid checkpoint in output_dir, the job is an auto-retry one
    # and we should continue training; otherwise we try first initialize from
    # config.init_checkpoint or config.init_dir
    init_checkpoint = tf.train.latest_checkpoint(output_dir)
    if not init_checkpoint:
      if config.init_checkpoint:
        init_checkpoint = config.init_checkpoint
      elif config.init_dir:
        init_checkpoint = tf.train.latest_checkpoint(config.init_dir)
    initial_epoch = 0
    if init_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(init_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
      logging.info('Loaded checkpoint %s. Initialize from epoch %s',
                   init_checkpoint, initial_epoch)
    else:
      model.vae_cell.init_bert_embedding_layers(config.model.vae_cell)

  def train_step(batch_size: int, config: config_dict.ConfigDict):

    @tf.function
    def _train_step(inputs: Sequence[tf.Tensor]):
      """Training step function."""

      (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
       label_id, label_mask, initial_state, initial_sample, _) = inputs[:9]
      if psl_learning:
        psl_inputs = inputs[-1]
        # Explicitly specify the batch size as PSL model now requires known
        # batch size.
        psl_inputs = tf.ensure_shape(
            psl_inputs, (batch_size, psl_inputs.shape[1], psl_inputs.shape[2]))
      else:
        psl_inputs = None

      model_inputs = [
          encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
          initial_state, initial_sample
      ]
      if with_label:
        model_inputs.extend([label_id, label_mask])

      with tf.GradientTape() as tape:
        # Set learning phase to enable dropout etc during training.
        model_outputs = model(model_inputs, training=True)

        losses = linear_vrnn.compute_loss(
            decoder_input_1[_INPUT_ID_NAME][:, :, 1:],
            decoder_input_2[_INPUT_ID_NAME][:, :, 1:],
            decoder_input_1[_INPUT_MASK_NAME][:, :, 1:],
            decoder_input_2[_INPUT_MASK_NAME][:, :, 1:],
            model_outputs,
            latent_label_id=label_id,
            latent_label_mask=label_mask,
            word_weights=word_weights,
            with_bpr=config.with_bpr,
            kl_loss_weight=config.kl_loss_weight,
            with_bow=config.with_bow,
            bow_loss_weight=config.bow_loss_weight,
            num_latent_states=num_latent_states,
            classification_loss_weight=config.classification_loss_weight,
            psl_constraint_model=psl_model,
            psl_inputs=psl_inputs,
            psl_constraint_loss_weight=config.psl_constraint_learning_weight)

      total_loss = losses[0]
      grads = tape.gradient(total_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      _update_loss_metrics(metrics, _metric_namespace(_TRAIN), losses,
                           config.psl_constraint_rule_names)

    return _train_step

  def test_step(split: str, batch_size: int, config: config_dict.ConfigDict):

    @tf.function
    def _test_step(inputs: Sequence[tf.Tensor]):
      """Evaluation step function."""

      (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
       label_id, label_mask, initial_state, initial_sample,
       domains) = inputs[:9]

      _, ind_mask = domains

      if psl_inference:
        psl_inputs = inputs[-1]
        # Explicitly specify the batch size as PSL model now requires known
        # batch size.
        psl_inputs = tf.ensure_shape(
            psl_inputs, (batch_size, psl_inputs.shape[1], psl_inputs.shape[2]))
      else:
        psl_inputs = None

      # In evaluation, don't provide label as a guidance.
      model_inputs = [
          encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
          initial_state, initial_sample
      ]
      model_outputs = model(model_inputs, training=False)

      if ind_mask is not None:
        ood_mask = 1 - ind_mask
        label_masks = {
            _metric_namespace(split, True): ind_mask * label_mask,
            _metric_namespace(split, False): ood_mask * label_mask
        }
      else:
        label_masks = {_metric_namespace(split): label_mask}

      for namespace, label_mask in label_masks.items():
        losses = linear_vrnn.compute_loss(
            decoder_input_1[_INPUT_ID_NAME][:, :, 1:],
            decoder_input_2[_INPUT_ID_NAME][:, :, 1:],
            decoder_input_1[_INPUT_MASK_NAME][:, :, 1:],
            decoder_input_2[_INPUT_MASK_NAME][:, :, 1:],
            model_outputs,
            latent_label_id=label_id,
            latent_label_mask=label_mask,
            word_weights=word_weights,
            with_bpr=config.with_bpr,
            kl_loss_weight=config.kl_loss_weight,
            with_bow=config.with_bow,
            bow_loss_weight=config.bow_loss_weight,
            num_latent_states=num_latent_states,
            classification_loss_weight=config.classification_loss_weight,
            psl_constraint_model=psl_model,
            psl_inputs=psl_inputs,
            psl_constraint_loss_weight=config.psl_constraint_inference_weight)

        _update_loss_metrics(metrics, namespace, losses,
                             config.psl_constraint_rule_names)

    return _test_step

  def inference_step(psl_inference: bool, batch_size: int,
                     config: config_dict.ConfigDict):

    @tf.function
    def _inference_step(inputs: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
      (encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
       label, _, initial_state, initial_sample, domains) = inputs[:9]
      model_inputs = [
          encoder_input_1, encoder_input_2, decoder_input_1, decoder_input_2,
          initial_state, initial_sample
      ]
      model_outputs = model(model_inputs, training=False)

      if psl_inference:
        psl_inputs = inputs[-1]
        # Explicitly specify the batch size as PSL model now requires known
        # batch size.
        psl_inputs = tf.ensure_shape(
            psl_inputs, (batch_size, psl_inputs.shape[1], psl_inputs.shape[2]))
        logits = psl_utils.update_logits(model, psl_optimizer, model_inputs,
                                         linear_vrnn.get_logits, psl_model,
                                         psl_inputs,
                                         config.psl_constraint_inference_steps,
                                         config.psl_constraint_inference_weight)
      else:
        logits = linear_vrnn.get_logits(model_outputs)

      prediction = linear_vrnn.get_prediction(logits)
      latent_state = model_outputs[0]
      domain_label, ind_mask = domains
      if ind_mask is None:
        ind_mask = tf.ones_like(domain_label)

      return latent_state, label, prediction, domain_label, ind_mask

    return _inference_step


  summary_writer = tf.summary.create_file_writer(
      os.path.join(output_dir, 'summaries'))
  if model_dir:
    best_summary_writer = tf.summary.create_file_writer(
        os.path.join(model_dir, 'summaries'))
  else:
    best_summary_writer = None

  run_train_steps = train_lib.create_run_steps_fn(
      train_step(config.train_batch_size, config),
      strategy,
      distributed=distributed_training)

  run_test_steps_map = {}
  for split in test_datasets:
    run_test_steps_map[split] = train_lib.create_run_steps_fn(
        test_step(split, config.eval_batch_size, config),
        strategy,
        distributed=distributed_training)

  run_inference_steps = train_lib.create_run_steps_fn(
      inference_step(psl_inference, config.inference_batch_size, config),
      strategy,
      distributed=distributed_inference,
      output_dtypes=[tf.float32, tf.int32, tf.int32, tf.int32, tf.int32],
  )

  fixed_train_epoch = config.patience < 0
  primary_metric = tf.constant(0.)
  out_of_patience = 0
  train_model_outputs = None
  test_model_outputs = None
  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, config.train_epochs):
    if not fixed_train_epoch and out_of_patience > config.patience:
      logging.info(
          'Found primary metric %s keeping being worse than the '
          'current best %.4f for %s evaluation cycles, early stop '
          'at epoch %s', _PRIMARY_METRIC_KEY, primary_metric, out_of_patience,
          epoch)
      break
    logging.info('Starting to run epoch: %s', epoch)
    run_train_steps(train_iterator, tf.cast(steps_per_epoch, tf.int32))
    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * config.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, config.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    if (epoch + 1) % config.evaluation_interval == 0:
      for dataset_name, test_dataset in test_datasets.items():
        test_iterator = iter(test_dataset)
        logging.info('Testing on dataset %s', dataset_name)
        logging.info('Starting to run eval at epoch: %s', epoch)
        run_test_steps_map[dataset_name](test_iterator,
                                         tf.cast(steps_per_eval[dataset_name],
                                                 tf.int32))
        logging.info('Done with testing on %s', dataset_name)

      (train_hidden_state, train_label, train_prediction, train_domain_label,
       _) = run_inference_steps(
           iter(inference_train_dataset), num_inference_train_steps)
      (test_hidden_state, test_label, test_prediction, test_domain_label,
       ind_mask) = run_inference_steps(
           iter(inference_test_dataset), num_inference_test_steps)

      if config.domain_adaptation:
        test_example_masks = [ind_mask, 1 - ind_mask]
      else:
        test_example_masks = [ind_mask]


      logging.info('Evaluating hidden representation learning.')

      input_size = config.model.vae_cell.encoder_projection_sizes[-1]
      results = _evaluate_hidden_state_model(
          input_size, config.model.num_states, train_hidden_state, train_label,
          test_hidden_state, test_label, test_example_masks,
          config.hidden_state_model_train_epochs,
          config.hidden_state_model_learning_rate)
      domain_results = _evaluate_hidden_state_model(
          input_size, data_utils.get_dataset_num_domains(config.dataset),
          train_hidden_state, train_domain_label, test_hidden_state,
          test_domain_label, test_example_masks,
          config.hidden_state_model_train_epochs,
          config.hidden_state_model_learning_rate)

      _update_hidden_state_model_metrics(metrics, metric_namespaces,
                                         (results, domain_results))

      _update_model_prediction_metrics(metrics, _metric_namespace(_TRAIN),
                                       train_label, train_prediction)
      for namespace, mask in zip(metric_namespaces[1:], test_example_masks):
        _update_model_prediction_metrics(metrics, namespace,
                                         tf.boolean_mask(test_label, mask),
                                         tf.boolean_mask(test_prediction, mask))

      for namespace in metric_namespaces:
        _log_metric_results(metrics, namespace)

      total_results = {
          name: metric.result() for name, metric in metrics.items()
      }
      with summary_writer.as_default():
        for name, result in total_results.items():
          tf.summary.scalar(name, result, step=epoch)

      train_model_outputs = [
          train_hidden_state, train_label, train_prediction, train_domain_label
      ]
      test_model_outputs = [
          test_hidden_state, test_label, test_prediction, test_domain_label
      ]
      if not fixed_train_epoch:
        if _primary_metric_improved(total_results, primary_metric,
                                    config.min_delta):
          primary_metric = total_results[_PRIMARY_METRIC_KEY]
          out_of_patience = 0
          if best_summary_writer:
            with best_summary_writer.as_default():
              for name, result in total_results.items():
                tf.summary.scalar(name, result, step=epoch)
          if model_dir:
            checkpoint_name = checkpoint.save(
                os.path.join(model_dir, 'checkpoint'))
            logging.info('Saved checkpoint to %s', checkpoint_name)
            _save_model_results(train_model_outputs, model_dir, _TRAIN)
            _save_model_results(test_model_outputs, model_dir, _TEST)
        else:
          out_of_patience += 1


    for metric in metrics.values():
      metric.reset_states()

    if (config.checkpoint_interval > 0 and
        (epoch + 1) % config.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(os.path.join(output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)
      if fixed_train_epoch:
        if train_model_outputs:
          _save_model_results(train_model_outputs, output_dir, _TRAIN)
        if test_model_outputs:
          _save_model_results(test_model_outputs, output_dir, _TEST)


  return False


if __name__ == '__main__':

  def _main(argv):
    """Main entry function."""
    del argv  # unused
    num_restarts = 0
    config = _CONFIG.value
    output_dir = _OUTPUT_DIR.value
    keep_running = True
    while keep_running:
      try:
        keep_running = run_experiment(config, output_dir)
      except tf.errors.UnavailableError as err:
        num_restarts += 1
        logging.warn(
            'Error encountered during experiment: %s. Will now try to recover.',
            err,
            exc_info=True)

  app.run(_main)
