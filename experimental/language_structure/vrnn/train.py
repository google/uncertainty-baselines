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
import bert_utils  # local file import from baselines.clinc_intent
from uncertainty_baselines.datasets import datasets
import data_preprocessor as preprocessor  # local file import from experimental.language_structure.vrnn
import data_utils  # local file import from experimental.language_structure.vrnn
import linear_vrnn  # local file import from experimental.language_structure.vrnn
import psl_utils  # local file import from experimental.language_structure.vrnn
import train_lib  # local file import from experimental.language_structure.vrnn
import utils  # local file import from experimental.language_structure.vrnn


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

# The metric used for early stopping.
_PRIMARY_METRIC_KEY = f'{_TEST}/hidden_state_class_balanced_mixed_accuracy'
_PRIMARY_METRIC_SHOULD_DECREASE = False

FLAGS = flags.FLAGS

_CONFIG = ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)

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

  labels = labels.numpy().flatten()
  if label_sample_mode == _LABEL_RATIO_MODE:
    # Compute number of labeled examples to be sampled in each class.
    label_counts = _label_count_map(labels)
    label_sample_map = {
        label: round(label_sample_map[label] * label_counts.get(label, 0))
        for label in label_sample_map
    }
  else:
    label_sample_map = {
        label: int(num_samples)
        for label, num_samples in label_sample_map.items()
    }

  # Summarize dialog turn ids for each class.
  label_dialog_turn_id_map = collections.defaultdict(list)
  for label, dialog_turn_id in zip(labels, dialog_turn_ids.numpy().flatten()):
    label_dialog_turn_id_map[label].append(dialog_turn_id)

  # Sample given number of labeled dialog turns.
  dialog_turn_ids = []
  rng = np.random.default_rng(seed=seed)
  for label in sorted(label_sample_map):
    if label_dialog_turn_id_map[label]:
      num_samples = min(
          len(label_dialog_turn_id_map[label]), label_sample_map[label])
      dialog_turn_ids.append(
          rng.choice(
              label_dialog_turn_id_map[label], num_samples, replace=False))

  if dialog_turn_ids:
    dialog_turn_ids = np.concatenate(dialog_turn_ids)
  return tf.constant(dialog_turn_ids, dtype=tf.int32)


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


# TODO(yquan): Create a class to manage metrics and re-organize namespaces.
def _create_metrics(
    splits: Sequence[str], few_shots: Sequence[int],
    few_shots_l2_weights: Sequence[float],
    psl_constraint_rule_names: Optional[Sequence[str]]) -> _MetricMap:
  """Creates metrics to be tracked in the training."""

  def _create_metrics_of_type(
      metric_names: Sequence[str],
      metric_type: Any,
      namespaces: Optional[Sequence[str]] = splits) -> _MetricMap:
    metrics = {}
    for namespace in namespaces:
      for metric_name in metric_names:
        metrics['{}/{}'.format(namespace, metric_name)] = metric_type()
    return metrics

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
  ]

  for rule_name in psl_constraint_rule_names:
    mean_type_metrics.append('constraint_loss_%s' % rule_name)

  accuracy_type_metrics = [
      'accuracy', 'masked_accuracy', 'class_balanced_accuracy'
  ]


  return {
      **_create_metrics_of_type(mean_type_metrics, tf.keras.metrics.Mean),
      **_create_metrics_of_type(accuracy_type_metrics,
                                tf.keras.metrics.Accuracy),
  }


def _update_metrics(metrics: _MetricMap, split: str, logits: tf.Tensor,
                    losses: Sequence[Any], label_id: tf.Tensor,
                    label_mask: tf.Tensor,
                    psl_constraint_rule_names: Optional[Sequence[str]]):
  """Updates metrics by model outputs, losses and labels."""
  prediction = linear_vrnn.get_prediction(logits)
  metrics['{}/masked_accuracy'.format(split)].update_state(
      label_id, prediction, label_mask)
  metrics['{}/accuracy'.format(split)].update_state(label_id, prediction,
                                                    tf.sign(label_id))

  (total_loss, rc_loss, kl_loss, bow_loss, classification_loss, constraint_loss,
   elbo, constraint_loss_per_rule) = losses
  metrics['{}/total_loss'.format(split)].update_state(total_loss)
  metrics['{}/elbo'.format(split)].update_state(elbo)
  metrics['{}/rc_loss'.format(split)].update_state(rc_loss)
  metrics['{}/kl_loss'.format(split)].update_state(kl_loss)
  metrics['{}/bow_loss'.format(split)].update_state(bow_loss)
  metrics['{}/cls_loss'.format(split)].update_state(classification_loss)
  metrics['{}/constraint_loss'.format(split)].update_state(constraint_loss)

  if constraint_loss_per_rule is not None:
    for rule_name, rule_loss in zip(psl_constraint_rule_names,
                                    constraint_loss_per_rule):
      metrics['{}/constraint_loss_{}'.format(split,
                                             rule_name)].update_state(rule_loss)


def _log_metric_results(metrics: _MetricMap, split: str):
  logging.info(
      '%s Accuracy (masked): %.4f, Accuracy: %.4f, Adjusted_Mutual_Information:'
      ' %.4f, Cluster_Purity: %.4f, Total Loss: %.4f, '
      'RC_Loss: %.4f, KL_Loss: %.4f, BOW_Loss: %.4f, CLS_loss: %.4f, '
      'PSL_Loss: %.4f, ELBO: %.4f, Hidden_State_Loss: %.4f, '
      'Hidden_State_Accuracy: %.4f, Hidden_State_Accuracy (balanced): %.4f, '
      'Hidden_State_Domain_Loss: %.4f, Hidden_State_Domain_Accuracy: %.4f, '
      'Hidden_State_Domain_Accuracy (balanced): %.4f', split,
      metrics['{}/masked_accuracy'.format(split)].result(),
      metrics['{}/accuracy'.format(split)].result(),
      metrics['{}/adjusted_mutual_info'.format(split)].result(),
      metrics['{}/cluster_purity'.format(split)].result(),
      metrics['{}/total_loss'.format(split)].result(),
      metrics['{}/rc_loss'.format(split)].result(),
      metrics['{}/kl_loss'.format(split)].result(),
      metrics['{}/bow_loss'.format(split)].result(),
      metrics['{}/cls_loss'.format(split)].result(),
      metrics['{}/constraint_loss'.format(split)].result(),
      metrics['{}/elbo'.format(split)].result(),
      metrics['{}/hidden_state_loss'.format(split)].result(),
      metrics['{}/hidden_state_accuracy'.format(split)].result(),
      metrics['{}/hidden_state_class_balanced_accuracy'.format(split)].result(),
      metrics['{}/hidden_state_domain_loss'.format(split)].result(),
      metrics['{}/hidden_state_domain_accuracy'.format(split)].result(),
      metrics['{}/hidden_state_domain_class_balanced_accuracy'.format(
          split)].result())


def _load_data_from_files(config: config_dict.ConfigDict):
  """Update config by data read from files."""
  with tf.io.gfile.GFile(config.vocab_file_path, 'r') as f:
    vocab_size = len(f.read()[:-1].split('\n'))
  config.model.vocab_size = config.model.vae_cell.vocab_size = vocab_size

  if config.model.vae_cell.shared_bert_embedding:
    with tf.io.gfile.GFile(os.path.join(config.bert_dir,
                                        'bert_config.json')) as config_file:
      config.model.vae_cell.shared_bert_embedding_config = json.load(
          config_file)

  if config.psl_config_file:
    with tf.io.gfile.GFile(config.psl_config_file, 'r') as file:
      config.psl = json.loads(file.read())


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
    metrics: _MetricMap, splits: Sequence[str],
    evaluation_results: Sequence[Sequence[float]]):
  """Updates hidden state model specific metrics."""
  hidden_state_model_metrics = [
      'hidden_state_loss',
      'hidden_state_accuracy',
      'hidden_state_class_balanced_accuracy',
      'hidden_state_domain_loss',
      'hidden_state_domain_accuracy',
      'hidden_state_domain_class_balanced_accuracy',
  ]
  for split, split_evaluation_results in zip(splits, evaluation_results):
    for key, value in zip(hidden_state_model_metrics, split_evaluation_results):
      metrics['{}/{}'.format(split, key)].update_state(value)
    metrics['{}/hidden_state_class_balanced_mixed_accuracy'.format(
        split)].update_state(
            (split_evaluation_results[2] + split_evaluation_results[5]) / 2)


def _update_clustering_metrics(metrics: _MetricMap, split: str,
                               label_id: tf.Tensor, prediction: tf.Tensor):
  """Updates clustering related metrics."""
  metrics['{}/adjusted_mutual_info'.format(split)].update_state(
      utils.adjusted_mutual_info(label_id, prediction))
  metrics['{}/cluster_purity'.format(split)].update_state(
      utils.cluster_purity(label_id, prediction))




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
                                 train_epochs: int, learning_rate: float):
  """Evaluates the hidden state representation."""
  train_x, train_y = _transform_hidden_representation(train_x, train_y)
  test_x, test_y = _transform_hidden_representation(test_x, test_y)

  model = train_lib.build_hidden_state_model(input_size, num_classes,
                                             learning_rate)
  model.fit(train_x, train_y, epochs=train_epochs, verbose=0)
  train_results = model.evaluate(
      train_x,
      train_y,
      sample_weight=utils.create_rebalanced_sample_weights(train_y),
      verbose=0)

  test_results = model.evaluate(
      test_x,
      test_y,
      sample_weight=utils.create_rebalanced_sample_weights(test_y),
      verbose=0)
  return train_results, test_results


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


def run_experiment(config: config_dict.ConfigDict, output_dir: str):
  """Runs training/evaluation experiment."""
  seed = config.get('seed', 0)


  logging.info('Config: %s', config)

  _load_data_from_files(config)

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

  # Initialize bert embedding preprocessor.
  if config.shared_bert_embedding:
    bert_preprocess_model = utils.BertPreprocessor(
        config.bert_embedding_preprocess_tfhub_url,
        config.model.vae_cell.max_seq_length)
    if bert_preprocess_model.vocab_size != config.model.vocab_size:
      raise ValueError(
          'Expect BERT preprocess model vocab size align with the model '
          'config, found {} and {}.'.format(bert_preprocess_model.vocab_size,
                                            config.model.vocab_size))
    preprocess_fn = preprocessor.BertDataPreprocessor(
        bert_preprocess_model, config.model.num_states,
        labeled_dialog_turn_ids).create_feature_and_label

  else:
    preprocess_fn = preprocessor.DataPreprocessor(
        config.model.num_states,
        labeled_dialog_turn_ids).create_feature_and_label

  # Load PSL configs
  psl_learning = config.psl_constraint_learning_weight > 0
  psl_inference = config.psl_constraint_inference_weight > 0
  if psl_learning or psl_inference:
    with tf.io.gfile.GFile(config.vocab_file_path, 'r') as f:
      vocab = f.read()[:-1].split('\n')
    preprocess_fn = psl_utils.psl_feature_mixin(preprocess_fn, config.dataset,
                                                config.psl, vocab)

  # Load datasets
  # TODO(yquan): invesigate why distributed training fails when using BERT
  # Failure example: https://xm2a.corp.google.com/experiments/33275459
  distributed_training = (not psl_learning and not psl_inference and
                          not config.shared_bert_embedding)
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

  distributed_inference = not config.shared_bert_embedding
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
  word_weights = np.ones((config.model.vocab_size), dtype=np.float32)
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

    metrics = _create_metrics(_SPLITS, config.few_shots,
                              config.few_shots_l2_weights,
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
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint, directory=output_dir, max_to_keep=None)
    if model_dir:
      best_model_checkpoint_manager = tf.train.CheckpointManager(
          checkpoint, directory=model_dir, max_to_keep=1)
    else:
      best_model_checkpoint_manager = None
    # checkpoint.restore must be within a strategy.scope() so that optimizer
    # slot variables are mirrored.
    latest_checkpoint = checkpoint_manager.restore_or_initialize()
    initial_epoch = 0
    if latest_checkpoint:
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
      logging.info('Loaded checkpoint %s. Initialize from epoch %s',
                   latest_checkpoint, initial_epoch)
    elif config.shared_bert_embedding:
      # load BERT from initial checkpoint
      bert_ckpt_dir = config.model.vae_cell.shared_bert_embedding_ckpt_dir
      (model.vae_cell.shared_embedding_layer, _,
       _) = bert_utils.load_bert_weight_from_ckpt(
           bert_model=model.vae_cell.shared_embedding_layer,
           bert_ckpt_dir=bert_ckpt_dir)
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

  def train_step(batch_size: int, config: config_dict.ConfigDict):

    @tf.function
    def _train_step(inputs: Sequence[tf.Tensor]):
      """Training step function."""

      (input_1, input_2, label_id, label_mask, initial_state, initial_sample,
       _) = inputs[:7]
      if psl_learning:
        psl_inputs = inputs[-1]
        # Explicitly specify the batch size as PSL model now requires known
        # batch size.
        psl_inputs = tf.ensure_shape(
            psl_inputs, (batch_size, psl_inputs.shape[1], psl_inputs.shape[2]))
      else:
        psl_inputs = None

      model_inputs = [input_1, input_2, initial_state, initial_sample]
      if with_label:
        model_inputs.extend([label_id, label_mask])

      with tf.GradientTape() as tape:
        # Set learning phase to enable dropout etc during training.
        model_outputs = model(model_inputs, training=True)

        losses = linear_vrnn.compute_loss(
            input_1[_INPUT_ID_NAME],
            input_2[_INPUT_ID_NAME],
            input_1[_INPUT_MASK_NAME],
            input_2[_INPUT_MASK_NAME],
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

      logits = linear_vrnn.get_logits(model_outputs)
      _update_metrics(metrics, _TRAIN, logits, losses, label_id, label_mask,
                      config.psl_constraint_rule_names)

    return _train_step

  def test_step(split: str, batch_size: int, config: config_dict.ConfigDict):

    @tf.function
    def _test_step(inputs: Sequence[tf.Tensor]):
      """Evaluation step function."""

      (input_1, input_2, label_id, label_mask, initial_state, initial_sample,
       _) = inputs[:7]
      if psl_inference:
        psl_inputs = inputs[-1]
        # Explicitly specify the batch size as PSL model now requires known
        # batch size.
        psl_inputs = tf.ensure_shape(
            psl_inputs, (batch_size, psl_inputs.shape[1], psl_inputs.shape[2]))
      else:
        psl_inputs = None

      # In evaluation, don't provide label as a guidance.
      model_inputs = [input_1, input_2, initial_state, initial_sample]
      model_outputs = model(model_inputs, training=False)

      losses = linear_vrnn.compute_loss(
          input_1[_INPUT_ID_NAME],
          input_2[_INPUT_ID_NAME],
          input_1[_INPUT_MASK_NAME],
          input_2[_INPUT_MASK_NAME],
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

      if psl_inference:
        logits = psl_utils.update_logits(model, psl_optimizer, model_inputs,
                                         linear_vrnn.get_logits, psl_model,
                                         psl_inputs,
                                         config.psl_constraint_inference_steps,
                                         config.psl_constraint_inference_weight)
      else:
        logits = linear_vrnn.get_logits(model_outputs)

      _update_metrics(metrics, split, logits, losses, label_id, label_mask,
                      config.psl_constraint_rule_names)

    return _test_step

  @tf.function
  def inference_step(inputs: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    (input_1, input_2, label, _, initial_state, initial_sample,
     domain_label) = inputs[:7]
    model_inputs = [input_1, input_2, initial_state, initial_sample]
    model_outputs = model(model_inputs, training=False)

    prediction = linear_vrnn.get_prediction(
        linear_vrnn.get_logits(model_outputs))
    latent_state = model_outputs[0]

    return latent_state, label, prediction, domain_label


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
      inference_step,
      strategy,
      distributed=distributed_inference,
      output_dtypes=[tf.float32, tf.int32, tf.int32, tf.int32],
  )

  primary_metric = tf.constant(0.)
  out_of_patience = 0
  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, config.train_epochs):
    if out_of_patience > config.patience:
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

    if epoch % config.evaluation_interval == 0:
      for dataset_name, test_dataset in test_datasets.items():
        test_iterator = iter(test_dataset)
        logging.info('Testing on dataset %s', dataset_name)
        logging.info('Starting to run eval at epoch: %s', epoch)
        run_test_steps_map[dataset_name](test_iterator,
                                         tf.cast(steps_per_eval[dataset_name],
                                                 tf.int32))
        logging.info('Done with testing on %s', dataset_name)

      (train_hidden_state, train_label, train_prediction,
       train_domain_label) = run_inference_steps(
           iter(inference_train_dataset), num_inference_train_steps)
      (test_hidden_state, test_label, test_prediction,
       test_domain_label) = run_inference_steps(
           iter(inference_test_dataset), num_inference_test_steps)


      logging.info('Evaluating hidden representation learning.')

      input_size = config.model.vae_cell.encoder_projection_sizes[-1]
      train_results, test_results = _evaluate_hidden_state_model(
          input_size, config.model.num_states, train_hidden_state, train_label,
          test_hidden_state, test_label, config.hidden_state_model_train_epochs,
          config.hidden_state_model_learning_rate)
      domain_train_results, domain_test_results = _evaluate_hidden_state_model(
          input_size, data_utils.get_dataset_num_domains(config.dataset),
          train_hidden_state, train_domain_label, test_hidden_state,
          test_domain_label, config.hidden_state_model_train_epochs,
          config.hidden_state_model_learning_rate)
      _update_hidden_state_model_metrics(metrics, _SPLITS, [
          train_results + domain_train_results,
          test_results + domain_test_results
      ])

      _update_clustering_metrics(metrics, _TRAIN, train_label, train_prediction)
      _update_clustering_metrics(metrics, _TEST, test_label, test_prediction)

      for split in _SPLITS:
        _log_metric_results(metrics, split)

      total_results = {
          name: metric.result() for name, metric in metrics.items()
      }
      with summary_writer.as_default():
        for name, result in total_results.items():
          tf.summary.scalar(name, result, step=epoch + 1)

      if _primary_metric_improved(total_results, primary_metric,
                                  config.min_delta):
        primary_metric = total_results[_PRIMARY_METRIC_KEY]
        out_of_patience = 0
        if best_model_checkpoint_manager:
          best_model_checkpoint_manager.save()
        if best_summary_writer:
          with best_summary_writer.as_default():
            for name, result in total_results.items():
              tf.summary.scalar(name, result, step=epoch + 1)
        if model_dir:
          _save_model_results([
              train_hidden_state, train_label, train_prediction,
              train_domain_label
          ], model_dir, _TRAIN)
          _save_model_results([
              test_hidden_state, test_label, test_prediction, test_domain_label
          ], model_dir, _TEST)
      else:
        out_of_patience += 1


    for metric in metrics.values():
      metric.reset_states()

    if (config.checkpoint_interval > 0 and
        (epoch + 1) % config.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(os.path.join(output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)


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
