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

"""BERT model with Monte Carlo dropout.

This script trains model on WikipediaTalk data, and evaluate on both
WikipediaTalk and CivilComment datasets.
"""

import os
import time

from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
from tensorflow_addons import losses as tfa_losses
from tensorflow_addons import metrics as tfa_metrics

import uncertainty_baselines as ub
from  baselines.toxic_comments import metrics as tc_metrics  # local file import
from  baselines.toxic_comments import utils  # local file import
from uncertainty_baselines.datasets import toxic_comments as ds
from tensorboard.plugins.hparams import api as hp

# Data flags
flags.DEFINE_string(
    'in_dataset_dir', None,
    'Path to in-domain dataset (WikipediaToxicityDataset).')
flags.DEFINE_string(
    'ood_dataset_dir', None,
    'Path to out-of-domain dataset (CivilCommentsDataset).')
flags.DEFINE_string(
    'identity_dataset_dir', None,
    'Path to out-of-domain dataset with identity labels '
    '(CivilCommentsIdentitiesDataset).')

# Model flags
flags.DEFINE_string('model_family', 'bert',
                    'Types of model to use. Can be either TextCNN or BERT.')

# Model flags, BERT.
flags.DEFINE_string(
    'bert_dir', None,
    'Directory to BERT pre-trained checkpoints and config files.')
flags.DEFINE_string(
    'bert_ckpt_dir', None, 'Directory to BERT pre-trained checkpoints. '
    'If None then then default to {bert_dir}/bert_model.ckpt.')
flags.DEFINE_string(
    'bert_config_dir', None, 'Directory to BERT config files. '
    'If None then then default to {bert_dir}/bert_config.json.')

# Dropout flags
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_bool(
    'channel_wise_dropout_all', True,
    'Whether to apply channel-wise dropout for all layers.')
flags.DEFINE_bool(
    'channel_wise_dropout_mha', False,
    'Whether to apply channel-wise dropout to the multi-head attention layer.')
flags.DEFINE_bool(
    'channel_wise_dropout_att', False,
    'Whether to apply channel-wise dropout to the attention output layer.')
flags.DEFINE_bool(
    'channel_wise_dropout_ffn', False,
    'Whether to apply channel-wise dropout to the hidden feedforward layer.')

flags.DEFINE_bool(
    'use_mc_dropout_mha', False,
    'Whether to apply Monte Carlo dropout to the multi-head attention layer.')
flags.DEFINE_bool(
    'use_mc_dropout_att', True,
    'Whether to apply Monte Carlo dropout to the attention output layer.')
flags.DEFINE_bool(
    'use_mc_dropout_ffn', True,
    'Whether to apply Monte Carlo dropout to the hidden feedforward layer.')
flags.DEFINE_bool(
    'use_mc_dropout_output', False,
    'Whether to apply Monte Carlo dropout to the dense output layer.')

# Optimization and evaluation flags
flags.DEFINE_integer('seed', 8, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 32, 'Batch size per TPU core/GPU.')
flags.DEFINE_float(
    'base_learning_rate', 5e-5,
    'Base learning rate when total batch size is 128. It is '
    'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'checkpoint_interval', 5,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 1,
                     'Number of epochs between evaluation.')
flags.DEFINE_integer('num_ece_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_integer(
    'num_approx_bins', 1000,
    'Number of bins for approximating collaborative and abstention metrics.')
flags.DEFINE_list(
    'fractions',
    ['0.0', '0.001', '0.005', '0.01', '0.02', '0.05', '0.1', '0.15', '0.2'],
    'A list of fractions of total examples to send to '
    'the moderators (up to 1).')
flags.DEFINE_string('output_dir', '/tmp/toxic_comments', 'Output directory.')
flags.DEFINE_integer('train_epochs', 5, 'Number of training epochs.')
flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')
flags.DEFINE_float(
    'ece_label_threshold', 0.7,
    'Threshold used to convert toxicity score into binary labels for computing '
    'Expected Calibration Error (ECE). Default is 0.7 which is the threshold '
    'value recommended by Jigsaw team.')

# Loss type
flags.DEFINE_enum('loss_type', 'cross_entropy',
                  ['cross_entropy', 'focal_cross_entropy', 'mse', 'mae'],
                  'Type of loss function to use.')
flags.DEFINE_float('focal_loss_alpha', 0.1,
                   'Multiplicative factor used in the focal loss [1]-[2] to '
                   'downweight common cases.')
flags.DEFINE_float('focal_loss_gamma', 5.,
                   'Exponentiate factor used in the focal loss [1]-[2] to '
                   'push model to minimize in-confident examples.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS


_MAX_SEQ_LENGTH = 512


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Model checkpoint will be saved at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  test_batch_size = batch_size
  data_buffer_size = batch_size * 10

  train_dataset_builder = ds.WikipediaToxicityDataset(
      split='train',
      data_dir=FLAGS.in_dataset_dir,
      shuffle_buffer_size=data_buffer_size)
  ind_dataset_builder = ds.WikipediaToxicityDataset(
      split='test',
      data_dir=FLAGS.in_dataset_dir,
      shuffle_buffer_size=data_buffer_size)
  ood_dataset_builder = ds.CivilCommentsDataset(
      split='test',
      data_dir=FLAGS.ood_dataset_dir,
      shuffle_buffer_size=data_buffer_size)
  ood_identity_dataset_builder = ds.CivilCommentsIdentitiesDataset(
      split='test',
      data_dir=FLAGS.identity_dataset_dir,
      shuffle_buffer_size=data_buffer_size)

  train_dataset_builders = {
      'wikipedia_toxicity_subtypes': train_dataset_builder
  }
  test_dataset_builders = {
      'ind': ind_dataset_builder,
      'ood': ood_dataset_builder,
      'ood_identity': ood_identity_dataset_builder,
  }
  if FLAGS.prediction_mode and FLAGS.identity_prediction:
    for dataset_name in utils.IDENTITY_LABELS:
      if utils.NUM_EXAMPLES[dataset_name]['test'] > 100:
        test_dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test',
            data_dir=os.path.join(
                FLAGS.identity_specific_dataset_dir, dataset_name),
            shuffle_buffer_size=data_buffer_size)
    for dataset_name in utils.IDENTITY_TYPES:
      if utils.NUM_EXAMPLES[dataset_name]['test'] > 100:
        test_dataset_builders[dataset_name] = ds.CivilCommentsIdentitiesDataset(
            split='test',
            data_dir=os.path.join(
                FLAGS.identity_type_dataset_dir, dataset_name),
            shuffle_buffer_size=data_buffer_size)

  class_weight = utils.create_class_weight(
      train_dataset_builders, test_dataset_builders)
  logging.info('class_weight: %s', str(class_weight))

  ds_info = train_dataset_builder.tfds_info
  # Positive and negative classes.
  num_classes = ds_info.metadata['num_classes']

  train_datasets = {}
  dataset_steps_per_epoch = {}
  total_steps_per_epoch = 0

  # TODO(jereliu): Apply strategy.experimental_distribute_dataset to the
  # dataset_builders.
  for dataset_name, dataset_builder in train_dataset_builders.items():
    train_datasets[dataset_name] = dataset_builder.load(
        batch_size=FLAGS.per_core_batch_size)
    dataset_steps_per_epoch[dataset_name] = (
        dataset_builder.num_examples // batch_size)
    total_steps_per_epoch += dataset_steps_per_epoch[dataset_name]

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in test_dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=test_batch_size)
    if dataset_name in ['ind', 'ood', 'ood_identity']:
      steps_per_eval[dataset_name] = (
          dataset_builder.num_examples // test_batch_size)
    else:
      steps_per_eval[dataset_name] = (
          utils.NUM_EXAMPLES[dataset_name]['test'] // test_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building %s model', FLAGS.model_family)

    bert_config_dir, bert_ckpt_dir = utils.resolve_bert_ckpt_and_config_dir(
        FLAGS.bert_model_type, FLAGS.bert_dir, FLAGS.bert_config_dir,
        FLAGS.bert_ckpt_dir)
    bert_config = utils.create_config(bert_config_dir)
    bert_config.hidden_dropout_prob = FLAGS.dropout_rate
    bert_config.attention_probs_dropout_prob = FLAGS.dropout_rate
    model, bert_encoder = ub.models.bert_dropout_model(
        num_classes=num_classes,
        bert_config=bert_config,
        use_mc_dropout_mha=FLAGS.use_mc_dropout_mha,
        use_mc_dropout_att=FLAGS.use_mc_dropout_att,
        use_mc_dropout_ffn=FLAGS.use_mc_dropout_ffn,
        use_mc_dropout_output=FLAGS.use_mc_dropout_output,
        channel_wise_dropout_mha=FLAGS.channel_wise_dropout_mha,
        channel_wise_dropout_att=FLAGS.channel_wise_dropout_att,
        channel_wise_dropout_ffn=FLAGS.channel_wise_dropout_ffn)

    # Create an AdamW optimizer with beta_2=0.999, epsilon=1e-6.
    optimizer = utils.create_optimizer(
        FLAGS.base_learning_rate,
        steps_per_epoch=total_steps_per_epoch,
        epochs=FLAGS.train_epochs,
        warmup_proportion=FLAGS.warmup_proportion,
        beta_1=1.0 - FLAGS.one_minus_momentum)

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    metrics = {
        'train/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'train/accuracy':
            tf.keras.metrics.Accuracy(),
        'train/accuracy_weighted':
            tf.keras.metrics.Accuracy(),
        'train/auroc':
            tf.keras.metrics.AUC(),
        'train/loss':
            tf.keras.metrics.Mean(),
        'train/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_ece_bins),
        'train/precision':
            tf.keras.metrics.Precision(),
        'train/recall':
            tf.keras.metrics.Recall(),
        'train/f1':
            tfa_metrics.F1Score(
                num_classes=num_classes,
                average='micro',
                threshold=FLAGS.ece_label_threshold),
    }

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if FLAGS.prediction_mode:
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.eval_checkpoint_dir)
    else:
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // total_steps_per_epoch
    elif FLAGS.model_family.lower() == 'bert':
      # load BERT from initial checkpoint
      bert_checkpoint = tf.train.Checkpoint(model=bert_encoder)
      bert_checkpoint.restore(bert_ckpt_dir).assert_existing_objects_matched()
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

    metrics.update({
        'test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'test/auroc':
            tf.keras.metrics.AUC(curve='ROC'),
        'test/aupr':
            tf.keras.metrics.AUC(curve='PR'),
        'test/brier':
            tf.keras.metrics.MeanSquaredError(),
        'test/brier_weighted':
            tf.keras.metrics.MeanSquaredError(),
        'test/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_ece_bins),
        'test/acc':
            tf.keras.metrics.Accuracy(),
        'test/acc_weighted':
            tf.keras.metrics.Accuracy(),
        'test/eval_time':
            tf.keras.metrics.Mean(),
        'test/precision':
            tf.keras.metrics.Precision(),
        'test/recall':
            tf.keras.metrics.Recall(),
        'test/f1':
            tfa_metrics.F1Score(
                num_classes=num_classes,
                average='micro',
                threshold=FLAGS.ece_label_threshold)
    })

    for policy in ('uncertainty', 'toxicity'):
      metrics.update({
          'test_{}/calibration_auroc'.format(policy):
              tc_metrics.CalibrationAUC(curve='ROC'),
          'test_{}/calibration_auprc'.format(policy):
              tc_metrics.CalibrationAUC(curve='PR')
      })

      for fraction in FLAGS.fractions:
        metrics.update({
            'test_{}/collab_acc_{}'.format(policy, fraction):
                rm.metrics.OracleCollaborativeAccuracy(
                    fraction=float(fraction), num_bins=FLAGS.num_approx_bins),
            'test_{}/abstain_prec_{}'.format(policy, fraction):
                tc_metrics.AbstainPrecision(
                    abstain_fraction=float(fraction),
                    num_approx_bins=FLAGS.num_approx_bins),
            'test_{}/abstain_recall_{}'.format(policy, fraction):
                tc_metrics.AbstainRecall(
                    abstain_fraction=float(fraction),
                    num_approx_bins=FLAGS.num_approx_bins),
            'test_{}/collab_auroc_{}'.format(policy, fraction):
                tc_metrics.OracleCollaborativeAUC(
                    oracle_fraction=float(fraction),
                    num_bins=FLAGS.num_approx_bins),
            'test_{}/collab_auprc_{}'.format(policy, fraction):
                tc_metrics.OracleCollaborativeAUC(
                    oracle_fraction=float(fraction),
                    curve='PR',
                    num_bins=FLAGS.num_approx_bins),
        })

    for dataset_name, test_dataset in test_datasets.items():
      if dataset_name != 'ind':
        metrics.update({
            'test/nll_{}'.format(dataset_name):
                tf.keras.metrics.Mean(),
            'test/auroc_{}'.format(dataset_name):
                tf.keras.metrics.AUC(curve='ROC'),
            'test/aupr_{}'.format(dataset_name):
                tf.keras.metrics.AUC(curve='PR'),
            'test/brier_{}'.format(dataset_name):
                tf.keras.metrics.MeanSquaredError(),
            'test/brier_weighted_{}'.format(dataset_name):
                tf.keras.metrics.MeanSquaredError(),
            'test/ece_{}'.format(dataset_name):
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_ece_bins
                                                   ),
            'test/acc_{}'.format(dataset_name):
                tf.keras.metrics.Accuracy(),
            'test/acc_weighted_{}'.format(dataset_name):
                tf.keras.metrics.Accuracy(),
            'test/eval_time_{}'.format(dataset_name):
                tf.keras.metrics.Mean(),
            'test/precision_{}'.format(dataset_name):
                tf.keras.metrics.Precision(),
            'test/recall_{}'.format(dataset_name):
                tf.keras.metrics.Recall(),
            'test/f1_{}'.format(dataset_name):
                tfa_metrics.F1Score(
                    num_classes=num_classes,
                    average='micro',
                    threshold=FLAGS.ece_label_threshold)
        })

        for policy in ('uncertainty', 'toxicity'):
          metrics.update({
              'test_{}/calibration_auroc_{}'.format(policy, dataset_name):
                  tc_metrics.CalibrationAUC(curve='ROC'),
              'test_{}/calibration_auprc_{}'.format(policy, dataset_name):
                  tc_metrics.CalibrationAUC(curve='PR'),
          })

          for fraction in FLAGS.fractions:
            metrics.update({
                'test_{}/collab_acc_{}_{}'.format(policy, fraction,
                                                  dataset_name):
                    rm.metrics.OracleCollaborativeAccuracy(
                        fraction=float(fraction),
                        num_bins=FLAGS.num_approx_bins),
                'test_{}/abstain_prec_{}_{}'.format(policy, fraction,
                                                    dataset_name):
                    tc_metrics.AbstainPrecision(
                        abstain_fraction=float(fraction),
                        num_approx_bins=FLAGS.num_approx_bins),
                'test_{}/abstain_recall_{}_{}'.format(policy, fraction,
                                                      dataset_name):
                    tc_metrics.AbstainRecall(
                        abstain_fraction=float(fraction),
                        num_approx_bins=FLAGS.num_approx_bins),
                'test_{}/collab_auroc_{}_{}'.format(policy, fraction,
                                                    dataset_name):
                    tc_metrics.OracleCollaborativeAUC(
                        oracle_fraction=float(fraction),
                        num_bins=FLAGS.num_approx_bins),
                'test_{}/collab_auprc_{}_{}'.format(policy, fraction,
                                                    dataset_name):
                    tc_metrics.OracleCollaborativeAUC(
                        oracle_fraction=float(fraction),
                        curve='PR',
                        num_bins=FLAGS.num_approx_bins),
            })

  @tf.function
  def generate_sample_weight(labels, class_weight, label_threshold=0.7):
    """Generate sample weight for weighted accuracy calculation."""
    if label_threshold != 0.7:
      logging.warning('The class weight was based on `label_threshold` = 0.7, '
                      'and weighted accuracy/brier will be meaningless if '
                      '`label_threshold` is not equal to this value, which is '
                      'recommended by Jigsaw Conversation AI team.')
    labels_int = tf.cast(labels > label_threshold, tf.int32)
    sample_weight = tf.gather(class_weight, labels_int)
    return sample_weight

  @tf.function
  def train_step(iterator, dataset_name, num_steps):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels, _ = utils.create_feature_and_label(inputs)

      with tf.GradientTape() as tape:
        logits = model(features, training=True)

        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        loss_logits = tf.squeeze(logits, axis=1)
        if FLAGS.loss_type == 'cross_entropy':
          logging.info('Using cross entropy loss')
          negative_log_likelihood = tf.nn.sigmoid_cross_entropy_with_logits(
              labels, loss_logits)
        elif FLAGS.loss_type == 'focal_cross_entropy':
          logging.info('Using focal cross entropy loss')
          negative_log_likelihood = tfa_losses.sigmoid_focal_crossentropy(
              labels, loss_logits,
              alpha=FLAGS.focal_loss_alpha, gamma=FLAGS.focal_loss_gamma,
              from_logits=True)
        elif FLAGS.loss_type == 'mse':
          logging.info('Using mean squared error loss')
          loss_probs = tf.nn.sigmoid(loss_logits)
          negative_log_likelihood = tf.keras.losses.mean_squared_error(
              labels, loss_probs)
        elif FLAGS.loss_type == 'mae':
          logging.info('Using mean absolute error loss')
          loss_probs = tf.nn.sigmoid(loss_logits)
          negative_log_likelihood = tf.keras.losses.mean_absolute_error(
              labels, loss_probs)

        negative_log_likelihood = tf.reduce_mean(negative_log_likelihood)

        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.sigmoid(logits)
      # Cast labels to discrete for ECE computation.
      ece_labels = tf.cast(labels > FLAGS.ece_label_threshold, tf.float32)
      one_hot_labels = tf.one_hot(tf.cast(ece_labels, tf.int32),
                                  depth=num_classes)
      ece_probs = tf.concat([1. - probs, probs], axis=1)
      auc_probs = tf.squeeze(probs, axis=1)
      pred_labels = tf.math.argmax(ece_probs, axis=-1)

      sample_weight = generate_sample_weight(
          labels, class_weight['train/{}'.format(dataset_name)],
          FLAGS.ece_label_threshold)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, pred_labels)
      metrics['train/accuracy_weighted'].update_state(
          ece_labels, pred_labels, sample_weight=sample_weight)
      metrics['train/auroc'].update_state(labels, auc_probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/ece'].add_batch(ece_probs, label=ece_labels)
      metrics['train/precision'].update_state(ece_labels, pred_labels)
      metrics['train/recall'].update_state(ece_labels, pred_labels)
      metrics['train/f1'].update_state(one_hot_labels, ece_probs)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn to log metrics."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels, _ = utils.create_feature_and_label(inputs)

      eval_start_time = time.time()
      logits = model(features, training=False)
      eval_time = (time.time() - eval_start_time) / FLAGS.per_core_batch_size

      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.sigmoid(logits)
      # Cast labels to discrete for ECE computation.
      ece_labels = tf.cast(labels > FLAGS.ece_label_threshold, tf.float32)
      one_hot_labels = tf.one_hot(tf.cast(ece_labels, tf.int32),
                                  depth=num_classes)
      ece_probs = tf.concat([1. - probs, probs], axis=1)
      pred_labels = tf.math.argmax(ece_probs, axis=-1)
      auc_probs = tf.squeeze(probs, axis=1)

      loss_logits = tf.squeeze(logits, axis=1)
      negative_log_likelihood = tf.reduce_mean(
          tf.nn.sigmoid_cross_entropy_with_logits(labels, loss_logits))

      # Use normalized binary predictive variance as the confidence score.
      # Since the prediction variance p*(1-p) is within range (0, 0.25),
      # normalize it by maximum value so the confidence is between (0, 1).
      calib_confidence = 1. - probs * (1. - probs) / .25

      sample_weight = generate_sample_weight(
          labels, class_weight['test/{}'.format(dataset_name)],
          FLAGS.ece_label_threshold)
      if dataset_name == 'ind':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/auroc'].update_state(labels, auc_probs)
        metrics['test/aupr'].update_state(labels, auc_probs)
        metrics['test/brier'].update_state(labels, auc_probs)
        metrics['test/brier_weighted'].update_state(
            tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
        metrics['test/ece'].add_batch(ece_probs, label=ece_labels)
        metrics['test/acc'].update_state(ece_labels, pred_labels)
        metrics['test/acc_weighted'].update_state(
            ece_labels, pred_labels, sample_weight=sample_weight)
        metrics['test/eval_time'].update_state(eval_time)
        metrics['test/precision'].update_state(ece_labels, pred_labels)
        metrics['test/recall'].update_state(ece_labels, pred_labels)
        metrics['test/f1'].update_state(one_hot_labels, ece_probs)

        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.squeeze(confidence)

          metrics['test_{}/calibration_auroc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)
          metrics['test_{}/calibration_auprc'.format(policy)].update_state(
              ece_labels, pred_labels, confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}'.format(policy, fraction)].add_batch(
                ece_probs,
                label=ece_labels,
                custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/abstain_recall_{}'.format(
                policy, fraction)].update_state(ece_labels, pred_labels,
                                                confidence)
            metrics['test_{}/collab_auroc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}'.format(
                policy, fraction)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/auroc_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/aupr_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/brier_{}'.format(dataset_name)].update_state(
            labels, auc_probs)
        metrics['test/brier_weighted_{}'.format(dataset_name)].update_state(
            tf.expand_dims(labels, -1), probs, sample_weight=sample_weight)
        metrics['test/ece_{}'.format(dataset_name)].add_batch(
            ece_probs, label=ece_labels)
        metrics['test/acc_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/acc_weighted_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels, sample_weight=sample_weight)
        metrics['test/eval_time_{}'.format(dataset_name)].update_state(
            eval_time)
        metrics['test/precision_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/recall_{}'.format(dataset_name)].update_state(
            ece_labels, pred_labels)
        metrics['test/f1_{}'.format(dataset_name)].update_state(
            one_hot_labels, ece_probs)

        for policy in ('uncertainty', 'toxicity'):
          # calib_confidence or decreasing toxicity score.
          confidence = 1. - probs if policy == 'toxicity' else calib_confidence
          binning_confidence = tf.squeeze(confidence)

          metrics['test_{}/calibration_auroc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)
          metrics['test_{}/calibration_auprc_{}'.format(
              policy, dataset_name)].update_state(ece_labels, pred_labels,
                                                  confidence)

          for fraction in FLAGS.fractions:
            metrics['test_{}/collab_acc_{}_{}'.format(
                policy, fraction, dataset_name)].add_batch(
                    ece_probs,
                    label=ece_labels,
                    custom_binning_score=binning_confidence)
            metrics['test_{}/abstain_prec_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/abstain_recall_{}_{}'.format(
                policy, fraction,
                dataset_name)].update_state(ece_labels, pred_labels, confidence)
            metrics['test_{}/collab_auroc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)
            metrics['test_{}/collab_auprc_{}_{}'.format(
                policy, fraction, dataset_name)].update_state(
                    labels, auc_probs, custom_binning_score=binning_confidence)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def final_eval_step(iterator):
    """Final Evaluation StepFn to save prediction to directory."""

    def step_fn(inputs):
      bert_features, labels, additional_labels = utils.create_feature_and_label(
          inputs)
      logits = model(bert_features, training=False)
      features = inputs['input_ids']
      return features, logits, labels, additional_labels

    (per_replica_texts, per_replica_logits, per_replica_labels,
     per_replica_additional_labels) = (
         strategy.run(step_fn, args=(next(iterator),)))

    if strategy.num_replicas_in_sync > 1:
      texts_list = tf.concat(per_replica_texts.values, axis=0)
      logits_list = tf.concat(per_replica_logits.values, axis=0)
      labels_list = tf.concat(per_replica_labels.values, axis=0)
      additional_labels_dict = {}
      for additional_label in utils.IDENTITY_LABELS:
        if additional_label in per_replica_additional_labels:
          additional_labels_dict[additional_label] = tf.concat(
              per_replica_additional_labels[additional_label], axis=0)
    else:
      texts_list = per_replica_texts
      logits_list = per_replica_logits
      labels_list = per_replica_labels
      additional_labels_dict = {}
      for additional_label in utils.IDENTITY_LABELS:
        if additional_label in per_replica_additional_labels:
          additional_labels_dict[
              additional_label] = per_replica_additional_labels[
                  additional_label]

    return texts_list, logits_list, labels_list, additional_labels_dict

  if FLAGS.prediction_mode:
    # Prediction and exit.
    for dataset_name, test_dataset in test_datasets.items():
      test_iterator = iter(test_dataset)  # pytype: disable=wrong-arg-types
      message = 'Final eval on dataset {}'.format(dataset_name)
      logging.info(message)

      texts_all = []
      logits_all = []
      labels_all = []
      additional_labels_all_dict = {}
      if 'identity' in dataset_name:
        for identity_label_name in utils.IDENTITY_LABELS:
          additional_labels_all_dict[identity_label_name] = []

      try:
        with tf.experimental.async_scope():
          for step in range(steps_per_eval[dataset_name]):
            if step % 20 == 0:
              message = 'Starting to run eval step {}/{} of dataset: {}'.format(
                  step, steps_per_eval[dataset_name], dataset_name)
              logging.info(message)

            (text_step, logits_step, labels_step,
             additional_labels_dict_step) = final_eval_step(test_iterator)

            texts_all.append(text_step)
            logits_all.append(logits_step)
            labels_all.append(labels_step)
            if 'identity' in dataset_name:
              for identity_label_name in utils.IDENTITY_LABELS:
                additional_labels_all_dict[identity_label_name].append(
                    additional_labels_dict_step[identity_label_name])

      except (StopIteration, tf.errors.OutOfRangeError):
        tf.experimental.async_clear_error()
        logging.info('Done with eval on %s', dataset_name)

      texts_all = tf.concat(texts_all, axis=0)
      logits_all = tf.concat(logits_all, axis=0)
      labels_all = tf.concat(labels_all, axis=0)
      additional_labels_all = []
      if additional_labels_all_dict:
        for identity_label_name in utils.IDENTITY_LABELS:
          additional_labels_all.append(
              tf.concat(
                  additional_labels_all_dict[identity_label_name], axis=0))
      additional_labels_all = tf.convert_to_tensor(additional_labels_all)

      utils.save_prediction(
          texts_all.numpy(),
          path=os.path.join(FLAGS.output_dir, 'texts_{}'.format(dataset_name)))
      utils.save_prediction(
          labels_all.numpy(),
          path=os.path.join(FLAGS.output_dir, 'labels_{}'.format(dataset_name)))
      utils.save_prediction(
          logits_all.numpy(),
          path=os.path.join(FLAGS.output_dir, 'logits_{}'.format(dataset_name)))
      if 'identity' in dataset_name:
        utils.save_prediction(
            additional_labels_all.numpy(),
            path=os.path.join(FLAGS.output_dir,
                              'additional_labels_{}'.format(dataset_name)))
      logging.info('Done with testing on %s', dataset_name)

  else:
    # Execute train / eval loop.
    start_time = time.time()
    train_iterators = {}
    for dataset_name, train_dataset in train_datasets.items():
      train_iterators[dataset_name] = iter(train_dataset)
    for epoch in range(initial_epoch, FLAGS.train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      for dataset_name, train_iterator in train_iterators.items():
        train_step(
            train_iterator, dataset_name, dataset_steps_per_epoch[dataset_name])

        current_step = (
            epoch * total_steps_per_epoch +
            dataset_steps_per_epoch[dataset_name])
        max_steps = total_steps_per_epoch * FLAGS.train_epochs
        time_elapsed = time.time() - start_time
        steps_per_sec = float(current_step) / time_elapsed
        eta_seconds = (max_steps - current_step) / steps_per_sec
        message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                   'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                       current_step / max_steps, epoch + 1,
                       FLAGS.train_epochs, steps_per_sec, eta_seconds / 60,
                       time_elapsed / 60))
        logging.info(message)

      if epoch % FLAGS.evaluation_interval == 0:
        for dataset_name, test_dataset in test_datasets.items():
          test_iterator = iter(test_dataset)  # pytype: disable=wrong-arg-types
          logging.info('Testing on dataset %s', dataset_name)

          try:
            with tf.experimental.async_scope():
              for step in range(steps_per_eval[dataset_name]):
                if step % 20 == 0:
                  logging.info('Starting to run eval step %s/%s of epoch: %s',
                               step, steps_per_eval[dataset_name], epoch)
                test_step(test_iterator, dataset_name)
          except (StopIteration, tf.errors.OutOfRangeError):
            tf.experimental.async_clear_error()
            logging.info('Done with testing on %s', dataset_name)

        logging.info('Train Loss: %.4f, AUROC: %.4f',
                     metrics['train/loss'].result(),
                     metrics['train/auroc'].result())
        logging.info('Test NLL: %.4f, AUROC: %.4f',
                     metrics['test/negative_log_likelihood'].result(),
                     metrics['test/auroc'].result())

        # record results
        total_results = {
            name: metric.result() for name, metric in metrics.items()
        }
        # Metrics from Robustness Metrics (like ECE) will return a dict with a
        # single key/value, instead of a scalar.
        total_results = {
            k: (list(v.values())[0] if isinstance(v, dict) else v)
            for k, v in total_results.items()
        }

        with summary_writer.as_default():
          for name, result in total_results.items():
            tf.summary.scalar(name, result, step=epoch + 1)

      for name, metric in metrics.items():
        metric.reset_states()

      checkpoint_interval = min(FLAGS.checkpoint_interval, FLAGS.train_epochs)
      if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
        checkpoint_name = checkpoint.save(
            os.path.join(FLAGS.output_dir, 'checkpoint'))
        logging.info('Saved checkpoint to %s', checkpoint_name)

    # Save model in SavedModel format on exit.
    final_save_name = os.path.join(FLAGS.output_dir, 'model')
    model.save(final_save_name)
    logging.info('Saved model to %s', final_save_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'dropout_rate': FLAGS.dropout_rate,
    })


if __name__ == '__main__':
  app.run(main)
