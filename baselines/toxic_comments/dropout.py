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

"""BERT model with Monte Carlo dropout.

This script trains model on WikipediaTalk data, and evaluate on both
WikipediaTalk and CivilComment datasets.
"""

import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
from tensorflow_addons import losses as tfa_losses

import uncertainty_baselines as ub
import utils  # local file import from baselines.toxic_comments
from tensorboard.plugins.hparams import api as hp

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

# Optimization flags.
flags.DEFINE_integer('seed', 8, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 32, 'Batch size per TPU core/GPU.')
flags.DEFINE_float(
    'base_learning_rate', 5e-5,
    'Base learning rate when total batch size is 128. It is '
    'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')
flags.DEFINE_integer(
    'checkpoint_interval', 5,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 1,
                     'Number of epochs between evaluation.')
flags.DEFINE_integer('train_epochs', 5, 'Number of training epochs.')

# Loss type.
flags.DEFINE_enum('loss_type', 'cross_entropy',
                  ['cross_entropy', 'focal_cross_entropy', 'mse', 'mae'],
                  'Type of loss function to use.')
flags.DEFINE_float('focal_loss_alpha', 0.1,
                   'Multiplicative factor used in the focal loss [1]-[2] to '
                   'downweight common cases.')
flags.DEFINE_float('focal_loss_gamma', 5.,
                   'Exponentiate factor used in the focal loss [1]-[2] to '
                   'push model to minimize in-confident examples.')


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

  # Create dataset builders.
  dataset_kwargs = dict(
      shuffle_buffer_size=data_buffer_size,
      tf_hub_preprocessor_url=FLAGS.bert_tokenizer_tf_hub_url)

  (train_dataset_builders, test_dataset_builders,
   train_split_name) = utils.make_train_and_test_dataset_builders(
       in_dataset_dir=FLAGS.in_dataset_dir,
       ood_dataset_dir=FLAGS.ood_dataset_dir,
       identity_dataset_dir=FLAGS.identity_dataset_dir,
       train_dataset_type=FLAGS.dataset_type,
       test_dataset_type='tfds',
       use_cross_validation=FLAGS.use_cross_validation,
       num_folds=FLAGS.num_folds,
       train_fold_ids=FLAGS.train_fold_ids,
       return_train_split_name=True,
       cv_split_name=FLAGS.train_cv_split_name,
       train_on_identity_subgroup_data=FLAGS.train_on_identity_subgroup_data,
       test_on_identity_subgroup_data=FLAGS.test_on_identity_subgroup_data,
       test_on_challenge_data=FLAGS.test_on_challenge_data,
       identity_type_dataset_dir=FLAGS.identity_type_dataset_dir,
       identity_specific_dataset_dir=FLAGS.identity_specific_dataset_dir,
       challenge_dataset_dir=FLAGS.challenge_dataset_dir,
       **dataset_kwargs)

  if FLAGS.prediction_mode:
    prediction_dataset_builders = utils.make_prediction_dataset_builders(
        add_identity_datasets=FLAGS.identity_prediction,
        identity_dataset_dir=FLAGS.identity_specific_dataset_dir,
        use_cross_validation=FLAGS.use_cross_validation,
        num_folds=FLAGS.num_folds,
        train_fold_ids=FLAGS.train_fold_ids,
        cv_split_name=FLAGS.test_cv_split_name,
        **dataset_kwargs)

    # Removes `cv_eval` since it overlaps with the `cv_eval_fold_*` datasets.
    test_dataset_builders.pop('cv_eval', None)
    test_dataset_builders.update(prediction_dataset_builders)

  class_weight = utils.create_class_weight(
      train_dataset_builders, test_dataset_builders)
  logging.info('class_weight: %s', str(class_weight))
  logging.info('train_split_name: %s', train_split_name)

  ds_info = test_dataset_builders['ind'].tfds_info
  # Positive and negative classes.
  num_classes = ds_info.metadata['num_classes']

  # Build datasets.
  train_datasets, test_datasets, dataset_steps_per_epoch, steps_per_eval = (
      utils.build_datasets(train_dataset_builders, test_dataset_builders,
                           batch_size, test_batch_size,
                           per_core_batch_size=FLAGS.per_core_batch_size))
  total_steps_per_epoch = sum(dataset_steps_per_epoch.values())
  logging.info('dataset_steps_per_epoch: %s', dataset_steps_per_epoch)
  logging.info('steps_per_eval: %s', steps_per_eval)

  if FLAGS.use_bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

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

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    if FLAGS.prediction_mode:
      eval_checkpoint_dir = FLAGS.eval_checkpoint_dir
      if FLAGS.checkpoint_name is not None:
        eval_checkpoint_dir = os.path.join(eval_checkpoint_dir,
                                           FLAGS.checkpoint_name)
      latest_checkpoint = tf.train.latest_checkpoint(eval_checkpoint_dir)
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

    metrics = utils.create_train_and_test_metrics(
        test_datasets,
        num_classes=num_classes,
        num_ece_bins=FLAGS.num_ece_bins,
        ece_label_threshold=FLAGS.ece_label_threshold,
        eval_collab_metrics=FLAGS.eval_collab_metrics,
        num_approx_bins=FLAGS.num_approx_bins)

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

      # Avoid directly modifying global variable `metrics` (which leads to an
      # assign-before-use error) by creating an update function instead.
      update_fn = utils.make_test_metrics_update_fn(
          dataset_name, sample_weight, labels,
          pred_labels, one_hot_labels, probs, auc_probs,
          ece_labels, ece_probs, calib_confidence,
          negative_log_likelihood, eval_time)
      update_fn(metrics)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def final_eval_step(iterator):
    """Final Evaluation StepFn to save prediction to directory."""

    def step_fn(inputs):
      ids = inputs['id']
      texts = inputs['features']
      text_ids = inputs['input_ids']
      bert_features, labels, additional_labels = utils.create_feature_and_label(
          inputs)
      logits = model(bert_features, training=False)
      return texts, text_ids, logits, labels, additional_labels, ids

    (per_replica_texts, per_replica_text_ids, per_replica_logits,
     per_replica_labels, per_replica_additional_labels, per_replica_ids) = (
         strategy.run(step_fn, args=(next(iterator),)))

    if strategy.num_replicas_in_sync > 1:
      texts_list = tf.concat(per_replica_texts.values, axis=0)
      text_ids_list = tf.concat(per_replica_text_ids.values, axis=0)
      logits_list = tf.concat(per_replica_logits.values, axis=0)
      labels_list = tf.concat(per_replica_labels.values, axis=0)
      ids_list = tf.concat(per_replica_ids.values, axis=0)
      additional_labels_dict = {}
      for additional_label in utils.IDENTITY_LABELS:
        if additional_label in per_replica_additional_labels:
          additional_labels_dict[additional_label] = tf.concat(
              per_replica_additional_labels[additional_label], axis=0)
    else:
      texts_list = per_replica_texts
      text_ids_list = per_replica_text_ids
      logits_list = per_replica_logits
      labels_list = per_replica_labels
      ids_list = per_replica_ids
      additional_labels_dict = {}
      for additional_label in utils.IDENTITY_LABELS:
        if additional_label in per_replica_additional_labels:
          additional_labels_dict[
              additional_label] = per_replica_additional_labels[
                  additional_label]

    return (texts_list, text_ids_list, logits_list, labels_list,
            additional_labels_dict, ids_list)

  if FLAGS.prediction_mode:
    # Prediction and exit.
    for dataset_name, test_dataset in test_datasets.items():
      test_iterator = iter(test_dataset)  # pytype: disable=wrong-arg-types
      message = 'Final eval on dataset {}'.format(dataset_name)
      logging.info(message)

      ids_all = []
      texts_all = []
      text_ids_all = []
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

            (text_step, text_ids_step, logits_step, labels_step,
             additional_labels_dict_step,
             ids_step) = final_eval_step(test_iterator)

            ids_all.append(ids_step)
            texts_all.append(text_step)
            text_ids_all.append(text_ids_step)
            logits_all.append(logits_step)
            labels_all.append(labels_step)
            if 'identity' in dataset_name:
              for identity_label_name in utils.IDENTITY_LABELS:
                additional_labels_all_dict[identity_label_name].append(
                    additional_labels_dict_step[identity_label_name])

      except (StopIteration, tf.errors.OutOfRangeError):
        tf.experimental.async_clear_error()
        logging.info('Done with eval on %s', dataset_name)

      ids_all = tf.concat(ids_all, axis=0)
      texts_all = tf.concat(texts_all, axis=0)
      text_ids_all = tf.concat(text_ids_all, axis=0)
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
          ids_all.numpy(),
          path=os.path.join(FLAGS.output_dir, 'ids_{}'.format(dataset_name)))
      utils.save_prediction(
          texts_all.numpy(),
          path=os.path.join(FLAGS.output_dir, 'texts_{}'.format(dataset_name)))
      utils.save_prediction(
          text_ids_all.numpy(),
          path=os.path.join(FLAGS.output_dir,
                            'text_ids_{}'.format(dataset_name)))
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
