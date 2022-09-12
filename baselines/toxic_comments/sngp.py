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

"""Bidirectional encoder representations from transformers (BERT) with SNGP.

Spectral-normalized neural Gaussian process (SNGP) [1] is a simple method to
improve a deterministic neural network's uncertainty. It simply applies spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## Note:

Different from the paper, this implementation computes the posterior using the
Laplace approximation based on the Gaussian likelihood (i.e., squared loss)
rather than that based on cross-entropy loss. As a result, the logits for all
classes share the same covariance. In the experiments, this approach is shown to
perform better and computationally more scalable when the number of output
classes are large.

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
[3]: Tsung-Yi Lin et al. Focal Loss for Dense Object Detection. In
      _International Conference on Computer Vision_, 2018.
     https://arxiv.org/abs/1708.02002.
[4]: Jishnu Mukhoti et al. Calibrating Deep Neural Networks using Focal Loss.
     _arXiv preprint arXiv:2002.09437_, 2020.
     https://arxiv.org/abs/2002.09437
"""

import os
import time
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import tensorflow as tf
from tensorflow_addons import losses as tfa_losses

import uncertainty_baselines as ub
import utils  # local file import from baselines.toxic_comments
from tensorboard.plugins.hparams import api as hp

# Normalization flags.
flags.DEFINE_bool(
    'use_layer_norm_att', True,
    'Whether to apply layer normalization to the self-attention layers.')
flags.DEFINE_bool(
    'use_layer_norm_ffn', True,
    'Whether to apply layer normalization to the feedforward layers.')
flags.DEFINE_bool(
    'use_spec_norm_att', False,
    'Whether to apply spectral normalization to the self-attention layers.')
flags.DEFINE_bool(
    'use_spec_norm_ffn', False,
    'Whether to apply spectral normalization to the feedforward layers.')
flags.DEFINE_bool(
    'use_spec_norm_plr', True,
    'Whether to apply spectral normalization to the final CLS pooler layer.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', .95,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', True,
    'Whether to normalize the input using LayerNorm for GP layer.'
    'This is similar to automatic relevance determination (ARD) in the classic '
    'GP learning.')
flags.DEFINE_float('gp_cov_ridge_penalty', 1e-3,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', 0.999,
    'The discount factor to compute the moving average of precision matrix.')
flags.DEFINE_float(
    'gp_mean_field_factor', 1e-1,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')

# Optimization flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 32, 'Batch size per TPU core/GPU.')
flags.DEFINE_float(
    'base_learning_rate', 2.5e-5,
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
flags.DEFINE_integer(
    'num_mc_samples', 1,
    'Number of Monte Carlo forward passes to collect for ensemble prediction. '
    'Currently can only be 1 since the model is deterministic.')

# Loss type.
flags.DEFINE_enum('loss_type', 'cross_entropy',
                  ['cross_entropy', 'focal_cross_entropy', 'mse', 'mae'],
                  'Type of loss function to use.')
flags.DEFINE_float(
    'focal_loss_alpha', 0.1,
    'Multiplicative factor used in the focal loss [3]-[4] to '
    'upweight inconfident examples.')
flags.DEFINE_float(
    'focal_loss_gamma', 1.,
    'Exponentiate factor used in the focal loss [3]-[4] to '
    'upweight inconfident examples.')


FLAGS = flags.FLAGS


_MAX_SEQ_LENGTH = 512


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
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
  test_batch_size = FLAGS.per_core_batch_size
  data_buffer_size = batch_size * 10

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
       train_on_multi_task_label=FLAGS.train_on_multi_task_label,
       multi_task_label_threshold=FLAGS.multi_task_label_threshold,
       test_on_challenge_data=FLAGS.test_on_challenge_data,
       identity_type_dataset_dir=FLAGS.identity_type_dataset_dir,
       identity_specific_dataset_dir=FLAGS.identity_specific_dataset_dir,
       challenge_dataset_dir=FLAGS.challenge_dataset_dir,
       **dataset_kwargs)

  if FLAGS.prediction_mode:
    prediction_dataset_builders = utils.make_prediction_dataset_builders(
        add_identity_datasets=FLAGS.identity_prediction,
        identity_dataset_dir=FLAGS.identity_specific_dataset_dir,
        add_cross_validation_datasets=FLAGS.use_cross_validation,
        cv_dataset_dir=FLAGS.in_dataset_dir,
        cv_dataset_type=FLAGS.dataset_type,
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
    logging.info('Building BERT %s model', FLAGS.bert_model_type)
    logging.info('use_gp_layer=%s', FLAGS.use_gp_layer)
    logging.info('use_spec_norm_att=%s', FLAGS.use_spec_norm_att)
    logging.info('use_spec_norm_ffn=%s', FLAGS.use_spec_norm_ffn)
    logging.info('use_layer_norm_att=%s', FLAGS.use_layer_norm_att)
    logging.info('use_layer_norm_ffn=%s', FLAGS.use_layer_norm_ffn)

    bert_config_dir, bert_ckpt_dir = utils.resolve_bert_ckpt_and_config_dir(
        FLAGS.bert_model_type, FLAGS.bert_dir, FLAGS.bert_config_dir,
        FLAGS.bert_ckpt_dir)
    bert_config = utils.create_config(bert_config_dir)

    gp_layer_kwargs = dict(
        num_inducing=FLAGS.gp_hidden_dim,
        gp_kernel_scale=FLAGS.gp_scale,
        gp_output_bias=FLAGS.gp_bias,
        normalize_input=FLAGS.gp_input_normalization,
        gp_cov_momentum=FLAGS.gp_cov_discount_factor,
        gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty)
    spec_norm_kwargs = dict(
        iteration=FLAGS.spec_norm_iteration,
        norm_multiplier=FLAGS.spec_norm_bound)

    model, bert_encoder = ub.models.bert_sngp_model(
        num_classes=num_classes,
        bert_config=bert_config,
        gp_layer_kwargs=gp_layer_kwargs,
        spec_norm_kwargs=spec_norm_kwargs,
        num_heads=2 if FLAGS.train_on_multi_task_label else 1,
        use_gp_layer=FLAGS.use_gp_layer,
        use_spec_norm_att=FLAGS.use_spec_norm_att,
        use_spec_norm_ffn=FLAGS.use_spec_norm_ffn,
        use_layer_norm_att=FLAGS.use_layer_norm_att,
        use_layer_norm_ffn=FLAGS.use_layer_norm_ffn,
        use_spec_norm_plr=FLAGS.use_spec_norm_plr)
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
    else:
      # load BERT from initial checkpoint
      bert_encoder, _, _ = utils.load_bert_weight_from_ckpt(
          bert_model=bert_encoder,
          bert_ckpt_dir=bert_ckpt_dir,
          repl_patterns=ub.models.bert_sngp.CHECKPOINT_REPL_PATTERNS)
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

    metrics = utils.create_train_and_test_metrics(
        test_datasets,
        num_classes=num_classes,
        num_ece_bins=FLAGS.num_ece_bins,
        ece_label_threshold=FLAGS.ece_label_threshold,
        eval_collab_metrics=FLAGS.eval_collab_metrics,
        num_approx_bins=FLAGS.num_approx_bins,
        train_on_multi_task_label=FLAGS.train_on_multi_task_label)

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

        if FLAGS.train_on_multi_task_label:
          logits, multi_task_logits = logits

        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
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
              labels,
              loss_logits,
              alpha=FLAGS.focal_loss_alpha,
              gamma=FLAGS.focal_loss_gamma,
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

        if FLAGS.train_on_multi_task_label and FLAGS.multi_task_loss_weight > 0:
          multi_task_logits = tf.squeeze(multi_task_logits, axis=1)
          multi_task_labels = inputs['multi_task_labels']
          multi_task_loss = tf.nn.sigmoid_cross_entropy_with_logits(
              multi_task_labels, multi_task_logits)
          loss += FLAGS.multi_task_loss_weight * multi_task_loss

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
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels, _ = utils.create_feature_and_label(inputs)

      eval_start_time = time.time()
      # Compute ensemble prediction over Monte Carlo forward-pass samples.
      logits_list = []
      stddev_list = []
      for _ in range(FLAGS.num_mc_samples):
        logits = model(features, training=False)

        multi_task_probs, multi_task_labels = None, None
        if FLAGS.train_on_multi_task_label:
          logits, multi_task_logits = logits
          multi_task_probs = tf.nn.sigmoid(multi_task_logits)
          multi_task_labels = inputs.get('multi_task_labels', None)

        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract both.
          logits, covmat = logits
        else:
          covmat = tf.eye(test_batch_size)

        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
          covmat = tf.cast(covmat, tf.float32)

        logits = ed.layers.utils.mean_field_logits(
            logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
        stddev = tf.sqrt(tf.linalg.diag_part(covmat))

        logits_list.append(logits)
        stddev_list.append(stddev)

      eval_time = (time.time() - eval_start_time) / FLAGS.per_core_batch_size
      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      stddev_list = tf.stack(stddev_list, axis=0)

      stddev = tf.reduce_mean(stddev_list, axis=0)
      probs_list = tf.nn.sigmoid(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      ce = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.broadcast_to(
              labels,
              [FLAGS.num_mc_samples, tf.shape(labels)[0]]),
          logits=tf.squeeze(logits_list, axis=-1))
      negative_log_likelihood = -tf.reduce_logsumexp(
          -ce, axis=0) + tf.math.log(float(FLAGS.num_mc_samples))
      negative_log_likelihood = tf.reduce_mean(negative_log_likelihood)

      sample_weight = generate_sample_weight(
          labels, class_weight['test/{}'.format(dataset_name)],
          FLAGS.ece_label_threshold)

      # Avoid directly modifying global variable `metrics` (which leads to an
      # assign-before-use error) by creating an update function instead.
      update_fn = utils.make_test_metrics_update_fn(
          dataset_name,
          sample_weight,
          num_classes,
          labels,
          probs,
          multi_task_labels,
          multi_task_probs,
          negative_log_likelihood,
          eval_time=eval_time,
          ece_label_threshold=FLAGS.ece_label_threshold,
          train_on_multi_task_label=FLAGS.train_on_multi_task_label,
          eval_collab_metrics=FLAGS.eval_collab_metrics)
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

      if FLAGS.train_on_multi_task_label:
        logits, multi_task_logits = logits
        # Not recording bias prediction for now.
        del multi_task_logits

      if isinstance(logits, (list, tuple)):
        # If model returns a tuple of (logits, covmat), extract both.
        logits, covmat = logits
      else:
        covmat = tf.eye(test_batch_size)

      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
        covmat = tf.cast(covmat, tf.float32)

      logits = ed.layers.utils.mean_field_logits(
          logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
      variances = tf.linalg.diag_part(covmat)
      return texts, text_ids, logits, variances, labels, additional_labels, ids

    (per_replica_texts, per_replica_text_ids, per_replica_logits,
     per_replica_variances, per_replica_labels, per_replica_additional_labels,
     per_replica_ids) = (
         strategy.run(step_fn, args=(next(iterator),)))

    if strategy.num_replicas_in_sync > 1:
      texts_list = tf.concat(per_replica_texts.values, axis=0)
      text_ids_list = tf.concat(per_replica_text_ids.values, axis=0)
      logits_list = tf.concat(per_replica_logits.values, axis=0)
      variances_list = tf.concat(per_replica_variances.values, axis=0)
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
      variances_list = per_replica_variances
      labels_list = per_replica_labels
      ids_list = per_replica_ids
      additional_labels_dict = {}
      for additional_label in utils.IDENTITY_LABELS:
        if additional_label in per_replica_additional_labels:
          additional_labels_dict[
              additional_label] = per_replica_additional_labels[
                  additional_label]

    return (texts_list, text_ids_list, logits_list, variances_list, labels_list,
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
      variances_all = []
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

            (text_step, text_ids_step, logits_step, variances_step, labels_step,
             additional_labels_dict_step,
             ids_step) = final_eval_step(test_iterator)

            ids_all.append(ids_step)
            texts_all.append(text_step)
            text_ids_all.append(text_ids_step)
            logits_all.append(logits_step)
            variances_all.append(variances_step)
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
      variances_all = tf.concat(variances_all, axis=0)
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
      utils.save_prediction(
          variances_all.numpy(),
          path=os.path.join(FLAGS.output_dir,
                            'variances_{}'.format(dataset_name)))

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
        try:
          with tf.experimental.async_scope():
            train_step(
                train_iterator,
                dataset_name,
                dataset_steps_per_epoch[dataset_name])

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
                           FLAGS.train_epochs, steps_per_sec,
                           eta_seconds / 60, time_elapsed / 60))
            logging.info(message)

        except (StopIteration, tf.errors.OutOfRangeError):
          tf.experimental.async_clear_error()
          logging.info('Done with testing on %s', dataset_name)

      if epoch % FLAGS.evaluation_interval == 0:
        for dataset_name, test_dataset in test_datasets.items():
          test_iterator = iter(test_dataset)
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

        logging.info('Train Loss: %.4f, ECE: %.2f, Accuracy: %.2f',
                     metrics['train/loss'].result(),
                     metrics['train/ece'].result()['ece'],
                     metrics['train/accuracy'].result())
        logging.info('Test NLL: %.4f, AUROC: %.4f',
                     metrics['test/negative_log_likelihood'].result(),
                     metrics['test/auroc'].result())

        # record results
        total_results = {}
        for name, metric in metrics.items():
          try:
            total_results[name] = metric.result()
          except tf.errors.InvalidArgumentError:
            logging.info('Error for metric "%s". Recording 0.', name)
            total_results[name] = 0

        # Metrics from Robustness Metrics (like ECE) will return a dict with a
        # single key/value, instead of a scalar.
        total_results = {
            k: (list(v.values())[0] if isinstance(v, dict) else v)
            for k, v in total_results.items()
        }
        with summary_writer.as_default():
          for name, result in total_results.items():
            tf.summary.scalar(name, result, step=epoch + 1)

      for metric in metrics.values():
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
        'gp_mean_field_factor': FLAGS.gp_mean_field_factor,
    })


if __name__ == '__main__':
  app.run(main)
