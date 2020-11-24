# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

"""BERT model with Monte Carlo dropout."""

import os
import time
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import uncertainty_baselines as ub
import bert_utils  # local file import
import uncertainty_metrics as um

# Data flags
flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the TFRecord datasets and the tokenizer for Clinc '
    'Intent Detection Data.')

# Checkpoint flags
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
flags.DEFINE_integer('num_dropout_samples', 10,
                     'Number of dropout samples to use for prediction.')

# Optimization and evaluation flags
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('eval_batch_size', 512, 'Batch size for CPU evaluation.')
flags.DEFINE_float(
    'base_learning_rate', 1e-4,
    'Base learning rate when total batch size is 128. It is '
    'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer(
    'checkpoint_interval', 150,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 5,
                     'Number of epochs between evaluation.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/clinc_intent', 'Output directory.')
flags.DEFINE_integer('train_epochs', 150, 'Number of training epochs.')
flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

if FLAGS.channel_wise_dropout_all:
  # Sets channel-wise dropout for all layer types.
  FLAGS.channel_wise_dropout_mha = True
  FLAGS.channel_wise_dropout_att = True
  FLAGS.channel_wise_dropout_ffn = True


def resolve_bert_ckpt_and_config_dir(bert_dir, bert_config_dir, bert_ckpt_dir):
  """Resolves BERT checkpoint and config file directories."""

  missing_ckpt_or_config_dir = not (bert_ckpt_dir and bert_config_dir)
  if missing_ckpt_or_config_dir:
    if not bert_dir:
      raise ValueError('bert_dir cannot be empty.')

    if not bert_config_dir:
      bert_config_dir = os.path.join(bert_dir, 'bert_config.json')

    if not bert_ckpt_dir:
      bert_ckpt_dir = os.path.join(bert_dir, 'bert_model.ckpt')
  return bert_config_dir, bert_ckpt_dir


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
  train_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='train',
      data_dir=FLAGS.data_dir,
      data_mode='ind')
  ind_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='ind')
  ood_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='ood')
  all_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='all')

  dataset_builders = {
      'clean': ind_dataset_builder,
      'ood': ood_dataset_builder,
      'all': all_dataset_builder
  }

  train_dataset = train_dataset_builder.load(batch_size=batch_size)

  ds_info = train_dataset_builder.tfds_info
  feature_size = ds_info.metadata['feature_size']
  # num_classes is number of valid intents plus out-of-scope intent
  num_classes = ds_info.features['intent_label'].num_classes + 1

  steps_per_epoch = train_dataset_builder.num_examples // batch_size

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=FLAGS.eval_batch_size)
    steps_per_eval[dataset_name] = (
        dataset_builder.num_examples // FLAGS.eval_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building BERT model')

    bert_config_dir, bert_ckpt_dir = resolve_bert_ckpt_and_config_dir(
        FLAGS.bert_dir, FLAGS.bert_config_dir, FLAGS.bert_ckpt_dir)
    bert_config = bert_utils.create_config(bert_config_dir)
    model, bert_encoder = ub.models.DropoutBertBuilder(
        num_classes=num_classes,
        bert_config=bert_config,
        use_mc_dropout_mha=FLAGS.use_mc_dropout_mha,
        use_mc_dropout_att=FLAGS.use_mc_dropout_att,
        use_mc_dropout_ffn=FLAGS.use_mc_dropout_ffn,
        use_mc_dropout_output=FLAGS.use_mc_dropout_output,
        channel_wise_dropout_mha=FLAGS.channel_wise_dropout_mha,
        channel_wise_dropout_att=FLAGS.channel_wise_dropout_att,
        channel_wise_dropout_ffn=FLAGS.channel_wise_dropout_ffn)
    optimizer = bert_utils.create_optimizer(
        FLAGS.base_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        warmup_proportion=FLAGS.warmup_proportion)

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
    }

    for dataset_name, test_dataset in test_datasets.items():
      if dataset_name != 'clean':
        metrics.update({
            'test/nll_{}'.format(dataset_name):
                tf.keras.metrics.Mean(),
            'test/accuracy_{}'.format(dataset_name):
                tf.keras.metrics.SparseCategoricalAccuracy(),
            'test/ece_{}'.format(dataset_name):
                um.ExpectedCalibrationError(num_bins=FLAGS.num_bins)
        })

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
    else:
      # load BERT from initial checkpoint
      bert_checkpoint = tf.train.Checkpoint(model=bert_encoder)
      bert_checkpoint.restore(
          bert_ckpt_dir).assert_existing_objects_matched()
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

  # Finally, define OOD metrics outside the accelerator scope for CPU eval.
  metrics.update({
      'test/auroc_all': tf.keras.metrics.AUC(curve='ROC'),
      'test/auprc_all': tf.keras.metrics.AUC(curve='PR')
  })

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels = bert_utils.create_feature_and_label(
          inputs, feature_size)

      with tf.GradientTape() as tape:
        # Set learning phase to enable dropout etc during training.
        logits = model(features, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels = bert_utils.create_feature_and_label(
          inputs, feature_size)

      # Compute ensemble prediction over Monte Carlo dropout samples.
      logits_list = []
      for _ in range(FLAGS.num_dropout_samples):
        logits = model(features, training=False)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        logits_list.append(logits)

      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_dropout_samples, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_dropout_samples)))

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        metrics['test/ece_{}'.format(dataset_name)].update_state(labels, probs)

      if dataset_name == 'all':
        ood_labels = tf.cast(labels == 150, labels.dtype)
        ood_probs = 1. - tf.reduce_max(probs, axis=-1)
        metrics['test/auroc_{}'.format(dataset_name)].update_state(
            ood_labels, ood_probs)
        metrics['test/auprc_{}'.format(dataset_name)].update_state(
            ood_labels, ood_probs)

    step_fn(next(iterator))

  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      train_step(train_iterator)

      current_step = epoch * steps_per_epoch + (step + 1)
      max_steps = steps_per_epoch * FLAGS.train_epochs
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                 'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                     current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                     steps_per_sec, eta_seconds / 60, time_elapsed / 60))
      if step % 20 == 0:
        logging.info(message)

    if epoch % FLAGS.evaluation_interval == 0:
      for dataset_name, test_dataset in test_datasets.items():
        test_iterator = iter(test_dataset)
        logging.info('Testing on dataset %s', dataset_name)
        for step in range(steps_per_eval[dataset_name]):
          if step % 20 == 0:
            logging.info('Starting to run eval step %s of epoch: %s', step,
                         epoch)
          test_step(test_iterator, dataset_name)
        logging.info('Done with testing on %s', dataset_name)

      logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                   metrics['train/loss'].result(),
                   metrics['train/accuracy'].result() * 100)
      logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                   metrics['test/negative_log_likelihood'].result(),
                   metrics['test/accuracy'].result() * 100)
      total_results = {
          name: metric.result() for name, metric in metrics.items()
      }
      with summary_writer.as_default():
        for name, result in total_results.items():
          tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)


if __name__ == '__main__':
  app.run(main)
