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

"""ResNet50 with Monte Carlo Dropout (Gal and Ghahramani 2016) on Kaggle's Diabetic Retinopathy Detection dataset."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import tensorflow_datasets as tfds

import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/dropout',
    'The directory where the model weights and training/evaluation summaries are '
    'stored. If you aim to use these as trained models for dropoutensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')

# Learning Rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 4e-4, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60'],
                  'Epochs at which we decay learning rate.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('l2', 5e-5, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('train_batch_size', 16,
                     'The per-core training batch size.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The per-core validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Dropout-related flags.
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate, between [0.0, 1.0).')
flags.DEFINE_integer('num_dropout_samples_eval', 10,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_bool(
    'filterwise_dropout', False,
    'Dropout whole convolutional filters instead of '
    'individual values in the feature map.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  # Initialize distribution strategy on flag-specified accelerator
  strategy = utils.init_distribution_strategy(FLAGS.force_use_cpu,
                                              FLAGS.use_gpu, FLAGS.tpu)
  use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  train_batch_size = (FLAGS.train_batch_size * FLAGS.num_cores)

  if use_tpu:
    logging.info(
        'Due to TPU requiring static shapes, we must fix the eval batch size '
        'to the train batch size: %d/', train_batch_size)
    eval_batch_size = train_batch_size
  else:
    eval_batch_size = (FLAGS.eval_batch_size *
                       FLAGS.num_cores) // FLAGS.num_dropout_samples_eval

  # As per the Kaggle challenge, we have split sizes:
  # train: 35,126
  # validation: 10,906 (currently unused)
  # test: 42,670
  ds_info = tfds.builder('diabetic_retinopathy_detection').info
  steps_per_epoch = ds_info.splits['train'].num_examples // train_batch_size
  steps_per_validation_eval = (
      ds_info.splits['validation'].num_examples // eval_batch_size)
  steps_per_test_eval = ds_info.splits['test'].num_examples // eval_batch_size

  data_dir = FLAGS.data_dir

  dataset_train_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='train', data_dir=data_dir)
  dataset_train = dataset_train_builder.load(batch_size=train_batch_size)
  dataset_train = strategy.experimental_distribute_dataset(dataset_train)

  dataset_validation_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='validation', data_dir=data_dir)
  dataset_validation = dataset_validation_builder.load(
      batch_size=eval_batch_size)
  dataset_validation = strategy.experimental_distribute_dataset(
      dataset_validation)

  dataset_test_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='test', data_dir=data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
  dataset_test = strategy.experimental_distribute_dataset(dataset_test)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 MC Dropout model.')
    model = ub.models.resnet50_dropout(
        input_shape=utils.load_input_shape(dataset_train),
        num_classes=1,  # binary classification task
        dropout_rate=FLAGS.dropout_rate,
        filterwise_dropout=FLAGS.filterwise_dropout)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate
    lr_decay_epochs = [
        (int(start_epoch_str) * FLAGS.train_epochs) // DEFAULT_NUM_EPOCHS
        for start_epoch_str in FLAGS.lr_decay_epochs
    ]

    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=use_tpu, num_bins=FLAGS.num_bins)
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope()
      # so that optimizer slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  # Finally, define OOD metrics outside the accelerator scope for CPU eval.
  # This will cause an error on TPU.
  if not use_tpu:
    metrics.update({
        'train/auc': tf.keras.metrics.AUC(),
        'validation/auc': tf.keras.metrics.AUC(),
        'test/auc': tf.keras.metrics.AUC()
    })

  @tf.function
  def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true=tf.expand_dims(labels, axis=-1),
                y_pred=logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      probs = tf.nn.sigmoid(logits)

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/auc'].update_state(labels, probs)

      if not use_tpu:
        metrics['train/ece'].update_state(labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split):
    """Evaluation step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = tf.convert_to_tensor(inputs['labels'])

      logits_list = []
      for _ in range(FLAGS.num_dropout_samples_eval):
        logits = model(images, training=False)
        logits = tf.squeeze(logits, axis=-1)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        logits_list.append(logits)

      # Logits dimension is (num_samples, batch_size).
      logits_list = tf.stack(logits_list, axis=0)
      probs_list = tf.nn.sigmoid(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)
      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_dropout_samples_eval, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.binary_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_dropout_samples_eval)))
      metrics[dataset_split + '/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics[dataset_split + '/accuracy'].update_state(labels, probs)
      metrics[dataset_split + '/auc'].update_state(labels, probs)

      if not use_tpu:
        metrics[dataset_split + '/ece'].update_state(labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  start_time = time.time()

  train_iterator = iter(dataset_train)
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch + 1)
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

    validation_iterator = iter(dataset_validation)
    for step in range(steps_per_validation_eval):
      if step % 20 == 0:
        logging.info('Starting to run validation eval step %s of epoch: %s',
                     step, epoch + 1)
      test_step(validation_iterator, 'validation')

    test_iterator = iter(dataset_test)
    for step in range(steps_per_test_eval):
      if step % 20 == 0:
        logging.info('Starting to run test eval step %s of epoch: %s', step,
                     epoch + 1)
      test_start_time = time.time()
      test_step(test_iterator, 'test')
      ms_per_example = (time.time() - test_start_time) * 1e6 / eval_batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)

    # Log and write to summary the epoch metrics
    utils.log_epoch_metrics(metrics=metrics, use_tpu=use_tpu)
    total_results = {name: metric.result() for name, metric in metrics.items()}
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

      # TODO(nband): debug checkpointing
      # Also save Keras model, due to checkpoint.save issue
      keras_model_name = os.path.join(FLAGS.output_dir,
                                      f'keras_model_{epoch + 1}')
      model.save(keras_model_name)
      logging.info('Saved keras model to %s', keras_model_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)

  keras_model_name = os.path.join(FLAGS.output_dir,
                                  f'keras_model_{FLAGS.train_epochs}')
  model.save(keras_model_name)
  logging.info('Saved keras model to %s', keras_model_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'dropout_rate': FLAGS.dropout_rate,
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
