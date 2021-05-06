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

# Lint as: python3
"""Multi-headed wide ResNet 28-10 on CIFAR-10 and CIFAR-100."""
import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_integer('ensemble_size', 3, 'Size of ensemble.')
flags.DEFINE_float('input_repetition_probability', 0.0,
                   'The probability that the inputs are identical for the'
                   'ensemble members.')
flags.DEFINE_integer('width_multiplier', 10, 'Integer to multiply the number of'
                     'typical filters by. "k" in ResNet-n-k.')
flags.DEFINE_integer('batch_repetitions', 4, 'Number of times an example is'
                     'repeated in a training batch. More repetitions lead to'
                     'lower variance gradients and increased training time.')
# Redefining default values
flags.FLAGS.set_default('corruptions_interval', 250)
flags.FLAGS.set_default('train_epochs', 250)
flags.FLAGS.set_default('l2', 3e-4)
flags.FLAGS.set_default('lr_decay_epochs', ['80', '160', '180'])
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  data_dir = utils.get_data_dir_from_flags(FLAGS)
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

  ds_info = tfds.builder(FLAGS.dataset).info
  train_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores // FLAGS.batch_repetitions
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  train_dataset_size = ds_info.splits['train'].num_examples
  steps_per_epoch = train_dataset_size // train_batch_size
  steps_per_eval = ds_info.splits['test'].num_examples // test_batch_size
  num_classes = ds_info.features['label'].num_classes

  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      validation_percent=1. - FLAGS.train_proportion)
  train_dataset = train_builder.load(batch_size=train_batch_size)
  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:
    validation_builder = ub.datasets.get(
        FLAGS.dataset,
        data_dir=data_dir,
        download_data=FLAGS.download_data,
        split=tfds.Split.VALIDATION,
        validation_percent=1. - FLAGS.train_proportion)
    validation_dataset = validation_builder.load(batch_size=test_batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // test_batch_size
  clean_test_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST)
  clean_test_dataset = clean_test_builder.load(batch_size=test_batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  steps_per_epoch = train_builder.num_examples // train_batch_size
  steps_per_eval = clean_test_builder.num_examples // test_batch_size
  num_classes = 100 if FLAGS.dataset == 'cifar100' else 10
  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar100':
      data_dir = FLAGS.cifar100_c_path
    corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
    for corruption_type in corruption_types:
      for severity in range(1, 6):
        dataset = ub.datasets.get(
            f'{FLAGS.dataset}_corrupted',
            corruption_type=corruption_type,
            data_dir=data_dir,
            severity=severity,
            split=tfds.Split.TEST).load(batch_size=test_batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
    model = ub.models.wide_resnet_mimo(
        input_shape=[FLAGS.ensemble_size] +
        list(ds_info.features['image'].shape),
        depth=28,
        width_multiplier=FLAGS.width_multiplier,
        num_classes=num_classes,
        ensemble_size=FLAGS.ensemble_size)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * train_batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        FLAGS.lr_decay_ratio,
        lr_decay_epochs,
        FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
    }
    eval_dataset_splits = ['test']
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
      })
      eval_dataset_splits += ['validation']
    for i in range(FLAGS.ensemble_size):
      for dataset_split in eval_dataset_splits:
        metrics[f'{dataset_split}/nll_member_{i}'] = tf.keras.metrics.Mean()
        metrics[f'{dataset_split}/accuracy_member_{i}'] = (
            tf.keras.metrics.SparseCategoricalAccuracy())
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, 6):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())
    test_diversity = {
        'test/disagreement': tf.keras.metrics.Mean(),
        'test/average_kl': tf.keras.metrics.Mean(),
        'test/cosine_similarity': tf.keras.metrics.Mean(),
    }

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      batch_size = tf.shape(images)[0]

      main_shuffle = tf.random.shuffle(tf.tile(
          tf.range(batch_size), [FLAGS.batch_repetitions]))
      to_shuffle = tf.cast(tf.cast(tf.shape(main_shuffle)[0], tf.float32)
                           * (1. - FLAGS.input_repetition_probability),
                           tf.int32)
      shuffle_indices = [
          tf.concat([tf.random.shuffle(main_shuffle[:to_shuffle]),
                     main_shuffle[to_shuffle:]], axis=0)
          for _ in range(FLAGS.ensemble_size)]
      images = tf.stack([tf.gather(images, indices, axis=0)
                         for indices in shuffle_indices], axis=1)
      labels = tf.stack([tf.gather(labels, indices, axis=0)
                         for indices in shuffle_indices], axis=1)

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True), axis=1))
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the BN parameters and bias terms.
          if ('kernel' in var.name or 'batch_norm' in var.name or
              'bias' in var.name):
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(tf.reshape(logits, [-1, num_classes]))
      flat_labels = tf.reshape(labels, [-1])
      metrics['train/ece'].add_batch(probs, label=flat_labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(flat_labels, probs)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(
          tf.expand_dims(images, 1), [1, FLAGS.ensemble_size, 1, 1, 1])
      logits = model(images, training=False)
      probs = tf.nn.softmax(logits)

      if dataset_name == 'clean':
        per_probs = tf.transpose(probs, perm=[1, 0, 2])
        diversity = rm.metrics.AveragePairwiseDiversity()
        diversity.add_batch(per_probs, num_models=FLAGS.ensemble_size)
        diversity_results = diversity.result()
        for k, v in diversity_results.items():
          test_diversity['test/' + k].update_state(v)

      for i in range(FLAGS.ensemble_size):
        member_probs = probs[:, i]
        member_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, member_probs)
        metrics[f'{dataset_split}/nll_member_{i}'].update_state(member_loss)
        metrics[f'{dataset_split}/accuracy_member_{i}'].update_state(
            labels, member_probs)

      # Negative log marginal likelihood computed in a numerically-stable way.
      labels_tiled = tf.tile(
          tf.expand_dims(labels, 1), [1, FLAGS.ensemble_size])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_tiled, logits, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[1]) +
          tf.math.log(float(FLAGS.ensemble_size)))
      probs = tf.math.reduce_mean(probs, axis=1)  # marginalize

      if dataset_name == 'clean':
        metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
        metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  train_iterator = iter(train_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    train_step(train_iterator)

    current_step = (epoch + 1) * steps_per_epoch
    max_steps = steps_per_epoch * (FLAGS.train_epochs)
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    if validation_dataset:
      validation_iterator = iter(validation_dataset)
      test_step(
          validation_iterator, 'validation', 'clean', steps_per_validation)
    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      logging.info('Starting to run eval at epoch: %s', epoch)
      test_start_time = time.time()
      test_step(test_iterator, 'test', dataset_name, steps_per_eval)
      ms_per_example = (time.time() - test_start_time) * 1e6 / test_batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)
      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    for i in range(FLAGS.ensemble_size):
      logging.info(
          'Member %d Test Loss: %.4f, Accuracy: %.2f%%', i,
          metrics['test/nll_member_{}'.format(i)].result(),
          metrics['test/accuracy_member_{}'.format(i)].result() * 100)

    metrics.update(test_diversity)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(corrupt_results)
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

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
        'batch_repetitions': FLAGS.batch_repetitions,
    })

if __name__ == '__main__':
  app.run(main)
