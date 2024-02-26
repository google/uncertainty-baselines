# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Wide ResNet 28-10 with Monte Carlo dropout on CIFAR-10."""

import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import ood_utils  # local file import from baselines.cifar
import utils  # local file import from baselines.cifar
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_integer('num_dropout_samples_training', 1,
                     'Number of dropout samples for training.')
flags.DEFINE_bool('filterwise_dropout', False, 'Dropout whole convolutional'
                  'filters instead of individual values in the feature map.')
flags.DEFINE_bool('residual_dropout', True,
                  'Apply dropout only to the residual connections as proposed'
                  'in the original paper.'
                  'Otherwise dropout is applied after every layer.')

# OOD flags.
flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', False,
                  'Whether to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
                  'list of OOD datasets to evaluate on.')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
                  'Wheter to use DempsterShafer Uncertainty score.')

# Redefining default values
flags.FLAGS.set_default('base_learning_rate', 0.05)
flags.FLAGS.set_default('l2', 3e-4)
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  data_dir = FLAGS.data_dir
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
  batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores
                // FLAGS.num_dropout_samples_training)
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
  steps_per_eval = ds_info.splits['test'].num_examples // test_batch_size
  num_classes = ds_info.features['label'].num_classes

  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      validation_percent=1. - FLAGS.train_proportion)
  train_dataset = train_builder.load(batch_size=batch_size)
  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:
    validation_builder = ub.datasets.get(
        FLAGS.dataset,
        data_dir=data_dir,
        split=tfds.Split.VALIDATION,
        validation_percent=1. - FLAGS.train_proportion,
        drop_remainder=FLAGS.drop_remainder_for_eval)
    validation_dataset = validation_builder.load(batch_size=test_batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // test_batch_size
  clean_test_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      split=tfds.Split.TEST,
      drop_remainder=FLAGS.drop_remainder_for_eval)
  clean_test_dataset = clean_test_builder.load(batch_size=test_batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_builder.num_examples // batch_size
  num_classes = 100 if FLAGS.dataset == 'cifar100' else 10

  if FLAGS.eval_on_ood:
    ood_dataset_names = FLAGS.ood_dataset
    ood_ds, steps_per_ood = ood_utils.load_ood_datasets(
        ood_dataset_names,
        clean_test_builder,
        1. - FLAGS.train_proportion,
        batch_size,
        drop_remainder=FLAGS.drop_remainder_for_eval)
    ood_datasets = {
        name: strategy.experimental_distribute_dataset(ds)
        for name, ds in ood_ds.items()
    }
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
            split=tfds.Split.TEST,
            drop_remainder=FLAGS.drop_remainder_for_eval).load(
                batch_size=test_batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')
    model = ub.models.wide_resnet_dropout(
        input_shape=(32, 32, 3),
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        l2=FLAGS.l2,
        dropout_rate=FLAGS.dropout_rate,
        residual_dropout=FLAGS.residual_dropout,
        filterwise_dropout=FLAGS.filterwise_dropout)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)
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
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
      })
    if FLAGS.eval_on_ood:
      ood_metrics = ood_utils.create_ood_metrics(ood_dataset_names)
      metrics.update(ood_metrics)
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
      images = tf.tile(images, [FLAGS.num_dropout_samples_training, 1, 1, 1])
      labels = tf.tile(labels, [FLAGS.num_dropout_samples_training])
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      logits_list = []
      for _ in range(FLAGS.num_dropout_samples):
        logits = model(images, training=False)
        logits_list.append(logits)

      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_dropout_samples, tf.shape(labels)[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_dropout_samples)))

      if dataset_name == 'clean':
        metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
        metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
      elif dataset_name.startswith('ood/'):
        ood_labels = 1 - inputs['is_in_distribution']
        if FLAGS.dempster_shafer_ood:
          ood_scores = ood_utils.DempsterShaferUncertainty(logits)
        else:
          ood_scores = 1 - tf.reduce_max(probs, axis=-1)

        for name, metric in metrics.items():
          if dataset_name in name:
            metric.update_state(ood_labels, ood_scores)
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
    max_steps = steps_per_epoch * FLAGS.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps,
                   epoch + 1,
                   FLAGS.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
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
      ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    if FLAGS.eval_on_ood:
      for ood_dataset_name, ood_dataset in ood_datasets.items():
        ood_iterator = iter(ood_dataset)
        logging.info('Calculating OOD on dataset %s', ood_dataset_name)
        logging.info('Running OOD eval at epoch: %s', epoch)
        test_step(ood_iterator, 'test', ood_dataset_name,
                  steps_per_ood[ood_dataset_name])

        logging.info('Done with OOD eval on %s', dataset_name)

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

    if FLAGS.corruptions_interval > 0:
      for metric in corrupt_metrics.values():
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
        'dropout_rate': FLAGS.dropout_rate,
        'num_dropout_samples': FLAGS.num_dropout_samples,
        'num_dropout_samples_training': FLAGS.num_dropout_samples_training,
    })


if __name__ == '__main__':
  app.run(main)
