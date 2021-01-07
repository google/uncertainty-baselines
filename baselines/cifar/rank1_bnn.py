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

"""Rank-1 BNN Wide ResNet 28-10 on CIFAR-10 and CIFAR-100.

A Rank-1 Bayesian neural net (Rank-1 BNN) [1] is an efficient and scalable
approach to variational BNNs that posits prior distributions on rank-1 factors
of the weights and optimizes global mixture variational posterior distributions.

References:

  [1]: Michael W. Dusenberry*, Ghassen Jerfel*, Yeming Wen, Yian Ma, Jasper
       Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran. Efficient
       and Scalable Bayesian Neural Nets with Rank-1 Factors. In Proc. of
       International Conference on Machine Learning (ICML) 2020.
       https://arxiv.org/abs/2005.07186
"""
import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
import uncertainty_metrics as um

flags.DEFINE_integer('kl_annealing_epochs', 200,
                     'Number of epoch over which to anneal the KL term to 1.')
flags.DEFINE_string('alpha_initializer', 'trainable_normal',
                    'Initializer name for the alpha parameters.')
flags.DEFINE_string('gamma_initializer', 'trainable_normal',
                    'Initializer name for the gamma parameters.')
flags.DEFINE_string('alpha_regularizer', 'normal_kl_divergence',
                    'Regularizer name for the alpha parameters.')
flags.DEFINE_string('gamma_regularizer', 'normal_kl_divergence',
                    'Regularizer name for the gamma parameters.')
flags.DEFINE_boolean('use_additive_perturbation', False,
                     'Use additive perturbations instead of multiplicative.')
flags.DEFINE_float('dropout_rate', 1e-3,
                   'Dropout rate. Only used if alpha/gamma initializers are, '
                   'e.g., trainable normal.')
flags.DEFINE_float('prior_mean', 1., 'Prior mean.')
flags.DEFINE_float('prior_stddev', 0.1,
                   'Prior stddev. Sort of like a prior on dropout rate, where '
                   'it encourages defaulting/shrinking to this value.')

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 64,
                     'Batch size per TPU core/GPU. The number of new '
                     'datapoints gathered per batch is this number divided by '
                     'ensemble_size (we tile the batch by that # of times).')
flags.DEFINE_float('random_sign_init', 0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('fast_weight_lr_multiplier', 1.0,
                   'fast weights lr multiplier.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when total training batch size is 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['80', '160', '180'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
# TODO(ghassen): consider adding CIFAR-100-C to TFDS.
flags.DEFINE_string('cifar100_c_path',
                    '',
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', -1,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

flags.DEFINE_integer('num_eval_samples', 1,
                     'Number of model predictions to sample per example at '
                     'eval time.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS


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

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = ((FLAGS.per_core_batch_size // FLAGS.ensemble_size) *
                FLAGS.num_cores)
  train_dataset_size = ds_info.splits['train'].num_examples
  steps_per_epoch = train_dataset_size // batch_size
  test_dataset_size = ds_info.splits['test'].num_examples
  steps_per_eval = test_dataset_size // batch_size
  num_classes = ds_info.features['label'].num_classes

  train_dataset = utils.load_dataset(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  clean_test_dataset = utils.load_dataset(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar10':
      load_c_dataset = utils.load_cifar10_c
    else:
      load_c_dataset = functools.partial(utils.load_cifar100_c,
                                         path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset = load_c_dataset(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
    model = ub.models.wide_resnet_rank1(
        input_shape=ds_info.features['image'].shape,
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        alpha_initializer=FLAGS.alpha_initializer,
        gamma_initializer=FLAGS.gamma_initializer,
        alpha_regularizer=FLAGS.alpha_regularizer,
        gamma_regularizer=FLAGS.gamma_regularizer,
        use_additive_perturbation=FLAGS.use_additive_perturbation,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        dropout_rate=FLAGS.dropout_rate,
        prior_mean=FLAGS.prior_mean,
        prior_stddev=FLAGS.prior_stddev)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [(int(start_epoch_str) * FLAGS.train_epochs) // 200
                       for start_epoch_str in FLAGS.lr_decay_epochs]
    lr_schedule = utils.LearningRateSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/kl': tf.keras.metrics.Mean(),
        'train/kl_scale': tf.keras.metrics.Mean(),
        'train/elbo': tf.keras.metrics.Mean(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/kl': tf.keras.metrics.Mean(),
        'test/elbo': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
    }
    if FLAGS.ensemble_size > 1:
      for i in range(FLAGS.ensemble_size):
        metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
        metrics['test/accuracy_member_{}'.format(i)] = (
            tf.keras.metrics.SparseCategoricalAccuracy())
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/kl_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/elbo_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  def compute_l2_loss(model):
    filtered_variables = []
    for var in model.trainable_variables:
      # Apply l2 on the BN parameters and bias terms. This
      # excludes only fast weight approximate posterior/prior parameters,
      # but pay caution to their naming scheme.
      if ('kernel' in var.name or
          'batch_norm' in var.name or
          'bias' in var.name):
        filtered_variables.append(tf.reshape(var, (-1,)))
    l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
        tf.concat(filtered_variables, axis=0))
    return l2_loss

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      if FLAGS.ensemble_size > 1:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
        labels = tf.tile(labels, [FLAGS.ensemble_size])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        l2_loss = compute_l2_loss(model)
        kl = sum(model.losses) / train_dataset_size
        kl_scale = tf.cast(optimizer.iterations + 1, kl.dtype)
        kl_scale /= steps_per_epoch * FLAGS.kl_annealing_epochs
        kl_scale = tf.minimum(1., kl_scale)
        kl_loss = kl_scale * kl

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss + kl_loss
        scaled_loss = loss / strategy.num_replicas_in_sync
        elbo = -(negative_log_likelihood + l2_loss + kl)

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weight approximate
          # posterior/prior parameters. This is excludes BN and slow weights,
          # but pay caution to the naming scheme.
          if ('kernel' not in var.name and
              'batch_norm' not in var.name and
              'bias' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/kl'].update_state(kl)
      metrics['train/kl_scale'].update_state(kl_scale)
      metrics['train/elbo'].update_state(elbo)
      metrics['train/loss'].update_state(loss)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/ece'].update_state(labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      if FLAGS.ensemble_size > 1:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      logits = tf.reshape(
          [model(images, training=False)
           for _ in range(FLAGS.num_eval_samples)],
          [FLAGS.num_eval_samples, FLAGS.ensemble_size, -1, num_classes])
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)

      if FLAGS.ensemble_size > 1:
        per_probs = tf.reduce_mean(probs, axis=0)  # marginalize samples
        for i in range(FLAGS.ensemble_size):
          member_probs = per_probs[i]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)

      # Negative log marginal likelihood computed in a numerically-stable way.
      labels_broadcasted = tf.broadcast_to(
          labels,
          [FLAGS.num_eval_samples, FLAGS.ensemble_size, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0, 1]) +
          tf.math.log(float(FLAGS.num_eval_samples * FLAGS.ensemble_size)))
      probs = tf.math.reduce_mean(probs, axis=[0, 1])  # marginalize

      l2_loss = compute_l2_loss(model)
      kl = sum(model.losses) / test_dataset_size
      elbo = -(negative_log_likelihood + l2_loss + kl)

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/kl'].update_state(kl)
        metrics['test/elbo'].update_state(elbo)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/kl_{}'.format(dataset_name)].update_state(kl)
        corrupt_metrics['test/elbo_{}'.format(dataset_name)].update_state(elbo)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
            labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

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
                     current_step / max_steps,
                     epoch + 1,
                     FLAGS.train_epochs,
                     steps_per_sec,
                     eta_seconds / 60,
                     time_elapsed / 60))
      if step % 20 == 0:
        logging.info(message)

    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      for step in range(steps_per_eval):
        if step % 20 == 0:
          logging.info('Starting to run eval step %s of epoch: %s', step,
                       epoch)
        test_step(test_iterator, dataset_name)
      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types,
                                                        max_intensity)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    if FLAGS.ensemble_size > 1:
      for i in range(FLAGS.ensemble_size):
        logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%',
                     i, metrics['test/nll_member_{}'.format(i)].result(),
                     metrics['test/accuracy_member_{}'.format(i)].result()*100)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(corrupt_results)
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

if __name__ == '__main__':
  app.run(main)
