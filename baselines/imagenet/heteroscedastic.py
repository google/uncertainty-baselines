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

"""Heteroscedastic [1] ResNet-50 on ImageNet trained with maximum likelihood.

This script supports using mixup [2], possibly combined with the rescaling of
the predictions proposed in [3] (see the metrics ending with `+rescaling`).
Mixup is enabled by setting ``mixup_alpha > 0`.

## References:

[1]: Mark Collier, Basil Mustafa, Efi Kokiopoulou, Rodolphe Jenatton and
     Jesse Berent. Correlated Input-Dependent Label Noise in Large-Scale Image
     Classification. In Proc. of the IEEE/CVF Conference on Computer Vision
     and Pattern Recognition (CVPR), 2021, pp. 1551-1560.
     https://arxiv.org/abs/2105.10305
[2]: Hongyi Zhang et al. mixup: Beyond Empirical Risk Minimization.
     _arXiv preprint arXiv:1710.09412_, 2017.
     https://arxiv.org/abs/1710.09412
[3]: Luigi Carratino et al. On Mixup Regularization.
     _arXiv preprint arXiv:2006.06049_, 2020.
     https://arxiv.org/abs/2006.06049
"""

import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import robustness_metrics as rm
import scipy
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 180, 'Number of training epochs.')
flags.DEFINE_integer('checkpoint_interval', 25,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# Mixup-related flags.
flags.DEFINE_float('mixup_alpha', 0., 'Coefficient of mixup distribution.')
flags.DEFINE_bool('same_mix_weight_per_batch', False,
                  'Whether to use a single mix weight across the batch.')
flags.DEFINE_bool('use_random_shuffling', False,
                  'Whether to use random shuffling to pair the points of mixup'
                  'within a batch.')
flags.DEFINE_bool('use_truncated_beta', True,
                  'Whether to sample the mixup weights from '
                  'Beta[0,1](alpha,alpha) or from the truncated distribution '
                  'Beta[1/2,1](alpha,alpha).')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

# Heteroscedastic flags.
flags.DEFINE_integer('num_factors', 15,
                     'Num factors to approximate full rank covariance matrix.')
flags.DEFINE_float('temperature', 1.5,
                   'Temperature for heteroscedastic head.')
flags.DEFINE_integer('num_mc_samples', 5000,
                     'Num MC samples for heteroscedastic layer.')

FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000


IMAGE_SHAPE = (224, 224, 3)


def mean_truncated_beta_distribution(alpha):
  """Expectation of a truncated beta(alpha, alpha) distribution in [1/2, 1]."""
  return 1. - scipy.special.betainc(alpha + 1, alpha, .5)


def main(argv):

  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

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

  enable_mixup = (FLAGS.mixup_alpha > 0.0)
  mixup_params = {
      'mixup_alpha': FLAGS.mixup_alpha,
      'adaptive_mixup': False,
      'same_mix_weight_per_batch': FLAGS.same_mix_weight_per_batch,
      'use_random_shuffling': FLAGS.use_random_shuffling,
      'use_truncated_beta': FLAGS.use_truncated_beta
  }

  train_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16,
      one_hot=True,
      mixup_params=mixup_params)
  test_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = train_builder.load(batch_size=batch_size, strategy=strategy)
  test_dataset = test_builder.load(batch_size=batch_size, strategy=strategy)

  if enable_mixup:

    mean_theta = mean_truncated_beta_distribution(FLAGS.mixup_alpha)

    # Train set to compute the means of the images and of the (one-hot) labels
    imagenet_train_no_mixup = ub.datasets.ImageNetDataset(
        split=tfds.Split.TRAIN,
        use_bfloat16=FLAGS.use_bfloat16,
        one_hot=True)
    tr_data_no_mixup = imagenet_train_no_mixup.load(
        batch_size=batch_size, strategy=strategy)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  with strategy.scope():

    if enable_mixup:
      # Variables used to track the means of the images and the (one-hot) labels
      count = tf.Variable(tf.zeros((1,), dtype=tf.float32))
      mean_images = tf.Variable(tf.zeros(IMAGE_SHAPE, dtype=tf.float32))
      mean_labels = tf.Variable(tf.zeros((NUM_CLASSES,), dtype=tf.float32))

    logging.info('Building Keras ResNet-50 model')
    model = ub.models.resnet50_heteroscedastic(
        input_shape=IMAGE_SHAPE, num_classes=NUM_CLASSES,
        temperature=FLAGS.temperature, num_factors=FLAGS.num_factors,
        num_mc_samples=FLAGS.num_mc_samples)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 256
    decay_epochs = [
        (FLAGS.train_epochs * 30) // 90,
        (FLAGS.train_epochs * 60) // 90,
        (FLAGS.train_epochs * 80) // 90,
    ]
    learning_rate = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch=steps_per_epoch,
        base_learning_rate=base_lr,
        decay_ratio=0.1,
        decay_epochs=decay_epochs,
        warmup_epochs=5)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
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
    logging.info('Finished building Keras ResNet-50 model')

    if enable_mixup:
      # With mixup enabled, we log the predictions with the rescaling from [2]
      metrics['test/negative_log_likelihood+rescaling'] = (tf.keras.metrics
                                                           .Mean())
      metrics['test/accuracy+rescaling'] = (tf.keras.metrics
                                            .SparseCategoricalAccuracy())
      metrics['test/ece+rescaling'] = rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  @tf.function
  def moving_average_step(iterator):
    """Training StepFn to compute the means of the images and labels."""

    def step_fn_labels(labels):
      return tf.reduce_mean(labels, axis=0)

    def step_fn_images(images):
      return tf.reduce_mean(tf.cast(images, tf.float32), axis=0)

    new_count = count + 1.
    count.assign(new_count)

    inputs = next(iterator)
    images = inputs['features']
    labels = inputs['labels']

    per_replica_means = strategy.run(step_fn_labels, args=(labels,))
    cr_replica_means = strategy.reduce('mean', per_replica_means, axis=0)
    mean_labels.assign(cr_replica_means/count + (count-1.)/count * mean_labels)

    per_replica_means = strategy.run(step_fn_images, args=(images,))
    cr_replica_means = strategy.reduce('mean', per_replica_means, axis=0)
    mean_images.assign(cr_replica_means/count + (count-1.)/count * mean_images)

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      with tf.GradientTape() as tape:

        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels, logits, from_logits=True))

        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the weights. This excludes BN parameters and biases, but
          # pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)

      # We go back from one-hot labels to integers
      labels = tf.argmax(labels, axis=-1)

      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def update_test_metrics(labels, logits, metric_suffix=''):
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, logits, from_logits=True))
    probs = tf.nn.softmax(logits)
    metrics['test/negative_log_likelihood' + metric_suffix].update_state(
        negative_log_likelihood)
    metrics['test/accuracy' + metric_suffix].update_state(labels, probs)
    metrics['test/ece' + metric_suffix].add_batch(probs, label=labels)

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)

      update_test_metrics(labels, logits)

      # Rescaling logic in Eq.(15) from [2]
      if enable_mixup:
        images *= mean_theta
        images += (1.-mean_theta) * tf.cast(mean_images, images.dtype)

        scaled_logits = model(images, training=False)
        if FLAGS.use_bfloat16:
          scaled_logits = tf.cast(scaled_logits, tf.float32)

        scaled_logits *= 1./mean_theta
        scaled_logits += (1.-1./mean_theta) * tf.cast(mean_labels, logits.dtype)

        update_test_metrics(labels, scaled_logits, '+rescaling')

    for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  if enable_mixup:
    logging.info('Starting to compute the means of labels and images')
    tr_iterator_no_mixup = iter(tr_data_no_mixup)
    for _ in range(steps_per_epoch):
      moving_average_step(tr_iterator_no_mixup)
    # Save stats required by the mixup rescaling [2] for subsequent predictions
    mixup_rescaling_stats = {
        'mean_labels': mean_labels.numpy(),
        'mean_images': mean_images.numpy(),
        'mean_theta': mean_theta
    }
    output_dir = os.path.join(FLAGS.output_dir, 'mixup_rescaling_stats.npz')
    with tf.io.gfile.GFile(output_dir, 'wb') as f:
      np.save(f, list(mixup_rescaling_stats.items()))
    logging.info('Finished to compute the means of labels and images')

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

    test_iterator = iter(test_dataset)
    logging.info('Starting to run eval at epoch: %s', epoch)
    test_start_time = time.time()
    test_step(test_iterator)
    ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    if enable_mixup:
      logging.info(
          'Test NLL (+ rescaling): %.4f, Accuracy (+ rescaling): %.2f%%',
          metrics['test/negative_log_likelihood+rescaling'].result(),
          metrics['test/accuracy+rescaling'].result() * 100)

    total_results = {name: metric.result() for name, metric in metrics.items()}
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
      checkpoint_name = checkpoint.save(os.path.join(
          FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_save_name = os.path.join(FLAGS.output_dir, 'model')
  model.save(final_save_name)
  logging.info('Saved model to %s', final_save_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
        'num_factors': FLAGS.num_factors,
        'temperature': FLAGS.temperature,
        'num_mc_samples': FLAGS.num_mc_samples,
    })


if __name__ == '__main__':
  app.run(main)
