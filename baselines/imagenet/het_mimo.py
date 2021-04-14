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

"""Multiheaded ResNet-50 with Heteroscedastic Layer."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_integer('ensemble_size', 2, 'Size of ensemble.')
flags.DEFINE_float('input_repetition_probability', 0.6,
                   'The probability that the inputs are identical for the'
                   'ensemble members.')
flags.DEFINE_integer('batch_repetitions', 2, 'Number of times an example is'
                     'repeated in a training batch. More repetitions lead to'
                     'lower variance gradients and increased training time.')
flags.DEFINE_integer('width_multiplier', 1, 'Integer to multiply the number of'
                     'typical filters by. "k" in ResNet-n-k.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 5,
    'Number of epochs for a linear warmup to the initial learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60', '80'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 150, 'Number of training epochs.')
flags.DEFINE_integer('checkpoint_interval', -1,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# Heteroscedastic flags.
flags.DEFINE_integer('num_factors', 15,
                     'Num factors to approximate full rank covariance matrix.')
flags.DEFINE_float('temperature', 1.5,
                   'Temperature for heteroscedastic head.')
flags.DEFINE_integer('num_mc_samples', 5000,
                     'Num MC samples for heteroscedastic layer.')
flags.DEFINE_bool('share_het_layer', True,
                  'Whether to use a single heteroscedastic layer of output size'
                  '(num_classes * ensemble_size), or to generate an ensemble'
                  'size number of het. layers with num_classes as output size.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  train_batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores
                      // FLAGS.batch_repetitions)
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // train_batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // test_batch_size

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

  train_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = train_builder.load(
      batch_size=train_batch_size, strategy=strategy)
  test_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16)
  test_dataset = test_builder.load(
      batch_size=test_batch_size, strategy=strategy)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = ub.models.resnet50_het_mimo(
        input_shape=(FLAGS.ensemble_size, 224, 224, 3),
        num_classes=NUM_CLASSES,
        ensemble_size=FLAGS.ensemble_size,
        num_factors=FLAGS.num_factors,
        temperature=FLAGS.temperature,
        num_mc_samples=FLAGS.num_mc_samples,
        share_het_layer=FLAGS.share_het_layer,
        width_multiplier=FLAGS.width_multiplier
        )
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * train_batch_size / 256
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

    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())
    test_diversity = {
        'test/disagreement': tf.keras.metrics.Mean(),
        'test/average_kl': tf.keras.metrics.Mean(),
        'test/cosine_similarity': tf.keras.metrics.Mean(),
    }
    logging.info('Finished building Keras ResNet-50 model')

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
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(tf.reduce_sum(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True),
            axis=1))
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

      probs = tf.nn.softmax(tf.reshape(logits, [-1, NUM_CLASSES]))
      flat_labels = tf.reshape(labels, [-1])
      metrics['train/ece'].add_batch(probs, label=flat_labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(flat_labels, probs)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(
          tf.expand_dims(images, 1), [1, FLAGS.ensemble_size, 1, 1, 1])
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)

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
        metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
        metrics['test/accuracy_member_{}'.format(i)].update_state(
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

      metrics['test/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['test/accuracy'].update_state(labels, probs)
      metrics['test/ece'].add_batch(probs, label=labels)

    for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
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

    test_iterator = iter(test_dataset)
    logging.info('Starting to run eval of epoch: %s', epoch)
    test_start_time = time.time()
    test_step(test_iterator)
    ms_per_example = (time.time() - test_start_time) * 1e6 / test_batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    for i in range(FLAGS.ensemble_size):
      logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%',
                   i, metrics['test/nll_member_{}'.format(i)].result(),
                   metrics['test/accuracy_member_{}'.format(i)].result() * 100)

    total_metrics = metrics.copy()
    total_metrics.update(test_diversity)
    total_results = {name: metric.result()
                     for name, metric in total_metrics.items()}
    # Metrics from Robustness Metrics (like ECE) will return a dict with a
    # single key/value, instead of a scalar.
    total_results = {
        k: (list(v.values())[0] if isinstance(v, dict) else v)
        for k, v in total_results.items()
    }
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for _, metric in total_metrics.items():
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
        'batch_repetitions': FLAGS.batch_repetitions,
    })

if __name__ == '__main__':
  app.run(main)
