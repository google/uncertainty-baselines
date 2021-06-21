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

"""EfficientNet."""

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

# ~312.78 steps per epoch for 4x4 TPU; per_core_batch_size=128; 350 epochs;

# General model flags
flags.DEFINE_enum('model_name',
                  default='efficientnet-b0',
                  enum_values=['efficientnet-b0', 'efficientnet-b1',
                               'efficientnet-b2', 'efficientnet-b3'],
                  help='Efficientnet model name.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.016,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 5e-6, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 350, 'Number of training epochs.')
flags.DEFINE_integer('checkpoint_interval', 15,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 5, 'How many epochs to run test.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_float('label_smoothing', 0.1, 'label smoothing constant.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')
flags.DEFINE_float('moving_average_decay', 0., 'moving average decay.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)

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

  width_coefficient, depth_coefficient, input_image_size, dropout_rate = (
      ub.models.efficientnet_utils.efficientnet_params(FLAGS.model_name))
  train_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16,
      image_size=input_image_size,
      normalize_input=True,
      one_hot=True)
  train_dataset = train_builder.load(batch_size=batch_size, strategy=strategy)
  test_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16,
      image_size=input_image_size,
      normalize_input=True,
      one_hot=True)
  clean_test_dataset = test_builder.load(
      batch_size=batch_size, strategy=strategy)
  test_datasets = {
      'clean': clean_test_dataset,
  }
  train_iterator = iter(train_dataset)
  test_iterator = iter(test_datasets['clean'])

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building %s model', FLAGS.model_name)
    model = ub.models.efficientnet(width_coefficient,
                                   depth_coefficient,
                                   dropout_rate)

    scaled_lr = FLAGS.base_learning_rate * (batch_size / 256.0)
    # Decay epoch is 2.4, warmup epoch is 5 according to the Efficientnet paper.
    decay_steps = steps_per_epoch * 2.4
    warmup_step = steps_per_epoch * 5
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        scaled_lr, decay_steps, decay_rate=0.97, staircase=True)
    learning_rate = ub.schedules.AddWarmupDecaySchedule(
        lr_schedule, warmup_step)
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate,
        rho=0.9,
        momentum=1.0 - FLAGS.one_minus_momentum,
        epsilon=0.001)
    if FLAGS.moving_average_decay > 0:
      optimizer = ub.optimizers.MovingAverage(
          optimizer,
          average_decay=FLAGS.moving_average_decay)
      optimizer.shadow_copy(model)

    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'train/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'train/loss': tf.keras.metrics.Mean(),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'test/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
    }
    logging.info('Finished building %s model', FLAGS.model_name)

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  def train_step(inputs):
    """Build `step_fn` for efficientnet learning."""
    images = inputs['features']
    labels = inputs['labels']

    num_replicas = tf.cast(strategy.num_replicas_in_sync, tf.float32)
    l2_coeff = tf.cast(FLAGS.l2, tf.float32)

    with tf.GradientTape() as tape:
      logits = model(images, training=True)
      logits = tf.cast(logits, tf.float32)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(
              labels,
              logits,
              from_logits=True,
              label_smoothing=FLAGS.label_smoothing))

      def _is_batch_norm(v):
        """Decide whether a variable belongs to `batch_norm`."""
        keywords = ['batchnorm', 'batch_norm', 'bn']
        return any([k in v.name.lower() for k in keywords])

      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_weights
                          if not _is_batch_norm(v)])
      loss = negative_log_likelihood + l2_coeff * l2_loss
      scaled_loss = loss / num_replicas

    gradients = tape.gradient(scaled_loss, model.trainable_weights)
    # MovingAverage optimizer automatically updates avg when applying gradients.
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))

    sparse_labels = tf.cast(
        tf.math.argmax(labels, axis=-1, output_type=tf.int32), tf.float32)
    probs = tf.nn.softmax(logits)
    metrics['train/loss'].update_state(loss)
    metrics['train/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['train/accuracy'].update_state(labels, logits)
    metrics['train/ece'].add_batch(probs, label=sparse_labels)

    step_info = {
        'loss/negative_log_likelihood': negative_log_likelihood / num_replicas,
        'loss/total_loss': scaled_loss,
    }
    return step_info

  def eval_step(inputs):
    """A single step."""
    images = inputs['features']
    labels = inputs['labels']
    logits = model(images, training=False)
    logits = tf.cast(logits, tf.float32)
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(
            labels, logits, from_logits=True))
    sparse_labels = tf.cast(
        tf.math.argmax(labels, axis=-1, output_type=tf.int32), tf.float32)
    probs = tf.nn.softmax(logits)
    metrics['test/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['test/accuracy'].update_state(labels, logits)
    metrics['test/ece'].add_batch(probs, label=sparse_labels)

  @tf.function
  def epoch_fn(should_eval):
    """Build `epoch_fn` for training and potential eval."""
    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      info = strategy.run(train_step, args=(next(train_iterator),))

      optim_step = optimizer.iterations
      if optim_step % tf.cast(100, optim_step.dtype) == 0:
        for k, v in info.items():
          v_reduce = strategy.reduce(tf.distribute.ReduceOp.SUM, v, None)
          tf.summary.scalar(k, v_reduce, optim_step)
        tf.summary.scalar('loss/lr', learning_rate(optim_step), optim_step)
        summary_writer.flush()

    if should_eval:
      if isinstance(optimizer, ub.optimizers.MovingAverage):
        optimizer.swap_weights(strategy)
      for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
        strategy.run(eval_step, args=(next(test_iterator),))
      if isinstance(optimizer, ub.optimizers.MovingAverage):
        optimizer.swap_weights(strategy)

  # Main training loop.
  start_time = time.time()
  with summary_writer.as_default():
    for epoch in range(initial_epoch, FLAGS.train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      should_eval = (epoch % FLAGS.evaluation_interval == 0)
      epoch_start_time = time.time()
      # Pass tf constant to avoid re-tracing.
      epoch_fn(tf.constant(should_eval))
      epoch_time = time.time() - epoch_start_time
      example_per_secs = (steps_per_epoch * batch_size) / epoch_time
      if not should_eval:
        tf.summary.scalar(
            'examples_per_secs', example_per_secs, optimizer.iterations)
        summary_writer.flush()

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

      logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                   metrics['train/loss'].result(),
                   metrics['train/accuracy'].result() * 100)

      if should_eval:
        logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                     metrics['test/negative_log_likelihood'].result(),
                     metrics['test/accuracy'].result() * 100)

      total_metrics = metrics.copy()
      total_results = {name: metric.result()
                       for name, metric in total_metrics.items()}
      total_results.update({'lr': learning_rate(optimizer.iterations)})
      # Metrics from Robustness Metrics (like ECE) will return a dict with a
      # single key/value, instead of a scalar.
      total_results = {
          k: (list(v.values())[0] if isinstance(v, dict) else v)
          for k, v in total_results.items()
      }
      with summary_writer.as_default():
        for name, result in total_results.items():
          if should_eval or 'test' not in name:
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
    })


if __name__ == '__main__':
  app.run(main)
