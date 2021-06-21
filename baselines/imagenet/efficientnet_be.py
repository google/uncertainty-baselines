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

"""EfficientNet with BatchEnsemble."""

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

# TODO(trandustin): Tune results.
# General model flags
flags.DEFINE_enum('model_name',
                  default='efficientnet-b0',
                  enum_values=['efficientnet-b0', 'efficientnet-b1',
                               'efficientnet-b2', 'efficientnet-b3'],
                  help='Efficientnet model name.')
flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.016,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.5,
                   'fast weights lr multiplier.')
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

  per_core_batch_size = FLAGS.per_core_batch_size // FLAGS.ensemble_size
  batch_size = per_core_batch_size * FLAGS.num_cores
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
    model = ub.models.efficientnet_batch_ensemble(
        width_coefficient,
        depth_coefficient,
        dropout_rate,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init)

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
    images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
    labels = tf.tile(labels, [FLAGS.ensemble_size, 1])

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

      filtered_variables = []
      for var in model.trainable_variables:
        # Apply l2 on the slow weights and bias terms. This excludes BN
        # parameters and fast weight approximate posterior/prior parameters,
        # but pay caution to their naming scheme.
        if 'kernel' in var.name or 'bias' in var.name:
          filtered_variables.append(tf.reshape(var, (-1,)))

      l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
          tf.concat(filtered_variables, axis=0))
      loss = negative_log_likelihood + l2_coeff * l2_loss
      scaled_loss = loss / num_replicas

    grads = tape.gradient(scaled_loss, model.trainable_weights)

    # Separate learning rate implementation.
    if FLAGS.fast_weight_lr_multiplier != 1.0:
      grads_and_vars = []
      for grad, var in zip(grads, model.trainable_variables):
        # Apply different learning rate on the fast weights. This excludes BN
        # and slow weights, but pay caution to the naming scheme.
        if ('batch_norm' not in var.name and 'kernel' not in var.name):
          grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier,
                                 var))
        else:
          grads_and_vars.append((grad, var))
      optimizer.apply_gradients(grads_and_vars)
    else:
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

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
    images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
    logits = model(images, training=False)
    logits = tf.cast(logits, tf.float32)
    probs = tf.nn.softmax(logits)
    per_probs = tf.split(
        probs, num_or_size_splits=FLAGS.ensemble_size, axis=0)
    probs = tf.reduce_mean(per_probs, axis=0)

    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(labels, probs))
    sparse_labels = tf.cast(
        tf.math.argmax(labels, axis=-1, output_type=tf.int32), tf.float32)
    metrics['test/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['test/accuracy'].update_state(labels, probs)
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
      for _ in tf.range(tf.cast(steps_per_eval, tf.int32)):
        strategy.run(eval_step, args=(next(test_iterator),))

  # Main training loop.
  start_time = time.time()
  with summary_writer.as_default():
    for epoch in range(initial_epoch, FLAGS.train_epochs):
      logging.info('Starting to run epoch: %s', epoch)
      should_eval = (epoch % FLAGS.evaluation_interval == 0)
      # Pass tf constant to avoid re-tracing.
      epoch_fn(tf.constant(should_eval))

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
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
        'random_sign_init': FLAGS.random_sign_init,
        'fast_weight_lr_multiplier': FLAGS.fast_weight_lr_multiplier,
    })


if __name__ == '__main__':
  app.run(main)
