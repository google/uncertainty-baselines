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

"""ResNet50 on Kaggle's Diabetic Retinopathy Detection trained with ML, GD."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import uncertainty_baselines as ub
import utils

flags.DEFINE_integer(
    name='seed',
    default=0,
    help='Random seed.')
flags.DEFINE_string(
    name='output_dir',
    default='/tmp/diabetic_retinopathy_detection',
    help='The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_float(
    name='l2',
    default=1e-4,
    help='L2 coefficient.')
flags.DEFINE_string(
    name='data_dir',
    default=None,
    help='Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_integer(
    name='train_epochs',
    default=90,
    help='Number of training epochs.')
flags.DEFINE_integer(
    name='checkpoint_interval',
    default=25,
    help='Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_bool(
    name='use_bfloat16',
    default=False,
    help='Whether to use mixed precision.')
flags.DEFINE_integer(
    name='batch_size',
    default=16,
    help='The training batch size.')
flags.DEFINE_integer(
    name='eval_batch_size',
    default=32,
    help='The validation/test batch size.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg

  # Parse command line arguments.
  seed = FLAGS.seed
  output_dir = FLAGS.output_dir
  data_dir = FLAGS.data_dir
  train_epochs = FLAGS.train_epochs
  checkpoint_interval = FLAGS.checkpoint_interval
  use_bfloat16 = FLAGS.use_bfloat16
  batch_size = FLAGS.batch_size
  eval_batch_size = FLAGS.eval_batch_size
  steps_per_epoch = 1  # TODO(filangel): function of FLAGS
  steps_per_eval = 1  # TODO(filangel): function of FLAGS

  tf.io.gfile.makedirs(output_dir)
  logging.info('Saving checkpoints at %s', output_dir)
  tf.random.set_seed(seed)

  # TODO(filangel): enable TPU support.
  logging.info('Use GPU')
  strategy = tf.distribute.MirroredStrategy()

  if use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(output_dir, 'summaries'))

  dataset_train_builder = utils.load_diabetic_retinopathy_detection(
      split='train',
      data_dir=data_dir)
  dataset_train = dataset_train_builder.load(batch_size=batch_size)
  dataset_train = strategy.experimental_distribute_dataset(dataset_train)
  dataset_test_builder = utils.load_diabetic_retinopathy_detection(
      split='test',
      data_dir=data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)
  dataset_test = strategy.experimental_distribute_dataset(dataset_test)

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')

    # Shape tuple access depends on number of distributed devices
    try:
      shape_tuple = dataset_train.element_spec['features'].shape
    except AttributeError:  # Multiple TensorSpec in a (nested) PerReplicaSpec.
      tensor_spec_list = dataset_train.element_spec[  # pylint: disable=protected-access
          'features']._flat_tensor_specs
      shape_tuple = tensor_spec_list[0].shape

    model = ub.models.resnet50_deterministic(
        input_shape=shape_tuple.as_list()[1:],
        num_classes=1)  # binary classification task
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    optimizer = tf.keras.optimizers.Adam(1e-4)

    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.BinaryAccuracy(),
        'train/auc': tf.keras.metrics.AUC(),
        'train/loss': tf.keras.metrics.Mean(),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.BinaryAccuracy(),
        'test/auc': tf.keras.metrics.AUC()}
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(output_dir)
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)

  def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']
      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(
                y_true=tf.expand_dims(labels, axis=-1),
                y_pred=logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss

      grads = tape.gradient(loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.squeeze(tf.nn.sigmoid(logits))
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/auc'].update_state(labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  def test_step(iterator):
    """Evaluation step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']
      logits = model(images, training=True)
      if use_bfloat16:
        logits = tf.cast(logits, tf.float32)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.binary_crossentropy(
              y_true=tf.expand_dims(labels, axis=-1),
              y_pred=logits,
              from_logits=True))
      probs = tf.squeeze(tf.nn.sigmoid(logits))
      metrics['test/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['test/accuracy'].update_state(labels, probs)
      metrics['test/auc'].update_state(labels, probs)

    strategy.run(step_fn, args=(next(iterator),))

  train_iterator = iter(dataset_train)
  test_iterator = iter(dataset_test)
  start_time = time.time()
  for epoch in range(train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      train_step(train_iterator)

      current_step = epoch * steps_per_epoch + (step + 1)
      max_steps = steps_per_epoch * train_epochs
      time_elapsed = time.time() - start_time
      steps_per_sec = float(current_step) / time_elapsed
      eta_seconds = (max_steps - current_step) / steps_per_sec
      message = (
          '{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
          'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
              current_step / max_steps, epoch + 1, train_epochs,
              steps_per_sec, eta_seconds / 60, time_elapsed / 60))
      if step % 20 == 0:
        logging.info(message)

    for step in range(steps_per_eval):
      if step % 20 == 0:
        logging.info('Starting to run eval step %s of epoch: %s', step, epoch)
      test_step(test_iterator)

    logging.info(
        'Train Loss (NLL+L2): %.4f, Accuracy: %.2f%%, AUC: %.2f%%',
        metrics['train/loss'].result(),
        metrics['train/accuracy'].result() * 100,
        metrics['train/auc'].result() * 100)
    logging.info(
        'Test NLL: %.4f, Accuracy: %.2f%%, AUC: %.2f%%',
        metrics['test/negative_log_likelihood'].result(),
        metrics['test/accuracy'].result() * 100,
        metrics['test/auc'].result() * 100)
    total_results = {name: metric.result() for name, metric in metrics.items()}
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if checkpoint_interval > 0 and (epoch + 1) % checkpoint_interval == 0:
      checkpoint_name = checkpoint.save(os.path.join(output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  final_checkpoint_name = checkpoint.save(
      os.path.join(output_dir, 'checkpoint'),)
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)


if __name__ == '__main__':
  app.run(main)
