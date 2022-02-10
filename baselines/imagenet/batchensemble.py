# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Batch Ensemble ResNet-50."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import from baselines.imagenet
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_integer('depth', 50, 'ResNet depth (50 or 101).')
flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.25,
                   'fast weights lr multiplier.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 135, 'Number of training epochs.')
flags.DEFINE_integer('corruptions_interval', 135,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 27,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')
flags.DEFINE_bool('use_ensemble_bn', False, 'Whether to use ensemble bn.')

# Data Augmentation flags.
flags.DEFINE_float('mixup_alpha', 0., 'Mixup regularization coefficient.')
flags.DEFINE_bool('adaptive_mixup', False, 'Whether to adaptive mixup.')
flags.DEFINE_integer('confidence_eval_iterations', 10,
                     'Number of iterations to evaluate the reliablity diagram.')

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

_LR_SCHEDULE = [    # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  per_core_batch_size = FLAGS.per_core_batch_size // FLAGS.ensemble_size
  batch_size = per_core_batch_size * FLAGS.num_cores
  steps_per_epoch = APPROX_IMAGENET_TRAIN_IMAGES // batch_size
  steps_per_eval = IMAGENET_VALIDATION_IMAGES // batch_size

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

  mixup_params = {
      'ensemble_size': FLAGS.ensemble_size,
      'mixup_alpha': FLAGS.mixup_alpha,
      'adaptive_mixup': FLAGS.adaptive_mixup,
      'num_classes': NUM_CLASSES,
  }
  train_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TRAIN,
      one_hot=(FLAGS.mixup_alpha > 0),
      use_bfloat16=FLAGS.use_bfloat16,
      mixup_params=mixup_params,
      ensemble_size=FLAGS.ensemble_size,
      data_dir=data_dir)
  train_dataset = train_builder.load(batch_size=batch_size, strategy=strategy)
  test_builder = ub.datasets.ImageNetDataset(
      split=tfds.Split.TEST, use_bfloat16=FLAGS.use_bfloat16, data_dir=data_dir)
  clean_test_dataset = test_builder.load(
      batch_size=batch_size, strategy=strategy)
  test_datasets = {
      'clean': clean_test_dataset,
  }
  if FLAGS.adaptive_mixup:
    validation_builder = ub.datasets.ImageNetDataset(
        split=tfds.Split.VALIDATION,
        run_mixup=True,
        use_bfloat16=FLAGS.use_bfloat16,
        data_dir=data_dir)
    imagenet_confidence_dataset = validation_builder.load(
        batch_size=FLAGS.per_core_batch_size * FLAGS.num_cores,
        strategy=strategy)
  if FLAGS.corruptions_interval > 0:
    corruption_types, max_intensity = utils.load_corrupted_test_info()
    for name in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset_name = '{0}_{1}'.format(name, intensity)
        dataset = utils.load_corrupted_test_dataset(
            batch_size=batch_size,
            corruption_name=name,
            corruption_intensity=intensity,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets[dataset_name] = (
            strategy.experimental_distribute_dataset(dataset))

  if FLAGS.use_bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = ub.models.resnet_batchensemble(
        input_shape=(224, 224, 3),
        num_classes=NUM_CLASSES,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        use_ensemble_bn=FLAGS.use_ensemble_bn,
        depth=FLAGS.depth)
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
        'train/diversity': rm.metrics.AveragePairwiseDiversity(),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'test/diversity': rm.metrics.AveragePairwiseDiversity(),
        'test/member_accuracy_mean': (
            tf.keras.metrics.SparseCategoricalAccuracy()),
        'test/member_ece_mean': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins)
    }

    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
          corrupt_metrics['test/member_acc_mean_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/member_ece_mean_{}'.format(dataset_name)] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())

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

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      if FLAGS.adaptive_mixup:
        images = tf.identity(images)
      else:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])

      if FLAGS.adaptive_mixup:
        labels = tf.identity(labels)
      elif FLAGS.mixup_alpha > 0:
        labels = tf.tile(labels, [FLAGS.ensemble_size, 1])
      else:
        labels = tf.tile(labels, [FLAGS.ensemble_size])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        probs = tf.nn.softmax(logits)
        per_probs = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        metrics['train/diversity'].add_batch(per_probs)

        if FLAGS.mixup_alpha > 0:
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.categorical_crossentropy(labels,
                                                       logits,
                                                       from_logits=True))
        else:
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the slow weights and bias terms. This excludes BN
          # parameters and fast weight approximate posterior/prior parameters,
          # but pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

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

      if FLAGS.mixup_alpha > 0:
        labels = tf.argmax(labels, axis=-1)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)

      if dataset_name == 'clean':
        per_probs_tensor = tf.reshape(
            probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
        metrics['test/diversity'].add_batch(per_probs_tensor)

      per_probs = tf.split(
          probs, num_or_size_splits=FLAGS.ensemble_size, axis=0)
      probs = tf.reduce_mean(per_probs, axis=0)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      for i in range(FLAGS.ensemble_size):
        member_probs = per_probs[i]
        if dataset_name == 'clean':
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)
          metrics['test/member_accuracy_mean'].update_state(
              labels, member_probs)
          metrics['test/member_ece_mean'].add_batch(member_probs, label=labels)
        elif dataset_name != 'confidence_validation':
          corrupt_metrics['test/member_acc_mean_{}'.format(
              dataset_name)].update_state(labels, member_probs)
          corrupt_metrics['test/member_ece_mean_{}'.format(
              dataset_name)].add_batch(member_probs, label=labels)

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)
      elif dataset_name != 'confidence_validation':
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

      if dataset_name == 'confidence_validation':
        return tf.stack(per_probs, 0), labels

    return strategy.run(step_fn, args=(next(iterator),))

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

    if FLAGS.adaptive_mixup:
      confidence_set_iterator = iter(imagenet_confidence_dataset)
      predictions_list = []
      labels_list = []
      for step in range(FLAGS.confidence_eval_iterations):
        temp_predictions, temp_labels = test_step(
            confidence_set_iterator, 'confidence_validation')
        predictions_list.append(temp_predictions)
        labels_list.append(temp_labels)
      predictions = [tf.concat(list(predictions_list[i].values), axis=1)
                     for i in range(len(predictions_list))]
      labels = [tf.concat(list(labels_list[i].values), axis=0)
                for i in range(len(labels_list))]
      predictions = tf.concat(predictions, axis=1)
      labels = tf.cast(tf.concat(labels, axis=0), tf.int64)

      def compute_acc_conf(preds, label, focus_class):
        class_preds = tf.boolean_mask(preds, label == focus_class, axis=1)
        class_pred_labels = tf.argmax(class_preds, axis=-1)
        confidence = tf.reduce_mean(tf.reduce_max(class_preds, axis=-1), -1)
        accuracy = tf.reduce_mean(tf.cast(
            class_pred_labels == focus_class, tf.float32), axis=-1)
        return accuracy - confidence

      calibration_per_class = [compute_acc_conf(
          predictions, labels, i) for i in range(NUM_CLASSES)]
      calibration_per_class = tf.stack(calibration_per_class, axis=1)
      logging.info('calibration per class')
      logging.info(calibration_per_class)
      mixup_coeff = tf.where(calibration_per_class > 0, 1.0, FLAGS.mixup_alpha)
      mixup_coeff = tf.clip_by_value(mixup_coeff, 0, 1)
      logging.info('mixup coeff')
      logging.info(mixup_coeff)
      mixup_params['mixup_coeff'] = mixup_coeff
      train_builder = ub.datasets.ImageNetDataset(
          split=tfds.Split.TRAIN,
          one_hot=(FLAGS.mixup_alpha > 0),
          use_bfloat16=FLAGS.use_bfloat16,
          mixup_params=mixup_params,
          data_dir=data_dir)
      train_dataset = train_builder.load(
          batch_size=batch_size, strategy=strategy)
      train_iterator = iter(train_dataset)

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
        test_start_time = time.time()
        test_step(test_iterator, dataset_name)
        ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
        metrics['test/ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(
          corrupt_metrics, corruption_types, max_intensity,
          FLAGS.alexnet_errors_path)

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

    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(corrupt_results)
    # Results from Robustness Metrics themselves return a dict, so flatten them.
    total_results = utils.flatten_dictionary(total_results)
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for _, metric in metrics.items():
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
        'random_sign_init': FLAGS.random_sign_init,
        'fast_weight_lr_multiplier': FLAGS.fast_weight_lr_multiplier,
    })


if __name__ == '__main__':
  app.run(main)
