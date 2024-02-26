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

"""ResNet-50 trained on ImageNet-PI with maximum likelihood and access to PI during training and approximate full marginalization during test.
"""

import copy
import os
import time

from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub


flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float(
    'base_learning_rate',
    0.1,
    'Base learning rate when train batch size is 256.',
)
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float('l2', 1e-5, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_string(
    'output_dir',
    '/tmp/imagenet',
    'The directory where the model weights and '
    'training/evaluation summaries are stored.',
)
flags.DEFINE_integer('train_epochs', 90, 'Number of training epochs.')
flags.DEFINE_integer(
    'checkpoint_interval',
    25,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.',
)
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')
flags.DEFINE_float(
    'train_proportion',
    1.0,
    'What proportion of the training set to use to train versus validate on.',
)
flags.register_validator(
    'train_proportion',
    lambda p: p > 0 and p <= 1,
    message='--train_proportion must be in (0, 1].',
)
flags.DEFINE_float(
    'max_validation_proportion',
    0.05,
    'Maximum proportion of the training set to use as validation set.',
)
flags.register_validator(
    'max_validation_proportion',
    lambda p: p >= 0 and p < 1,
    message='--max_validation_proportion must be in [0, 1).',
)

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', True, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None, 'Name of the TPU. Only used if use_gpu is False.'
)

# Model flags
flags.DEFINE_float(
    'model_width_multiplier',
    1.0,
    'Factor used to resize the width of the model.',
)
flags.DEFINE_integer(
    'pi_tower_width', 1024, 'Width of the hidden layers in the PI tower.'
)

# ImageNet-PI flags.
flags.DEFINE_integer(
    'num_annotators_per_example',
    16,
    'Number of annotators to load per training example.',
)
flags.DEFINE_enum(
    'annotator_sampling_strategy',
    'uniform',
    ['uniform', 'best', 'worst'],
    help='Strategy used to sample annotators.',
)
flags.DEFINE_integer(
    'num_annotators_per_example_and_step',
    None,
    'Number of annotators to load per example and step during training.',
)
flags.DEFINE_string(
    'annotations_path',
    None,
    'Path to a file that stores the model annotations.',
)
flags.DEFINE_float(
    'max_reliability',
    1.0,
    'Maximum reliability threshold of the annotators. Any annotator with a'
    ' higher reliability is discarded.',
)
flags.DEFINE_float(
    'min_reliability',
    0.0,
    'Minimum reliability threshold of the annotators. Any annotator with a'
    ' lower reliability is discarded.',
)
flags.DEFINE_integer(
    'reliability_estimation_batch_size',
    4096,
    'Number of examples to process in parallel when estimating reliability of'
    ' ImageNet-PI.',
)
flags.DEFINE_bool(
    'disable_reliability_estimation',
    False,
    'Whether to disable the expensive reliability estimation step when loading'
    ' the data.',
)
flags.DEFINE_integer(
    'artificial_id_increase_factor',
    None,
    'Number of times the set of availble ids will be artificially increased.',
)
flags.DEFINE_bool(
    'use_annotator_labels',
    True,
    'Whether or not to use the annotation labels in the cross-entropy loss.',
)
flags.DEFINE_string(
    'pi_subset',
    'annotator_ids,annotator_confidences,annotator_features',
    'Comma-separated string defining subset of additional information to use'
    ' as PI.',
)
flags.DEFINE_integer('random_pi_length', 1, 'Length of random PI features.')
flags.DEFINE_integer(
    'pi_seed', None, 'Seed controlling randomness on PI sampling.'
)
flags.DEFINE_integer(
    'num_adversarial_annotators_per_example',
    0,
    'Number of adversarial annotators to add per example.',
)

# Marginalization flags.
flags.DEFINE_integer(
    'num_mc_samples',
    1000,
    'Number of MC samples to use during the approximate full marginalization of'
    ' the PI.',
)
FLAGS = flags.FLAGS

# Number of images in ImageNet-1k train dataset.
APPROX_IMAGENET_TRAIN_IMAGES = 1281167
# Number of images in eval dataset.
IMAGENET_VALIDATION_IMAGES = 50000
NUM_CLASSES = 1000

_LR_SCHEDULE = [  # (multiplier, epoch to start) tuples
    (1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)
]

IMAGE_SHAPE = (224, 224, 3)


def main(argv):

  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

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

  train_builder = ub.datasets.ImageNetPIDataset(
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16,
      one_hot=True,
      validation_percent=1.0 - FLAGS.train_proportion,
      data_dir=data_dir,
      annotations_path=FLAGS.annotations_path,
      num_annotators_per_example=FLAGS.num_annotators_per_example,
      num_annotators_per_example_and_step=FLAGS
      .num_annotators_per_example_and_step,
      reliability_interval=(FLAGS.min_reliability, FLAGS.max_reliability),
      reliability_estimation_batch_size=FLAGS.reliability_estimation_batch_size,
      artificial_id_increase_factor=FLAGS.artificial_id_increase_factor,
      pi_seed=FLAGS.pi_seed,
      seed=FLAGS.seed,
      num_adversarial_annotators_per_example=FLAGS
      .num_adversarial_annotators_per_example,
      annotator_sampling_strategy=FLAGS.annotator_sampling_strategy,
      random_pi_length=FLAGS.random_pi_length,
      disable_reliability_estimation=FLAGS.disable_reliability_estimation,
  )
  steps_per_epoch = train_builder.num_examples // batch_size
  test_builder = ub.datasets.ImageNetPIDataset(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16,
      data_dir=data_dir,
      annotations_path=FLAGS.annotations_path,
      num_annotators_per_example=FLAGS.num_annotators_per_example,
      reliability_interval=(FLAGS.min_reliability, FLAGS.max_reliability),
      reliability_estimation_batch_size=FLAGS.reliability_estimation_batch_size,
      artificial_id_increase_factor=FLAGS.artificial_id_increase_factor,
      pi_seed=FLAGS.pi_seed,
      seed=FLAGS.seed,
      num_adversarial_annotators_per_example=FLAGS
      .num_adversarial_annotators_per_example,
      annotator_sampling_strategy=FLAGS.annotator_sampling_strategy,
      random_pi_length=FLAGS.random_pi_length,
      disable_reliability_estimation=FLAGS.disable_reliability_estimation,
  )

  # Load one example to infer pi_shape.
  dummy_example = train_builder.load(batch_size=1).take(1).get_single_element()

  train_dataset = train_builder.load(batch_size=batch_size, strategy=strategy)
  test_dataset = test_builder.load(batch_size=batch_size, strategy=strategy)
  steps_per_test_eval = IMAGENET_VALIDATION_IMAGES // batch_size
  validation_dataset = None
  steps_per_validation_eval = 0
  if FLAGS.train_proportion < 1.0:
    validation_proportion = min(1.0 - FLAGS.train_proportion,
                                FLAGS.max_validation_proportion)
    logging.info('Setting aside %f%% of the training set for validation',
                 validation_proportion * 100)
    # Note we do not one_hot the validation set.
    validation_builder = ub.datasets.ImageNetPIDataset(
        split=tfds.Split.VALIDATION,
        use_bfloat16=FLAGS.use_bfloat16,
        validation_percent=validation_proportion,
        data_dir=data_dir,
        annotations_path=FLAGS.annotations_path,
        num_annotators_per_example=FLAGS.num_annotators_per_example,
        num_annotators_per_example_and_step=FLAGS
        .num_annotators_per_example_and_step,
        reliability_interval=(FLAGS.min_reliability, FLAGS.max_reliability),
        reliability_estimation_batch_size=FLAGS
        .reliability_estimation_batch_size,
        artificial_id_increase_factor=FLAGS.artificial_id_increase_factor,
        pi_seed=FLAGS.pi_seed,
        seed=FLAGS.seed,
        num_adversarial_annotators_per_example=FLAGS
        .num_adversarial_annotators_per_example,
        annotator_sampling_strategy=FLAGS.annotator_sampling_strategy,
        random_pi_length=FLAGS.random_pi_length,
        disable_reliability_estimation=FLAGS.disable_reliability_estimation,
    )
    validation_dataset = validation_builder.load(
        batch_size=batch_size, strategy=strategy)
    steps_per_validation_eval = validation_builder.num_examples // batch_size

  if FLAGS.use_bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  def annotator_ids_encoding_fn(example):
    return tf.one_hot(
        tf.cast(example['pi_features']['annotator_ids'], tf.int32),
        train_builder.num_dataset_annotators,
        dtype=tf.int32)

  def annotator_label_if_incorrect_encoding(example):
    return pi_utils.annotator_label_if_incorrect_encoding_fn(example=example)

  encoding_fn_dict = {
      'annotator_ids':
          annotator_ids_encoding_fn,
      'annotator_confidences':
          lambda e: e['pi_features']['annotator_confidences'],
      'annotator_features':
          lambda e: e['pi_features']['annotator_features'],
      'annotator_labels':
          lambda e: e['pi_features']['annotator_labels'],
      'random_pi':
          lambda e: e['pi_features']['random_pi'],
      'clean_labels':
          lambda e: e['clean_labels'],
      'is_annotator_incorrect':
          lambda e: pi_utils.find_noisy_annotators(e, flatten_annotators=False),
      'annotator_label_if_incorrect':
          annotator_label_if_incorrect_encoding,
  }
  privileged_information_fn = pi_utils.get_privileged_information_fn(
      pi_subset=FLAGS.pi_subset.split(','), encoding_fn_dict=encoding_fn_dict)

  pi_shape = privileged_information_fn(dummy_example).shape[1:]
  pi_mc_samples = pi_utils.sample_pi_features(train_builder,
                                              privileged_information_fn,
                                              FLAGS.num_mc_samples)

  # The test/validation sets do not one-hot encode labels, so we process them
  # when used as PI.
  def one_hot_encoding_fn(labels):
    return tf.one_hot(tf.cast(labels, tf.int32), NUM_CLASSES, dtype=tf.float32)

  def test_annotator_label_if_incorrect_encoding(example):
    return pi_utils.annotator_label_if_incorrect_encoding_fn(
        example=example, label_encoding_fn=one_hot_encoding_fn)

  test_encoding_fn_dict = copy.deepcopy(encoding_fn_dict)
  test_encoding_fn_dict.update(
      {'clean_labels': lambda e: one_hot_encoding_fn(e['clean_labels'])})
  test_encoding_fn_dict.update({
      'clean_labels':
          lambda e: one_hot_encoding_fn(e['clean_labels']),
      'annotator_labels':
          lambda e: one_hot_encoding_fn(e['pi_features']['annotator_labels']),
      'annotator_label_if_incorrect':
          test_annotator_label_if_incorrect_encoding,
  })

  test_privileged_information_fn = pi_utils.get_privileged_information_fn(
      pi_subset=FLAGS.pi_subset.split(','),
      encoding_fn_dict=test_encoding_fn_dict)

  with strategy.scope():

    logging.info('Building Keras ResNet-50 TRAM model')

    model = ub.models.resnet50_pi_full_marginalization(  # type: ignore
        input_shape=IMAGE_SHAPE,
        pi_input_shape=pi_shape,
        width_multiplier=FLAGS.model_width_multiplier,
        pi_tower_width=FLAGS.pi_tower_width,
        num_mc_samples=FLAGS.num_mc_samples,
        num_classes=NUM_CLASSES)

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
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=1.0 - FLAGS.one_minus_momentum,
        nesterov=True)

    metrics = {'train/loss': tf.keras.metrics.Mean()}
    # `fm` stands for full marginalization.
    for setting in ['pi', 'fm']:
      metrics.update({
          f'train/negative_log_likelihood_{setting}': tf.keras.metrics.Mean(),
      })
      for noise_split in ['', '_clean', '_noisy']:
        metrics.update({
            f'train/accuracy_{setting}{noise_split}':
                tf.keras.metrics.SparseCategoricalAccuracy(),
            f'train/ece_{setting}{noise_split}':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        })

      metrics.update({
          f'test/negative_log_likelihood_{setting}':
              tf.keras.metrics.Mean(),
          f'test/accuracy_{setting}':
              tf.keras.metrics.SparseCategoricalAccuracy(),
          f'test/ece_{setting}':
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
      })

    if FLAGS.train_proportion < 1.0:
      for setting in ['pi', 'fm']:
        for validation_type in ['validation_clean', 'validation_noisy']:
          metrics.update({
              f'{validation_type}/negative_log_likelihood_{setting}':
                  tf.keras.metrics.Mean(),
              f'{validation_type}/accuracy_{setting}':
                  tf.keras.metrics.SparseCategoricalAccuracy(),
              f'{validation_type}/ece_{setting}':
                  rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
          })
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

  if not FLAGS.disable_reliability_estimation:
    mean_reliability = train_builder.mean_reliability
    logging.info(
        'Training on a slice of ImageNet-PI with mean reliability: %1.3f',
        mean_reliability)
    with summary_writer.as_default():
      tf.summary.scalar('mean_reliability', mean_reliability, step=0)

    objective = work_unit.get_measurement_series('mean_reliability')
    objective.create_measurement(mean_reliability, 0)

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      if FLAGS.use_annotator_labels:
        labels = inputs['pi_features'][
            'annotator_labels']  # We use the relabeled values to train.
        labels = pi_utils.flatten_annotator_axis(labels)
      else:
        labels = inputs['clean_labels']
        labels = pi_utils.repeat_across_annotators(
            labels,
            num_annotators_per_example=train_builder  # pytype: disable=attribute-error
            .num_annotators_per_example_and_step)
        labels = pi_utils.flatten_annotator_axis(labels)

      privileged_information = privileged_information_fn(inputs)

      noisy_idx = pi_utils.find_noisy_annotators(inputs)

      batch_size = tf.shape(images)[0]
      batched_pi_mc_samples = tf.tile(
          tf.expand_dims(pi_mc_samples, axis=0),
          multiples=[batch_size, 1, 1, 1])

      with tf.GradientTape() as tape:

        (logits_pi, probs_fm) = model(
            (images, privileged_information, batched_pi_mc_samples),
            training=True)
        if FLAGS.use_bfloat16:
          probs_fm = tf.cast(probs_fm, tf.float32)
          logits_pi = tf.cast(logits_pi, tf.float32)

        # Flatten the annotator axis.
        probs_fm = pi_utils.flatten_annotator_axis(probs_fm)
        logits_pi = pi_utils.flatten_annotator_axis(logits_pi)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels, probs_fm, from_logits=False))

        negative_log_likelihood_pi = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                labels, logits_pi, from_logits=True))

        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the weights. This excludes BN parameters and biases, but
          # pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood_pi + l2_loss
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs_pi = tf.nn.softmax(logits_pi)

      # We go back from one-hot labels to integers
      labels = tf.argmax(labels, axis=-1)

      stats = {
          'pi': {
              'outputs': logits_pi,
              'probs': probs_pi,
              'negative_log_likelihood': negative_log_likelihood_pi
          },
          'fm': {
              'outputs': probs_fm,
              'probs': probs_fm,
              'negative_log_likelihood': negative_log_likelihood
          },
      }

      for setting in ['pi', 'fm']:
        metrics[f'train/negative_log_likelihood_{setting}'].update_state(
            stats[setting]['negative_log_likelihood'])
        for noise_split in ['', '_clean', '_noisy']:
          sample_weight = None
          if noise_split == '_clean':
            sample_weight = 1.0 - noisy_idx
          elif noise_split == '_noisy':
            sample_weight = noisy_idx

          metrics[f'train/ece_{setting}{noise_split}'].add_batch(
              stats[setting]['probs'],
              label=labels,
              sample_weight=sample_weight)
          metrics[f'train/accuracy_{setting}{noise_split}'].update_state(
              labels, stats[setting]['outputs'], sample_weight=sample_weight)  # pylint: disable=redundant-keyword-arg

      metrics['train/loss'].update_state(loss)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def update_test_metrics(labels,
                          outputs,
                          metric_prefix='test',
                          metric_suffix='',
                          from_logits=True):
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, outputs, from_logits=from_logits))
    probs = tf.nn.softmax(outputs)
    metrics[metric_prefix + '/negative_log_likelihood' +
            metric_suffix].update_state(negative_log_likelihood)
    metrics[metric_prefix + '/accuracy' + metric_suffix].update_state(
        labels, probs)
    metrics[metric_prefix + '/ece' + metric_suffix].add_batch(
        probs, label=labels)

  @tf.function
  def test_step(metrics_prefix, iterator, steps_per_eval):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      if metrics_prefix == 'validation_noisy':
        labels = inputs['pi_features']['annotator_labels']
      else:
        labels = inputs['clean_labels']  # We use the original labels to test.
        labels = pi_utils.repeat_across_annotators(
            labels,
            num_annotators_per_example=test_builder  # pytype: disable=attribute-error
            .num_annotators_per_example_and_step)
      labels = tf.reshape(labels, [-1])
      privileged_information = test_privileged_information_fn(inputs)

      batch_size = tf.shape(images)[0]

      # We repeat the pi_mc_samples across examples:
      # (num_mc_samples, num_annotators_per_example, pi_length) ->
      # (batch_size, num_mc_samples, num_annotators_per_example, pi_length).
      batched_pi_mc_samples = tf.tile(
          tf.expand_dims(pi_mc_samples, axis=0),
          multiples=[batch_size, 1, 1, 1])

      (logits_pi, probs_fm) = model(
          (images, privileged_information, batched_pi_mc_samples),
          training=False)
      if FLAGS.use_bfloat16:
        probs_fm = tf.cast(probs_fm, tf.float32)
        logits_pi = tf.cast(logits_pi, tf.float32)

      # Flatten the annotator axis.
      probs_fm = pi_utils.flatten_annotator_axis(probs_fm)
      logits_pi = pi_utils.flatten_annotator_axis(logits_pi)

      update_test_metrics(
          labels,
          probs_fm,
          metric_prefix=metrics_prefix,
          metric_suffix='_fm',
          from_logits=False)
      update_test_metrics(
          labels,
          logits_pi,
          metric_prefix=metrics_prefix,
          metric_suffix='_pi',
          from_logits=True)

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
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)
    test_iterator = iter(test_dataset)
    logging.info('Starting to run eval at epoch: %s', epoch)
    if FLAGS.train_proportion < 1.0:
      validation_iterator = iter(validation_dataset)
      test_step(
          metrics_prefix='validation_noisy',
          iterator=validation_iterator,
          steps_per_eval=steps_per_validation_eval)
      # NOTE: We reinitialize the iterator to avoid OutOfRangeError.
      validation_iterator = iter(validation_dataset)
      test_step(
          metrics_prefix='validation_clean',
          iterator=validation_iterator,
          steps_per_eval=steps_per_validation_eval)
    test_start_time = time.time()
    test_step(
        metrics_prefix='test',
        iterator=test_iterator,
        steps_per_eval=steps_per_test_eval)
    ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

    logging.info(
        'Train Loss: %.4f, Accuracy (FM): %.2f%%, Accuracy (PI): %.2f%%',
        metrics['train/loss'].result(),
        metrics['train/accuracy_fm'].result() * 100,
        metrics['train/accuracy_pi'].result() * 100)
    if FLAGS.train_proportion < 1.0:
      logging.info(
          'Validation NLL (FM / noisy): %.4f, Accuracy (FM /noisy): %.2f%%',
          metrics['validation_noisy/negative_log_likelihood_fm'].result(),
          metrics['validation_noisy/accuracy_fm'].result() * 100)
      logging.info(
          'Validation NLL (PI / noisy): %.4f, Accuracy (PI / noisy): %.2f%%',
          metrics['validation_noisy/negative_log_likelihood_pi'].result(),
          metrics['validation_noisy/accuracy_pi'].result() * 100)
      logging.info(
          'Validation NLL (FM / clean): %.4f, Accuracy (FM /clean): %.2f%%',
          metrics['validation_clean/negative_log_likelihood_fm'].result(),
          metrics['validation_clean/accuracy_fm'].result() * 100)
      logging.info(
          'Validation NLL (PI / clean): %.4f, Accuracy (PI / clean): %.2f%%',
          metrics['validation_clean/negative_log_likelihood_pi'].result(),
          metrics['validation_clean/accuracy_pi'].result() * 100)

    logging.info('Test NLL (FM): %.4f, Accuracy (FM): %.2f%%',
                 metrics['test/negative_log_likelihood_fm'].result(),
                 metrics['test/accuracy_fm'].result() * 100)
    logging.info('Test NLL (PI): %.4f, Accuracy (PI): %.2f%%',
                 metrics['test/negative_log_likelihood_pi'].result(),
                 metrics['test/accuracy_pi'].result() * 100)

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
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
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
