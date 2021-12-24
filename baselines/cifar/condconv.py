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
import utils  # local file import from baselines.cifar
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_integer('num_dropout_samples_training', 1,
                     'Number of dropout samples for training.')
flags.DEFINE_bool(
    'filterwise_dropout', False, 'Dropout whole convolutional'
    'filters instead of individual values in the feature map.')
flags.DEFINE_bool(
    'residual_dropout', True,
    'Apply dropout only to the residual connections as proposed'
    'in the original paper.'
    'Otherwise dropout is applied after every layer.')

# Accelerator flags.
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')

# CondConv Flags
flags.DEFINE_integer('num_experts', 4, 'Number of experts to aggregate over.')
flags.DEFINE_bool('use_cond_dense', True, 'Whether to use CondDense.')
flags.DEFINE_bool('reduce_dense_outputs', False,
                  'Whether to aggregate on kernels or outputs.')
flags.DEFINE_enum(
    'loss',
    'gibbs_ce',
    enum_values=[
        'gibbs_ce', 'unweighted_gibbs_ce', 'moe', 'unweighted_moe', 'poe',
        'unweighted_poe'
    ],
    help='Choice of loss function/consensus algorithm')

flags.DEFINE_enum(
    'cond_placement',
    'dropout',
    enum_values=['dropout', 'all', 'none'],
    help='Where to place the CondConv layers.')

# TODO(ghassen): consider separating the CondDense and CondConv routing flags.
flags.DEFINE_enum(
    'routing_fn',
    'sigmoid',
    enum_values=[
        'sigmoid', 'softmax', 'noisy_softmax', 'onehot_top_k',
        'noisy_onehot_top_k', 'softmax_top_k', 'noisy_softmax_top_k'
    ],
    help='The choice of routing function for CondConv and CondDense.')
flags.DEFINE_bool('normalize_routing', False,
                  'Whether to normalize CondConv routing weights.')
flags.DEFINE_bool('normalize_dense_routing', False,
                  'Whether to normalize the final CondDense routing weights.')
flags.DEFINE_enum(
    'routing_pooling',
    'global_average',
    enum_values=[
        'global_average', 'global_max', 'average_8', 'max_8', 'flatten'
    ],
    help='Type of pooling to apply to the inputs of the routing functions.')
flags.DEFINE_integer('top_k', -1, 'The number of experts to select from.')
flags.DEFINE_integer('resnet_width_multiplier', 5,
                     'WideResNet width multiplier.')
# Redefining default values
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

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

  if FLAGS.dataset == 'cifar10':
    dataset_builder_class = ub.datasets.Cifar10Dataset
  else:
    dataset_builder_class = ub.datasets.Cifar100Dataset
  train_builder = dataset_builder_class(
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16,
      validation_percent=1. - FLAGS.train_proportion)
  train_dataset = train_builder.load(batch_size=batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)

  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:
    validation_builder = dataset_builder_class(
        data_dir=data_dir,
        split=tfds.Split.VALIDATION,
        use_bfloat16=FLAGS.use_bfloat16,
        validation_percent=1. - FLAGS.train_proportion)
    validation_dataset = validation_builder.load(batch_size=batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // batch_size

  clean_test_dataset_builder = dataset_builder_class(
      data_dir=data_dir,
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16)
  clean_test_dataset = clean_test_dataset_builder.load(
      batch_size=test_batch_size)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_dataset_builder.num_examples // batch_size
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
            severity=severity,
            split=tfds.Split.TEST,
            data_dir=data_dir).load(batch_size=batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')
    model = ub.models.wide_resnet_condconv(
        input_shape=(32, 32, 3),
        depth=28,
        width_multiplier=FLAGS.resnet_width_multiplier,
        num_classes=num_classes,
        num_experts=FLAGS.num_experts,
        per_core_batch_size=FLAGS.per_core_batch_size,
        use_cond_dense=FLAGS.use_cond_dense,
        reduce_dense_outputs=FLAGS.reduce_dense_outputs,
        cond_placement=FLAGS.cond_placement,
        routing_fn=FLAGS.routing_fn,
        normalize_routing=FLAGS.normalize_routing,
        normalize_dense_routing=FLAGS.normalize_dense_routing,
        top_k=FLAGS.top_k,
        routing_pooling=FLAGS.routing_pooling,
        l2=FLAGS.l2)
    # reuse_routing=FLAGS.reuse_routing,
    # shared_routing_type=FLAGS.shared_routing_type)
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
    if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
      for dataset_split in eval_dataset_splits:
        metrics.update({
            f'{dataset_split}/nll_poe':
                tf.keras.metrics.Mean(),
            f'{dataset_split}/nll_moe':
                tf.keras.metrics.Mean(),
            f'{dataset_split}/nll_unweighted_poe':
                tf.keras.metrics.Mean(),
            f'{dataset_split}/nll_unweighted_moe':
                tf.keras.metrics.Mean(),
            f'{dataset_split}/unweighted_gibbs_ce':
                tf.keras.metrics.Mean(),
            f'{dataset_split}/ece_unweighted_moe':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
            f'{dataset_split}/accuracy_unweighted_moe':
                tf.keras.metrics.SparseCategoricalAccuracy(),
            f'{dataset_split}/ece_poe':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
            f'{dataset_split}/accuracy_poe':
                tf.keras.metrics.SparseCategoricalAccuracy(),
            f'{dataset_split}/ece_unweighted_poe':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
            f'{dataset_split}/accuracy_unweighted_poe':
                tf.keras.metrics.SparseCategoricalAccuracy(),
        })
        for idx in range(FLAGS.num_experts):
          metrics[f'{dataset_split}/dense_routing_weight_{idx}'] = (
              tf.keras.metrics.Mean())
          metrics[f'{dataset_split}/dense_routing_weight_normalized_{idx}'] = (
              tf.keras.metrics.Mean())

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
          corrupt_metrics['test/nll_weighted_moe_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_weighted_moe_{}'.format(
              dataset_name)] = (
                  tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_weighted_moe_{}'.format(dataset_name)] = (
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

  def _process_3d_logits(logits, routing_weights, labels):
    routing_weights_3d = tf.expand_dims(routing_weights, axis=-1)
    weighted_logits = tf.math.reduce_mean(routing_weights_3d * logits, axis=1)
    unweighted_logits = tf.math.reduce_mean(logits, axis=1)

    probs = tf.nn.softmax(logits)
    unweighted_probs = tf.math.reduce_mean(probs, axis=1)
    weighted_probs = tf.math.reduce_sum(routing_weights_3d * probs, axis=1)

    labels_broadcasted = tf.tile(
        tf.reshape(labels, (-1, 1)), (1, FLAGS.num_experts))
    neg_log_likelihoods = tf.keras.losses.sparse_categorical_crossentropy(
        labels_broadcasted, logits, from_logits=True)
    unweighted_gibbs_ce = tf.math.reduce_mean(neg_log_likelihoods)
    weighted_gibbs_ce = tf.math.reduce_mean(
        tf.math.reduce_sum(routing_weights * neg_log_likelihoods, axis=1))
    return {
        'weighted_logits': weighted_logits,
        'unweighted_logits': unweighted_logits,
        'unweighted_probs': unweighted_probs,
        'weighted_probs': weighted_probs,
        'neg_log_likelihoods': neg_log_likelihoods,
        'unweighted_gibbs_ce': unweighted_gibbs_ce,
        'weighted_gibbs_ce': weighted_gibbs_ce
    }

  def _process_3d_logits_train(logits, routing_weights, labels):
    processing_results = _process_3d_logits(logits, routing_weights, labels)
    if FLAGS.loss == 'gibbs_ce':
      probs = processing_results['weighted_probs']
      negative_log_likelihood = processing_results['weighted_gibbs_ce']
    elif FLAGS.loss == 'unweighted_gibbs_ce':
      probs = processing_results['unweighted_probs']
      negative_log_likelihood = processing_results['unweighted_gibbs_ce']
    elif FLAGS.loss == 'moe':
      probs = processing_results['weighted_probs']
      negative_log_likelihood = tf.math.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, probs, from_logits=False))
    elif FLAGS.loss == 'unweighted_moe':
      probs = processing_results['unweighted_probs']
      negative_log_likelihood = tf.math.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, probs, from_logits=False))
    elif FLAGS.loss == 'poe':
      probs = tf.softmax(processing_results['weighted_logits'])
      negative_log_likelihood = tf.math.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, processing_results['weighted_logits'], from_logits=True))
    elif FLAGS.loss == 'unweighted_poe':
      probs = tf.softmax(processing_results['unweighted_logits'])
      negative_log_likelihood = tf.math.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(
              labels, processing_results['unweighted_logits'],
              from_logits=True))
    return probs, negative_log_likelihood

  def _process_3d_logits_test(routing_weights, logits, labels):
    processing_results = _process_3d_logits(logits, routing_weights, labels)
    nll_poe = tf.math.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, processing_results['weighted_logits'], from_logits=True))
    nll_unweighted_poe = tf.math.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, processing_results['unweighted_logits'], from_logits=True))
    nll_moe = tf.math.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, processing_results['weighted_probs'], from_logits=False))
    nll_unweighted_moe = tf.math.reduce_mean(
        tf.keras.losses.sparse_categorical_crossentropy(
            labels, processing_results['unweighted_probs'], from_logits=False))
    return {
        'nll_poe': nll_poe,
        'nll_moe': nll_moe,
        'nll_unweighted_poe': nll_unweighted_poe,
        'nll_unweighted_moe': nll_unweighted_moe,
        'unweighted_gibbs_ce': processing_results['unweighted_gibbs_ce'],
        'weighted_gibbs_ce': processing_results['weighted_gibbs_ce'],
        'weighted_probs': processing_results['weighted_probs'],
        'unweighted_probs': processing_results['unweighted_probs'],
        'weighted_logits': processing_results['weighted_logits'],
        'unweighted_logits': processing_results['unweighted_logits']
    }

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
        # if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
        if not isinstance(logits, (list, tuple)):
          raise ValueError('Logits are not a tuple.')
        # logits is a `Tensor` of shape [batch_size, num_experts, num_classes]
        logits, all_routing_weights = logits
        # routing_weights is a `Tensor` of shape [batch_size, num_experts]
        routing_weights = all_routing_weights[-1]
        if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
          probs, negative_log_likelihood = _process_3d_logits_train(
              logits, routing_weights, labels)
        else:
          probs = tf.nn.softmax(logits)
          # Prior to reduce_mean the NLLs are of the shape [batch, num_experts].
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  labels, logits, from_logits=True))

        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      logits = model(images, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      if not isinstance(logits, (list, tuple)):
        raise ValueError('Logits not a tuple')
      # logits is a `Tensor` of shape [batch_size, num_experts, num_classes]
      # routing_weights is a `Tensor` of shape [batch_size, num_experts]
      logits, all_routing_weights = logits
      routing_weights = all_routing_weights[-1]
      if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
        results = _process_3d_logits_test(routing_weights, logits, labels)
      else:
        probs = tf.nn.softmax(logits)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
          metrics[f'{dataset_split}/nll_poe'].update_state(results['nll_poe'])
          metrics[f'{dataset_split}/nll_moe'].update_state(results['nll_moe'])
          metrics[f'{dataset_split}/nll_unweighted_poe'].update_state(
              results['nll_unweighted_poe'])
          metrics[f'{dataset_split}/nll_unweighted_moe'].update_state(
              results['nll_unweighted_moe'])
          metrics[f'{dataset_split}/unweighted_gibbs_ce'].update_state(
              results['unweighted_gibbs_ce'])
          metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
              results['weighted_gibbs_ce'])
          metrics[f'{dataset_split}/ece'].add_batch(
              results['weighted_probs'], label=labels)
          metrics[f'{dataset_split}/accuracy'].update_state(
              labels, results['weighted_probs'])
          metrics[f'{dataset_split}/ece_unweighted_moe'].add_batch(
              results['unweighted_probs'], label=labels)
          metrics[f'{dataset_split}/accuracy_unweighted_moe'].update_state(
              labels, results['unweighted_probs'])
          metrics[f'{dataset_split}/ece_poe'].add_batch(
              results['weighted_logits'], label=labels)
          metrics[f'{dataset_split}/accuracy_poe'].update_state(
              labels, results['weighted_logits'])
          metrics[f'{dataset_split}/ece_unweighted_poe'].add_batch(
              results['unweighted_logits'], label=labels)
          metrics[f'{dataset_split}/accuracy_unweighted_poe'].update_state(
              labels, results['unweighted_logits'])
          # TODO(ghassen): summarize all routing weights not only last layer's.
          average_routing_weights = tf.math.reduce_mean(routing_weights, axis=0)
          routing_weights_sum = tf.math.reduce_sum(average_routing_weights)
          for idx in range(FLAGS.num_experts):
            metrics[f'{dataset_split}/dense_routing_weight_{idx}'].update_state(
                average_routing_weights[idx])
            key = f'{dataset_split}/dense_routing_weight_normalized_{idx}'
            metrics[key].update_state(
                average_routing_weights[idx] / routing_weights_sum)
          # TODO(ghassen): add more metrics for expert utilization,
          # load loss and importance/balance loss.
        else:
          metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
              negative_log_likelihood)
          metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
          metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
      else:
        # TODO(ghassen): figure out how to aggregate probs for the OOD case.
        if not FLAGS.reduce_dense_outputs and FLAGS.use_cond_dense:
          corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
              results['unweighted_gibbs_ce'])
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
              labels, results['unweighted_probs'])
          corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
              results['unweighted_probs'], label=labels)

          corrupt_metrics['test/nll_weighted_moe{}'.format(
              dataset_name)].update_state(results['weighted_gibbs_ce'])
          corrupt_metrics['test/accuracy_weighted_moe_{}'.format(
              dataset_name)].update_state(labels, results['weighted_probs'])
          corrupt_metrics['test/ece_weighted_moe{}'.format(
              dataset_name)].add_batch(results['weighted_probs'], label=labels)
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
      ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
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
        'dropout_rate': FLAGS.dropout_rate,
        'num_dropout_samples': FLAGS.num_dropout_samples,
        'num_dropout_samples_training': FLAGS.num_dropout_samples_training,
    })


if __name__ == '__main__':
  app.run(main)
