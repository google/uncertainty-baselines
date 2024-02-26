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

"""Hyper-BatchEnsemble Wide ResNet 28-10 on CIFAR-10 and CIFAR-100."""

import os
import pickle
import time

from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp
import uncertainty_baselines as ub
import utils  # local file import from baselines.cifar
from uncertainty_baselines.models import hyperbatchensemble_e_factory as e_factory
from uncertainty_baselines.models import HyperBatchEnsembleLambdaConfig as LambdaConfig
from uncertainty_baselines.models import wide_resnet_hyperbatchensemble
from tensorboard.plugins.hparams import api as hp

# General model, training, and evaluation flags
flags.DEFINE_boolean('restore_checkpoint', False,
                     'Start training from latest checkpoint.')

# Data flags
# Hyper-batchensemble flags
flags.DEFINE_bool('e_model_use_bias', False, 'Whether to use bias in e models.')
flags.DEFINE_float('min_l2_range', 1e-1, 'Min value of l2 range.')
flags.DEFINE_float('max_l2_range', 1e2, 'Max value of l2 range.')
flags.DEFINE_float(
    'e_body_hidden_units', 0, 'Number of hidden units used in e_models. '
    'If zero a linear model is used.')
flags.DEFINE_float(
    'l2_batchnorm', 15,
    'L2 reg. parameter for batchnorm layers (not tuned, constant).')
flags.DEFINE_float('ens_init_delta_bounds', 0.2,
                   'If ensemble is initialized with lambdas, this values'
                   'determines the spread of the log-uniform distribution'
                   'around it (used by ens_init: random, default).')
flags.DEFINE_float('init_emodels_stddev', 1e-4, 'Init e_models weights.')
flags.DEFINE_integer('ensemble_size', 4, 'Size of the ensemble.')
flags.DEFINE_float('lr_tuning', 1e-3, 'Learning rate for hparam tuning.')
flags.DEFINE_float('tau', 1e-3,
                   'Regularization of the entropy of the lambda distribution.')
flags.DEFINE_bool('use_gibbs_ce', True, 'Use Gibbs cross entropy for training.')
flags.DEFINE_bool(
    'sample_and_tune', True,
    'Whether to do tuning step with sampling from lambda distribution or not.')
flags.DEFINE_float('random_sign_init', -0.75,
                   'Use random sign init for fast weights.')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.5,
                   'fast weights lr multiplier to scale (alpha, gamma).')
flags.DEFINE_integer('tuning_warmup_epochs', 0,
                     'Number of epochs before starting tuning of lambdas')
flags.DEFINE_integer('tuning_every_x_step', 3,
                     'Do tunning step after x training steps.')
flags.DEFINE_bool('regularize_fast_weights', False,
                  'Whether to egularize fast weights in BatchEnsemble layers.')
flags.DEFINE_bool('fast_weights_eq_contraint', True, 'If true, set u,v:=r,s')
flags.DEFINE_integer(
    'num_eval_samples', 0, 'Number of samples taken for each batch-ens. member.'
    'If >=0, we take num_eval_samples + the mean of lambdas.'
    '(by default, notice that when = 0, predictions are with the mean only)'
    'If < 0, we take -num_eval_samples, without including the mean of lambdas.')
# Redefining default values
flags.FLAGS.set_default('lr_decay_epochs', ['100', '200', '225'])
flags.FLAGS.set_default('train_epochs', 250)
flags.FLAGS.set_default('train_proportion', 0.95)
FLAGS = flags.FLAGS


@tf.function
def log_uniform_sample(sample_size,
                       lambda_parameters):
  """Sample batch of lambdas distributed according log-unif(lower, upper)."""
  log_lower, log_upper = lambda_parameters
  ens_size = log_lower.shape[0]
  lambdas_dim = log_lower.shape[1]

  log_lower_ = tf.expand_dims(log_lower, 1)  # (ens_size, 1, lambdas_dim)
  log_upper_ = tf.expand_dims(log_upper, 1)  # (ens_size, 1, lambdas_dim)

  u = tf.random.uniform(shape=(ens_size, sample_size, lambdas_dim))
  return  tf.exp((log_upper_-log_lower_) * u + log_lower_)


@tf.function
def log_uniform_mean(lambda_parameters):
  """Mean of a log-uniform distribution."""
  # (see https://en.wikipedia.org/wiki/Reciprocal_distribution)
  log_lower, log_upper = lambda_parameters
  lower = tf.exp(log_lower)
  upper = tf.exp(log_upper)
  return (upper - lower) / (log_upper-log_lower)


@tf.function
def log_uniform_entropy(lambda_parameters):
  """Entropy of log-uniform(lower, upper)."""
  log_lower, log_upper = lambda_parameters
  r = log_upper - log_lower
  log_r = tf.math.log(r)
  # By definition, the entropy is given by:
  #  0.5/r*(tf.square(log_r + log_upper) - tf.square(log_r + log_lower))
  # which can be simplified into:
  entropy = 0.5 * (log_upper + log_lower) + log_r
  return tf.reduce_mean(entropy)


@tf.function
def ensemble_crossentropy(labels, logits, ensemble_size):
  """Return ensemble cross-entropy."""
  tile_logp = tf.nn.log_softmax(logits, axis=-1)
  # (1,ens_size*batch,n_classes)
  tile_logp = tf.expand_dims(tile_logp, 0)
  tile_logp = tf.concat(
      tf.split(tile_logp, ensemble_size, axis=1), 0)
  logp = tfp.math.reduce_logmeanexp(tile_logp, axis=0)

  mask = tf.stack([
      tf.range(len(labels), dtype=tf.int32),
      tf.cast(labels, dtype=tf.int32)], axis=1)
  return -tf.reduce_mean(tf.gather_nd(logp, mask))


@tf.function
def clip_lambda_parameters(lambda_parameters, lambdas_config):
  """Do cross-replica updates of lambda parameters."""
  # We want the projection to guarantee:
  #   log_min <= log_lower <= log_upper <= log_max
  # Since we manipulate expressions involving log(log_upper-log_lower), we add
  # some eps for numerical stability, to ensure log_upper-log_lower >= eps.
  # The eps > 0 is defined relative to the width of the interval.
  eps = 1e-6 * 0.5 * (lambdas_config.log_max - lambdas_config.log_min)
  log_lower, log_upper = lambda_parameters
  log_lower.assign(
      tf.clip_by_value(
          log_lower,
          clip_value_min=lambdas_config.log_min,
          clip_value_max=lambdas_config.log_max - 2*eps))
  log_upper.assign(
      tf.clip_by_value(
          log_upper,
          clip_value_min=log_lower + eps,
          clip_value_max=lambdas_config.log_max))


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

  per_core_batch_size = FLAGS.per_core_batch_size // FLAGS.ensemble_size
  batch_size = per_core_batch_size * FLAGS.num_cores
  check_bool = FLAGS.train_proportion > 0 and FLAGS.train_proportion <= 1
  assert check_bool, 'Proportion of train set has to meet 0 < prop <= 1.'

  drop_remainder_validation = True
  if not FLAGS.use_gpu:
    # This has to be True for TPU traing, otherwise the batchsize of images in
    # the validation set can't be determined by TPU compile.
    assert drop_remainder_validation, 'drop_remainder must be True in TPU mode.'

  validation_percent = 1 - FLAGS.train_proportion
  train_dataset = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      validation_percent=validation_percent).load(batch_size=batch_size)
  validation_dataset = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.VALIDATION,
      validation_percent=validation_percent,
      drop_remainder=drop_remainder_validation).load(batch_size=batch_size)
  validation_dataset = validation_dataset.repeat()
  clean_test_dataset = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST).load(batch_size=batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  validation_dataset = strategy.experimental_distribute_dataset(
      validation_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
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
            split=tfds.Split.TEST).load(batch_size=batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  ds_info = tfds.builder(FLAGS.dataset).info
  train_sample_size = ds_info.splits[
      'train'].num_examples * FLAGS.train_proportion
  steps_per_epoch = int(train_sample_size / batch_size)
  train_sample_size = int(train_sample_size)

  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  logging.info('Building Keras model.')
  depth = 28
  width = 10

  dict_ranges = {'min': FLAGS.min_l2_range, 'max': FLAGS.max_l2_range}
  ranges = [dict_ranges for _ in range(6)]  # 6 independent l2 parameters
  model_config = {
      'key_to_index': {
          'input_conv_l2_kernel': 0,
          'group_l2_kernel': 1,
          'group_1_l2_kernel': 2,
          'group_2_l2_kernel': 3,
          'dense_l2_kernel': 4,
          'dense_l2_bias': 5,
      },
      'ranges': ranges,
      'test': None
  }
  lambdas_config = LambdaConfig(model_config['ranges'],
                                model_config['key_to_index'])

  if FLAGS.e_body_hidden_units > 0:
    e_body_arch = '({},)'.format(FLAGS.e_body_hidden_units)
  else:
    e_body_arch = '()'
  e_shared_arch = '()'
  e_activation = 'tanh'
  filters_resnet = [16]
  for i in range(0, 3):  # 3 groups of blocks
    filters_resnet.extend([16 * width * 2**i] * 9)  # 9 layers in each block
  # e_head dim for conv2d is just the number of filters (only
  # kernel) and twice num of classes for the last dense layer (kernel + bias)
  e_head_dims = [x for x in filters_resnet] + [2 * num_classes]

  with strategy.scope():
    e_models = e_factory(
        lambdas_config.input_shape,
        e_head_dims=e_head_dims,
        e_body_arch=eval(e_body_arch),  # pylint: disable=eval-used
        e_shared_arch=eval(e_shared_arch),  # pylint: disable=eval-used
        activation=e_activation,
        use_bias=FLAGS.e_model_use_bias,
        e_head_init=FLAGS.init_emodels_stddev)

    model = wide_resnet_hyperbatchensemble(
        input_shape=ds_info.features['image'].shape,
        depth=depth,
        width_multiplier=width,
        num_classes=num_classes,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        config=lambdas_config,
        e_models=e_models,
        l2_batchnorm_layer=FLAGS.l2_batchnorm,
        regularize_fast_weights=FLAGS.regularize_fast_weights,
        fast_weights_eq_contraint=FLAGS.fast_weights_eq_contraint,
        version=2)

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # build hyper-batchensemble complete -------------------------

    # Initialize Lambda distributions for tuning
    lambdas_mean = tf.reduce_mean(
        log_uniform_mean(
            [lambdas_config.log_min, lambdas_config.log_max]))
    lambdas0 = tf.random.normal((FLAGS.ensemble_size, lambdas_config.dim),
                                lambdas_mean,
                                0.1 * FLAGS.ens_init_delta_bounds)
    lower0 = lambdas0 - tf.constant(FLAGS.ens_init_delta_bounds)
    lower0 = tf.maximum(lower0, 1e-8)
    upper0 = lambdas0 + tf.constant(FLAGS.ens_init_delta_bounds)

    log_lower = tf.Variable(tf.math.log(lower0))
    log_upper = tf.Variable(tf.math.log(upper0))
    lambda_parameters = [log_lower, log_upper]  # these variables are tuned
    clip_lambda_parameters(lambda_parameters, lambdas_config)

    # Optimizer settings to train model weights
    # Linearly scale learning rate and the decay epochs by vanilla settings.
    # Note: Here, we don't divide the epochs by 200 as for the other uncertainty
    # baselines.
    base_lr = FLAGS.base_learning_rate * batch_size / 128
    lr_decay_epochs = [int(l) for l in FLAGS.lr_decay_epochs]

    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)

    # tuner used for optimizing lambda_parameters
    tuner = tf.keras.optimizers.Adam(FLAGS.lr_tuning)

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
        'test/gibbs_nll': tf.keras.metrics.Mean(),
        'test/gibbs_accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/diversity': rm.metrics.AveragePairwiseDiversity(),
        'validation/loss': tf.keras.metrics.Mean(),
        'validation/loss_entropy': tf.keras.metrics.Mean(),
        'validation/loss_ce': tf.keras.metrics.Mean()
    }
    corrupt_metrics = {}

    for i in range(FLAGS.ensemble_size):
      metrics['test/nll_member_{}'.format(i)] = tf.keras.metrics.Mean()
      metrics['test/accuracy_member_{}'.format(i)] = (
          tf.keras.metrics.SparseCategoricalAccuracy())
    if FLAGS.corruptions_interval > 0:
      for intensity in range(1, 6):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    checkpoint = tf.train.Checkpoint(
        model=model, lambda_parameters=lambda_parameters, optimizer=optimizer)

    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint and FLAGS.restore_checkpoint:
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
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])

      # generate lambdas
      lambdas = log_uniform_sample(
          per_core_batch_size, lambda_parameters)
      lambdas = tf.reshape(
          lambdas,
          (FLAGS.ensemble_size * per_core_batch_size, lambdas_config.dim))

      with tf.GradientTape() as tape:
        logits = model([images, lambdas], training=True)

        if FLAGS.use_gibbs_ce:
          # Average of single model CEs
          # tiling of labels should be only done for Gibbs CE loss
          labels = tf.tile(labels, [FLAGS.ensemble_size])
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                              logits,
                                                              from_logits=True))
        else:
          # Ensemble CE uses no tiling of the labels
          negative_log_likelihood = ensemble_crossentropy(
              labels, logits, FLAGS.ensemble_size)
        # Note: Divide l2_loss by sample_size (this differs from uncertainty_
        # baselines implementation.)
        l2_loss = sum(model.losses) / train_sample_size
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate for fast weights.
      grads_and_vars = []
      for grad, var in zip(grads, model.trainable_variables):
        if (('alpha' in var.name or 'gamma' in var.name) and
            'batch_norm' not in var.name):
          grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
        else:
          grads_and_vars.append((grad, var))
      optimizer.apply_gradients(grads_and_vars)

      probs = tf.nn.softmax(logits)
      per_probs = tf.split(
          probs, num_or_size_splits=FLAGS.ensemble_size, axis=0)
      per_probs_stacked = tf.stack(per_probs, axis=0)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)
      metrics['train/diversity'].add_batch(per_probs_stacked)

      if grads_and_vars:
        grads, _ = zip(*grads_and_vars)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def tuning_step(iterator):
    """Tuning StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])

      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(lambda_parameters)

        # sample lambdas
        if FLAGS.sample_and_tune:
          lambdas = log_uniform_sample(
              per_core_batch_size, lambda_parameters)
        else:
          lambdas = log_uniform_mean(lambda_parameters)
          lambdas = tf.repeat(lambdas, per_core_batch_size, axis=0)
        lambdas = tf.reshape(lambdas,
                             (FLAGS.ensemble_size * per_core_batch_size,
                              lambdas_config.dim))
        # ensemble CE
        logits = model([images, lambdas], training=False)
        ce = ensemble_crossentropy(labels, logits, FLAGS.ensemble_size)
        # entropy penalty for lambda distribution
        entropy = FLAGS.tau * log_uniform_entropy(
            lambda_parameters)
        loss = ce - entropy
        scaled_loss = loss / strategy.num_replicas_in_sync

      gradients = tape.gradient(loss, lambda_parameters)
      tuner.apply_gradients(zip(gradients, lambda_parameters))

      metrics['validation/loss_ce'].update_state(ce /
                                                 strategy.num_replicas_in_sync)
      metrics['validation/loss_entropy'].update_state(
          entropy / strategy.num_replicas_in_sync)
      metrics['validation/loss'].update_state(scaled_loss)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name, num_eval_samples=0):
    """Evaluation StepFn."""

    n_samples = num_eval_samples if num_eval_samples >= 0 else -num_eval_samples
    if num_eval_samples >= 0:
      # the +1 accounts for the fact that we add the mean of lambdas
      ensemble_size = FLAGS.ensemble_size * (1 + n_samples)
    else:
      ensemble_size = FLAGS.ensemble_size * n_samples

    def step_fn(inputs):
      """Per-Replica StepFn."""
      # Note that we don't use tf.tile for labels here
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(images, [ensemble_size, 1, 1, 1])

      # get lambdas
      samples = log_uniform_sample(n_samples, lambda_parameters)
      if num_eval_samples >= 0:
        lambdas = log_uniform_mean(lambda_parameters)
        lambdas = tf.expand_dims(lambdas, 1)
        lambdas = tf.concat((lambdas, samples), 1)
      else:
        lambdas = samples

      # lambdas with shape (ens size, samples, dim of lambdas)
      rep_lambdas = tf.repeat(lambdas, per_core_batch_size, axis=1)
      rep_lambdas = tf.reshape(rep_lambdas,
                               (ensemble_size * per_core_batch_size, -1))

      # eval on testsets
      logits = model([images, rep_lambdas], training=False)
      probs = tf.nn.softmax(logits)
      per_probs = tf.split(probs,
                           num_or_size_splits=ensemble_size,
                           axis=0)

      # per member performance and gibbs performance (average per member perf)
      if dataset_name == 'clean':
        for i in range(FLAGS.ensemble_size):
          # we record the first sample of lambdas per batch-ens member
          first_member_index = i * (ensemble_size // FLAGS.ensemble_size)
          member_probs = per_probs[first_member_index]
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)

        labels_tile = tf.tile(labels, [ensemble_size])
        metrics['test/gibbs_nll'].update_state(tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels_tile,
                                                            logits,
                                                            from_logits=True)))
        metrics['test/gibbs_accuracy'].update_state(labels_tile, probs)

      # ensemble performance
      negative_log_likelihood = ensemble_crossentropy(labels, logits,
                                                      ensemble_size)
      probs = tf.reduce_mean(per_probs, axis=0)
      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

      if dataset_name == 'clean':
        per_probs_stacked = tf.stack(per_probs, axis=0)
        metrics['test/diversity'].add_batch(per_probs_stacked)

    strategy.run(step_fn, args=(next(iterator),))

  logging.info(
      '--- Starting training using %d examples. ---', train_sample_size)
  train_iterator = iter(train_dataset)
  validation_iterator = iter(validation_dataset)
  start_time = time.time()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      train_step(train_iterator)
      do_tuning = (epoch >= FLAGS.tuning_warmup_epochs)
      if do_tuning and ((step + 1) % FLAGS.tuning_every_x_step == 0):
        tuning_step(validation_iterator)
        # clip lambda parameters if outside of range
        clip_lambda_parameters(lambda_parameters, lambdas_config)

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

    # evaluate on test data
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
        test_step(test_iterator, dataset_name, FLAGS.num_eval_samples)
      logging.info('Done with testing on %s', dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types)
    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['train/loss'].result(),
                 metrics['train/accuracy'].result() * 100)
    logging.info('Validation Loss: %.4f, CE: %.4f, Entropy: %.4f',
                 metrics['validation/loss'].result(),
                 metrics['validation/loss_ce'].result(),
                 metrics['validation/loss_entropy'].result())
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['test/negative_log_likelihood'].result(),
                 metrics['test/accuracy'].result() * 100)
    for i in range(FLAGS.ensemble_size):
      logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%',
                   i, metrics['test/nll_member_{}'.format(i)].result(),
                   metrics['test/accuracy_member_{}'.format(i)].result() * 100)

    total_results = {name: metric.result() for name, metric in metrics.items()}
    total_results.update(
        {name: metric.result() for name, metric in corrupt_metrics.items()})
    total_results.update(corrupt_results)
    # Results from Robustness Metrics themselves return a dict, so flatten them.
    total_results = utils.flatten_dictionary(total_results)
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    # save checkpoint and lambdas config
    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(
          os.path.join(FLAGS.output_dir, 'checkpoint'))
      lambdas_cf = lambdas_config.get_config()
      filepath = os.path.join(FLAGS.output_dir, 'lambdas_config.p')
      with tf.io.gfile.GFile(filepath, 'wb') as fp:
        pickle.dump(lambdas_cf, fp, protocol=pickle.HIGHEST_PROTOCOL)
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
