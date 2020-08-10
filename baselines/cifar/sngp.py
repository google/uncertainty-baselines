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

"""Wide ResNet 28-10 with SNGP on CIFAR-10.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## Combining with MC Dropout:

As a single-model method, SNGP can be combined with other classic
uncertainty techniques (e.g., Monte Carlo dropout, deep ensemble) to further
improve performance.

This script supports adding Monte Carlo dropout to
SNGP by setting `use_mc_dropout=True`, setting `num_dropout_samples=10`
(or any integer larger than 1). Additionally we recommend adjust
`gp_mean_field_factor` slightly, since averaging already calibrated
individual models (in this case single SNGPs) can sometimes lead to
under-confidence [3].

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
[3]: Rahul Rahaman, Alexandre H. Thiery. Uncertainty Quantification and Deep
     Ensembles.  _arXiv preprint arXiv:2007.08792_, 2020.
     https://arxiv.org/abs/2007.08792
"""

import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
from edward2.experimental import sngp

import tensorflow as tf
import tensorflow_datasets as tfds
import utils  # local file import

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('base_learning_rate', 0.04,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 3e-4, 'L2 regularization coefficient.')

# Dropout flags
flags.DEFINE_bool('use_mc_dropout', False,
                  'Whether to use Monte Carlo dropout for the hidden layers.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_integer('num_dropout_samples_training', 1,
                     'Number of dropout samples for training.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')

flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', 250,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer('checkpoint_interval', 250,
                     'Number of epochs between saving checkpoints. Use -1 to '
                     'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

# SNGP flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')

# Spectral normalization flags.
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', 6.,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_input_dim', 128,
    'The dimension to reduce the neural network input to for the GP layer '
    '(via random Gaussian projection which preserves distance by the '
    ' Johnson-Lindenstrauss lemma). If -1 the no dimension reduction.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used to for the approximation ')
flags.DEFINE_bool(
    'gp_input_normalization', True,
    'Whether to normalize the input using LayerNorm for GP layer.'
    'This is similar to automatic relevance determination (ARD) in the classic '
    'GP learning.')
flags.DEFINE_float('gp_cov_ridge_penalty', 1e-3,
                   'The Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', 0.999,
    'The discount factor to compute the moving average of '
    'precision matrix.')
flags.DEFINE_float(
    'gp_mean_field_factor', 0.001,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')
FLAGS = flags.FLAGS

# pylint: disable=invalid-name

BatchNormalization = functools.partial(
    tf.keras.layers.BatchNormalization,
    epsilon=1e-5,  # using epsilon and momentum defaults from Torch
    momentum=0.9)

GaussianProcess = functools.partial(
    sngp.RandomFeatureGaussianProcess,
    num_inducing=FLAGS.gp_hidden_dim,
    gp_kernel_scale=FLAGS.gp_scale,
    gp_output_bias=FLAGS.gp_bias,
    normalize_input=FLAGS.gp_input_normalization,
    gp_cov_momentum=FLAGS.gp_cov_discount_factor,
    gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty)

Conv2DBase = functools.partial(
    tf.keras.layers.Conv2D,
    kernel_size=3,
    padding='same',
    use_bias=False,
    kernel_initializer='he_normal')


def Conv2DNormed(*conv_args, **conv_kwargs):
  conv_layer = Conv2DBase(*conv_args, **conv_kwargs)
  return sngp.SpectralNormalizationConv2D(
      conv_layer,
      iteration=FLAGS.spec_norm_iteration,
      norm_multiplier=FLAGS.spec_norm_bound)


Conv2D = Conv2DNormed if FLAGS.use_spec_norm else Conv2DBase
# pylint: enable=invalid-name


def apply_dropout(inputs, dropout_rate, use_mc_dropout):
  """Applies a filter-wise dropout layer to the inputs."""
  logging.info('apply_dropout input shape %s', inputs.shape)
  dropout_layer = tf.keras.layers.Dropout(
      dropout_rate, noise_shape=[inputs.shape[0], 1, 1, inputs.shape[3]])

  if use_mc_dropout:
    return dropout_layer(inputs, training=True)

  return dropout_layer(inputs)


def basic_block(inputs, filters, strides, l2, dropout_rate, use_mc_dropout):
  """Basic residual block of two 3x3 convs.

  Args:
    inputs: tf.Tensor.
    filters: Number of filters for Conv2D.
    strides: Stride dimensions for Conv2D.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    use_mc_dropout: Whether to apply Monte Carlo dropout.

  Returns:
    tf.Tensor.
  """
  x = inputs
  y = inputs
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = apply_dropout(y, dropout_rate, use_mc_dropout)

  y = Conv2D(filters,
             strides=strides,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(y)
  y = tf.keras.layers.Activation('relu')(y)
  y = apply_dropout(y, dropout_rate, use_mc_dropout)

  y = Conv2D(filters,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(y)
  if not x.shape.is_compatible_with(y.shape):
    x = Conv2D(filters,
               kernel_size=1,
               strides=strides,
               kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    y = apply_dropout(y, dropout_rate, use_mc_dropout)

  x = tf.keras.layers.add([x, y])
  return x


def group(inputs, filters, strides, num_blocks, l2, dropout_rate,
          use_mc_dropout):
  """Group of residual blocks."""
  x = basic_block(inputs,
                  filters=filters,
                  strides=strides,
                  l2=l2,
                  dropout_rate=dropout_rate,
                  use_mc_dropout=use_mc_dropout)
  for _ in range(num_blocks - 1):
    x = basic_block(x,
                    filters=filters,
                    strides=1,
                    l2=l2,
                    dropout_rate=dropout_rate,
                    use_mc_dropout=use_mc_dropout)
  return x


def wide_resnet(input_shape, batch_size, depth, width_multiplier, num_classes,
                l2, dropout_rate, use_mc_dropout, gp_input_dim, use_gp_layer):
  """Builds Wide ResNet.

  Following Zagoruyko and Komodakis (2016), it accepts a width multiplier on the
  number of filters. Using three groups of residual blocks, the network maps
  spatial features of size 32x32 -> 16x16 -> 8x8.

  Args:
    input_shape: tf.Tensor.
    batch_size: The batch size of the input layer. Required by the spectral
      normalization.
    depth: Total number of convolutional layers. "n" in WRN-n-k. It differs from
      He et al. (2015)'s notation which uses the maximum depth of the network
      counting non-conv layers like dense.
    width_multiplier: Integer to multiply the number of typical filters by. "k"
      in WRN-n-k.
    num_classes: Number of output classes.
    l2: L2 regularization coefficient.
    dropout_rate: Dropout rate.
    use_mc_dropout: Whether to apply Monte Carlo dropout.
    gp_input_dim: The input dimension to GP layer.
    use_gp_layer: Whether to use Gaussian process layer as the output layer.

  Returns:
    tf.keras.Model.
  """
  if (depth - 4) % 6 != 0:
    raise ValueError('depth should be 6n+4 (e.g., 16, 22, 28, 40).')
  num_blocks = (depth - 4) // 6
  inputs = tf.keras.layers.Input(shape=input_shape, batch_size=batch_size)

  x = Conv2D(16,
             strides=1,
             kernel_regularizer=tf.keras.regularizers.l2(l2))(inputs)
  x = apply_dropout(x, dropout_rate, use_mc_dropout)

  x = group(x,
            filters=16 * width_multiplier,
            strides=1,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout)
  x = group(x,
            filters=32 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout)
  x = group(x,
            filters=64 * width_multiplier,
            strides=2,
            num_blocks=num_blocks,
            l2=l2,
            dropout_rate=dropout_rate,
            use_mc_dropout=use_mc_dropout)
  x = BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(l2),
                         gamma_regularizer=tf.keras.regularizers.l2(l2))(x)
  x = tf.keras.layers.Activation('relu')(x)
  x = tf.keras.layers.AveragePooling2D(pool_size=8)(x)
  x = tf.keras.layers.Flatten()(x)

  if use_gp_layer:
    # Uses random projection to reduce the input dimension of the GP layer.
    if gp_input_dim > 0:
      x = tf.keras.layers.Dense(
          gp_input_dim,
          kernel_initializer='random_normal',
          use_bias=False,
          trainable=False)(x)
    logits, covmat = GaussianProcess(num_classes)(x)
  else:
    logits = tf.keras.layers.Dense(
        num_classes,
        kernel_initializer='he_normal',
        kernel_regularizer=tf.keras.regularizers.l2(l2),
        bias_regularizer=tf.keras.regularizers.l2(l2))(x)
    covmat = tf.eye(batch_size)

  return tf.keras.Model(inputs=inputs, outputs=[logits, covmat])


def mean_field_logits(logits, covmat, mean_field_factor=1.):
  """Adjust the predictive logits so its softmax approximates posterior mean."""
  logits_scale = tf.sqrt(1. + tf.linalg.diag_part(covmat) * mean_field_factor)
  if mean_field_factor > 0:
    logits = logits / tf.expand_dims(logits_scale, axis=-1)

  return logits


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
    strategy = tf.distribute.experimental.TPUStrategy(resolver)

  train_input_fn = utils.load_input_fn(
      split=tfds.Split.TRAIN,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size //
      FLAGS.num_dropout_samples_training,
      use_bfloat16=FLAGS.use_bfloat16)
  clean_test_input_fn = utils.load_input_fn(
      split=tfds.Split.TEST,
      name=FLAGS.dataset,
      batch_size=FLAGS.per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  train_dataset = strategy.experimental_distribute_datasets_from_function(
      train_input_fn)
  test_datasets = {
      'clean': strategy.experimental_distribute_datasets_from_function(
          clean_test_input_fn),
  }
  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar10':
      load_c_input_fn = utils.load_cifar10_c_input_fn
    else:
      load_c_input_fn = functools.partial(utils.load_cifar100_c_input_fn,
                                          path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        input_fn = load_c_input_fn(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=FLAGS.per_core_batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_datasets_from_function(input_fn))

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores
                // FLAGS.num_dropout_samples_training)
  steps_per_epoch = ds_info.splits['train'].num_examples // batch_size
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  steps_per_eval = ds_info.splits['test'].num_examples // test_batch_size
  num_classes = ds_info.features['label'].num_classes

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model')
    if FLAGS.use_spec_norm:
      logging.info('Use Spectral Normalization with norm bound %.2f',
                   FLAGS.spec_norm_bound)
    if FLAGS.use_gp_layer:
      logging.info('Use GP layer with hidden units %d', FLAGS.gp_hidden_dim)

    model = wide_resnet(input_shape=ds_info.features['image'].shape,
                        batch_size=batch_size,
                        depth=28,
                        width_multiplier=10,
                        num_classes=num_classes,
                        l2=FLAGS.l2,
                        dropout_rate=FLAGS.dropout_rate,
                        use_mc_dropout=FLAGS.use_mc_dropout,
                        gp_input_dim=FLAGS.gp_input_dim,
                        use_gp_layer=FLAGS.use_gp_layer)
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
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': ed.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': ed.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
        'test/stddev': tf.keras.metrics.Mean(),
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
              ed.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
          corrupt_metrics['test/stddev_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())

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
      images, labels = inputs
      images = tf.tile(images, [FLAGS.num_dropout_samples_training, 1, 1, 1])
      labels = tf.tile(labels, [FLAGS.num_dropout_samples_training])

      with tf.GradientTape() as tape:
        logits, _ = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs

      logits_list = []
      stddev_list = []
      for _ in range(FLAGS.num_dropout_samples):
        logits, covmat = model(images, training=False)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        logits = mean_field_logits(
            logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
        stddev = tf.sqrt(tf.linalg.diag_part(covmat))

        stddev_list.append(stddev)
        logits_list.append(logits)

      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      stddev_list = tf.stack(stddev_list, axis=0)

      stddev = tf.reduce_mean(stddev_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_dropout_samples, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_dropout_samples)))

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
        metrics['test/stddev'].update_state(stddev)
      else:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/stddev_{}'.format(dataset_name)].update_state(
            stddev)

    strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

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
        test_start_time = time.time()
        test_step(test_iterator, dataset_name)
        ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
        metrics['test/ms_per_example'].update_state(ms_per_example)

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
