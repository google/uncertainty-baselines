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

"""Wide ResNet 28-10 with SNGP on CIFAR-10.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## Reproducibility Instruction for CIFAR-100:

When running this script on CIFAR-100, set base_learning_rate=0.08 and
gp_mean_field_factor=12.5 to reproduce the benchmark result.

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
[4]: Hendrycks, Dan et al. AugMix: A Simple Data Processing Method to Improve
     Robustness and Uncertainty. In _International Conference on Learning
     Representations_, 2020.
     https://arxiv.org/abs/1912.02781
[5]: Zhang, Hongyi et al. mixup: Beyond Empirical Risk Minimization. In
     _International Conference on Learning Representations_, 2018.
     https://arxiv.org/abs/1710.09412
"""

import functools
import os
import time
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import utils  # local file import
import uncertainty_metrics as um

flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_float('base_learning_rate', 0.05,
                   'Base learning rate when total batch size is 128. It is '
                   'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_integer('lr_warmup_epochs', 1,
                     'Number of epochs for a linear warmup to the initial '
                     'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['60', '120', '160'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('l2', 3e-4, 'L2 regularization coefficient.')

flags.DEFINE_float('train_proportion', 1.,
                   'Only a fraction (between 0 and 1) of the train set is used '
                   'for training. The remainder can be used for validation.')
flags.DEFINE_enum('dataset', 'cifar10',
                  enum_values=['cifar10', 'cifar100'],
                  help='Dataset.')
flags.DEFINE_string('cifar100_c_path', None,
                    'Path to the TFRecords files for CIFAR-100-C. Only valid '
                    '(and required) if dataset is cifar100 and corruptions.')
flags.DEFINE_integer('corruptions_interval', -1,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer(
    'checkpoint_interval', -1,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/cifar', 'Output directory.')
flags.DEFINE_integer('train_epochs', 250, 'Number of training epochs.')

# Data Augmentation flags.
flags.DEFINE_bool('augmix', False,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer('aug_count', 1,
                     'Number of augmentation operations in AugMix to perform '
                     'on the input image. In the simgle model context, it'
                     'should be 1. In the ensembles context, it should be'
                     'ensemble_size if we perform random_augment only; It'
                     'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer('augmix_depth', -1,
                     'Augmix depth, -1 meaning sampled depth. This corresponds'
                     'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer('augmix_width', 3,
                     'Augmix width. This corresponds to the k in line 5 in the'
                     'Algorithm box in [4].')
flags.DEFINE_float('mixup_alpha', 0., 'Mixup hyperparameter, 0. to diable.')

# Dropout flags
flags.DEFINE_bool('use_mc_dropout', False,
                  'Whether to use Monte Carlo dropout for the hidden layers.')
flags.DEFINE_bool('use_filterwise_dropout', True,
                  'Whether to use filterwise dropout for the hidden layers.')
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate.')
flags.DEFINE_integer('num_dropout_samples', 1,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_integer('num_dropout_samples_training', 1,
                     'Number of dropout samples for training.')

# SNGP flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', 6.,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_input_dim', 128,
    'The dimension to reduce the neural network input for the GP layer '
    '(via random Gaussian projection which preserves distance by the '
    ' Johnson-Lindenstrauss lemma). If -1, no dimension reduction.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', True,
    'Whether to normalize the input using LayerNorm for GP layer.'
    'This is similar to automatic relevance determination (ARD) in the classic '
    'GP learning.')
flags.DEFINE_string(
    'gp_random_feature_type', 'orf',
    'The type of random feature to use. One of "rff" (random fourier feature), '
    '"orf" (orthogonal random feature).')
flags.DEFINE_float('gp_cov_ridge_penalty', 1.,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', -1.,
    'The discount factor to compute the moving average of precision matrix'
    'across epochs. If -1 then compute the exact precision matrix within the '
    'latest epoch.')
flags.DEFINE_float(
    'gp_mean_field_factor', 25.,
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
    strategy = tf.distribute.TPUStrategy(resolver)

  batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores
                // FLAGS.num_dropout_samples_training)
  test_batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  num_classes = 10 if FLAGS.dataset == 'cifar10' else 100

  aug_params = {
      'augmix': FLAGS.augmix,
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth,
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width,
      'ensemble_size': 1,
      'mixup_alpha': FLAGS.mixup_alpha,
  }
  validation_proportion = 1. - FLAGS.train_proportion
  use_validation_set = validation_proportion > 0.
  if FLAGS.dataset == 'cifar10':
    dataset_builder_class = ub.datasets.Cifar10Dataset
  else:
    dataset_builder_class = ub.datasets.Cifar100Dataset
  train_dataset_builder = dataset_builder_class(
      split=tfds.Split.TRAIN,
      use_bfloat16=FLAGS.use_bfloat16,
      aug_params=aug_params,
      validation_percent=validation_proportion)
  train_dataset = train_dataset_builder.load(batch_size=batch_size)
  train_sample_size = train_dataset_builder.num_examples
  if validation_proportion > 0.:
    validation_dataset_builder = dataset_builder_class(
        split=tfds.Split.VALIDATION,
        use_bfloat16=FLAGS.use_bfloat16,
        aug_params=aug_params,
        validation_percent=validation_proportion)
    validation_dataset = validation_dataset_builder.load(batch_size=batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    val_sample_size = validation_dataset_builder.num_examples
    steps_per_val = steps_per_epoch = int(val_sample_size / batch_size)
  clean_test_dataset_builder = dataset_builder_class(
      split=tfds.Split.TEST,
      use_bfloat16=FLAGS.use_bfloat16)
  clean_test_dataset = clean_test_dataset_builder.load(
      batch_size=test_batch_size)

  steps_per_epoch = int(train_sample_size / batch_size)
  steps_per_epoch = train_dataset_builder.num_examples // batch_size
  steps_per_eval = clean_test_dataset_builder.num_examples // batch_size
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar10':
      load_c_dataset = utils.load_cifar10_c
    else:
      load_c_dataset = functools.partial(utils.load_cifar100_c,
                                         path=FLAGS.cifar100_c_path)
    corruption_types, max_intensity = utils.load_corrupted_test_info(
        FLAGS.dataset)
    for corruption in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset = load_c_dataset(
            corruption_name=corruption,
            corruption_intensity=intensity,
            batch_size=test_batch_size,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets['{0}_{1}'.format(corruption, intensity)] = (
            strategy.experimental_distribute_dataset(dataset))

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

    model = ub.models.wide_resnet_sngp(
        input_shape=(32, 32, 3),
        batch_size=batch_size,
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        l2=FLAGS.l2,
        use_mc_dropout=FLAGS.use_mc_dropout,
        use_filterwise_dropout=FLAGS.use_filterwise_dropout,
        dropout_rate=FLAGS.dropout_rate,
        use_gp_layer=FLAGS.use_gp_layer,
        gp_input_dim=FLAGS.gp_input_dim,
        gp_hidden_dim=FLAGS.gp_hidden_dim,
        gp_scale=FLAGS.gp_scale,
        gp_bias=FLAGS.gp_bias,
        gp_input_normalization=FLAGS.gp_input_normalization,
        gp_random_feature_type=FLAGS.gp_random_feature_type,
        gp_cov_discount_factor=FLAGS.gp_cov_discount_factor,
        gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty,
        use_spec_norm=FLAGS.use_spec_norm,
        spec_norm_iteration=FLAGS.spec_norm_iteration,
        spec_norm_bound=FLAGS.spec_norm_bound)
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
    optimizer = tf.keras.optimizers.SGD(lr_schedule,
                                        momentum=0.9,
                                        nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/stddev': tf.keras.metrics.Mean(),
    }
    if use_validation_set:
      metrics.update({
          'val/negative_log_likelihood': tf.keras.metrics.Mean(),
          'val/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
          'val/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
          'val/stddev': tf.keras.metrics.Mean(),
      })
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
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
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
  def train_step(iterator, step):
    """Training StepFn."""

    def step_fn(inputs, step):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      if tf.equal(step, 0) and FLAGS.gp_cov_discount_factor < 0:
        # Resetting covaraince estimator at the begining of a new epoch.
        model.layers[-1].reset_covariance_matrix()

      if FLAGS.augmix and FLAGS.aug_count >= 1:
        # Index 0 at augmix preprocessing is the unperturbed image.
        images = images[:, 1, ...]
        # This is for the case of combining AugMix and Mixup.
        if FLAGS.mixup_alpha > 0:
          labels = tf.split(labels, FLAGS.aug_count + 1, axis=0)[1]
      images = tf.tile(images, [FLAGS.num_dropout_samples_training, 1, 1, 1])
      if FLAGS.mixup_alpha > 0:
        labels = tf.tile(labels, [FLAGS.num_dropout_samples_training, 1])
      else:
        labels = tf.tile(labels, [FLAGS.num_dropout_samples_training])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
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

        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      if FLAGS.mixup_alpha > 0:
        labels = tf.argmax(labels, axis=-1)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    strategy.run(step_fn, args=(next(iterator), step))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      logits_list = []
      stddev_list = []
      for _ in range(FLAGS.num_dropout_samples):
        logits = model(images, training=False)
        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract both
          logits, covmat = logits
        else:
          covmat = tf.eye(FLAGS.per_core_batch_size)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
        logits = ed.layers.utils.mean_field_logits(
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
      elif dataset_name == 'val':
        metrics['val/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['val/accuracy'].update_state(labels, probs)
        metrics['val/ece'].update_state(labels, probs)
        metrics['val/stddev'].update_state(stddev)
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

  step_variable = tf.Variable(0, dtype=tf.int32)
  train_iterator = iter(train_dataset)
  start_time = time.time()

  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    for step in range(steps_per_epoch):
      step_variable.assign(step)
      # Pass `step` as a tf.Variable to train_step to prevent the tf.function
      # train_step() re-compiling itself at each function call.
      train_step(train_iterator, step_variable)

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
    if use_validation_set:
      datasets_to_evaluate['val'] = validation_dataset
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      steps_per_eval = steps_per_val if dataset_name == 'val' else steps_per_eval
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
    if use_validation_set:
      logging.info('Val NLL: %.4f, Accuracy: %.2f%%',
                   metrics['val/negative_log_likelihood'].result(),
                   metrics['val/accuracy'].result() * 100)
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

  final_save_name = os.path.join(FLAGS.output_dir, 'model')
  model.save(final_save_name)
  logging.info('Saved model to %s', final_save_name)

if __name__ == '__main__':
  app.run(main)
