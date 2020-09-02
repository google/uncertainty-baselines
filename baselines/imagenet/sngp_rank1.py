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

"""ResNet-50 on ImageNet using SNGP + BatchEnsemble.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to hidden weights, and then replace the dense output layer with
a Gaussian process.

## Combining with BatchEnsemble [3]:

BatchEnsemble layers capture a multimodal representation of NN weight
distributions that has proven to improve both the calibration and OOD
generalization of single (deterministic) models.

## References:
[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
[3]: Yeming Wen, Dustin Tran, Jimmy Ba. BatchEnsemble: an Alternative
     Approach to Efficient Ensemble and Lifelong Learning.
     _arXiv preprint arXiv:2002.06715_, 2020.

"""

import os
import time

from absl import app
from absl import flags
from absl import logging

from edward2.experimental import sngp
import tensorflow as tf
import uncertainty_baselines as ub
import utils  # local file import
import uncertainty_metrics as um

flags.DEFINE_integer('per_core_batch_size', 128, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_float('base_learning_rate', 0.1,
                   'Base learning rate when train batch size is 256.')
flags.DEFINE_float('l2', 1e-4, 'L2 coefficient.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.mark_flag_as_required('data_dir')
flags.DEFINE_string('output_dir', '/tmp/imagenet',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('train_epochs', 90, 'Number of training epochs.')
flags.DEFINE_integer('corruptions_interval', 90,
                     'Number of epochs between evaluating on the corrupted '
                     'test data. Use -1 to never evaluate.')
flags.DEFINE_integer(
    'checkpoint_interval', -1,
    'Number of epochs between saving checkpoints. Use -1 to '
    'only save the last checkpoints.')
flags.DEFINE_string('alexnet_errors_path', None,
                    'Path to AlexNet corruption errors file.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# Spectral normalization flags.
flags.DEFINE_bool('use_spec_norm', True,
                  'Whether to apply spectral normalization.')
flags.DEFINE_bool('input_spec_norm', False,
                  'Whether to apply spectral normalization to the input layer')
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
    'gp_scale', 1.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_hidden_dim', 1024,
    'The hidden dimension of the GP layer, which corresponds to the number of '
    'random features used for the approximation.')
flags.DEFINE_bool(
    'gp_input_normalization', False,
    'Whether to normalize the input for GP layer using LayerNorm. This is '
    'similar to applying automatic relevance determination (ARD) in the '
    'classic GP literature.')
flags.DEFINE_float('gp_cov_ridge_penalty', 1e-3,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', 0.999,
    'The discount factor to compute the moving average of precision matrix.')
flags.DEFINE_float(
    'gp_mean_field_factor', 1e-6,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')
flags.DEFINE_bool(
    'gp_output_imagenet_initializer', True,
    'Whether to initialize GP output layer using Gaussian with small '
    'standard deviation (sd=0.01).')

# Rank1 BNN flags.
flags.DEFINE_integer('kl_annealing_epochs', 90,
                     'Number of epochs over which to anneal the KL term to 1.')
flags.DEFINE_string('alpha_initializer', 'trainable_normal',
                    'Initializer name for the alpha parameters.')
flags.DEFINE_string('gamma_initializer', 'trainable_normal',
                    'Initializer name for the gamma parameters.')
flags.DEFINE_string('alpha_regularizer', 'normal_kl_divergence',
                    'Regularizer name for the alpha parameters.')
flags.DEFINE_string('gamma_regularizer', 'normal_kl_divergence',
                    'Regularizer name for the gamma parameters.')
flags.DEFINE_boolean('use_additive_perturbation', False,
                     'Use additive perturbations instead of multiplicative.')
flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights. See [3] for detail')
flags.DEFINE_float('fast_weight_lr_multiplier', 0.25,
                   'fast weights lr multiplier.')
flags.DEFINE_bool('use_ensemble_bn', False, 'Whether to use ensemble bn.')
flags.DEFINE_float('dropout_rate', 1e-3,
                   'Dropout rate. Only used if alpha/gamma initializers are, '
                   'e.g., trainable normal with a fixed stddev.')
flags.DEFINE_float('prior_stddev', 0.05,
                   'Prior stddev. Sort of like a prior on dropout rate, where '
                   'it encourages defaulting/shrinking to this value.')
flags.DEFINE_integer('num_eval_samples', 1,
                     'Number of model predictions to sample per example at '
                     'eval time.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
# TODO(jereliu): Support use_bfloat16=True which currently raises error with
# spectral normalization.
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 32, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

flags.DEFINE_string('initializer', 'he_normal', '...')

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

  imagenet_train = utils.ImageNetInput(
      is_training=True,
      data_dir=FLAGS.data_dir,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  imagenet_eval = utils.ImageNetInput(
      is_training=False,
      data_dir=FLAGS.data_dir,
      batch_size=per_core_batch_size,
      use_bfloat16=FLAGS.use_bfloat16)
  test_datasets = {
      'clean':
          strategy.experimental_distribute_datasets_from_function(
              imagenet_eval.input_fn)
  }
  if FLAGS.corruptions_interval > 0:
    corruption_types, max_intensity = utils.load_corrupted_test_info()
    for name in corruption_types:
      for intensity in range(1, max_intensity + 1):
        dataset_name = '{0}_{1}'.format(name, intensity)
        corrupt_input_fn = utils.corrupt_test_input_fn(
            batch_size=per_core_batch_size,
            corruption_name=name,
            corruption_intensity=intensity,
            use_bfloat16=FLAGS.use_bfloat16)
        test_datasets[dataset_name] = (
            strategy.experimental_distribute_datasets_from_function(
                corrupt_input_fn))

  train_dataset = strategy.experimental_distribute_datasets_from_function(
      imagenet_train.input_fn)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  print('> alpha_regularizer = {}'.format(FLAGS.alpha_regularizer))
  print('> gamma_regularizer = {}'.format(FLAGS.gamma_regularizer))
  with strategy.scope():
    logging.info('Building Keras ResNet-50 model')
    model = ub.models.resnet50_sngp_rank1(
        input_shape=(224, 224, 3),
        batch_size=batch_size,
        num_classes=NUM_CLASSES,
        alpha_initializer=FLAGS.alpha_initializer,
        gamma_initializer=FLAGS.gamma_initializer,
        alpha_regularizer=FLAGS.alpha_regularizer,
        gamma_regularizer=FLAGS.gamma_regularizer,
        use_additive_perturbation=FLAGS.use_additive_perturbation,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        dropout_rate=FLAGS.dropout_rate,
        prior_stddev=FLAGS.prior_stddev,
        use_ensemble_bn=FLAGS.use_ensemble_bn,
        use_gp_layer=FLAGS.use_gp_layer,
        gp_hidden_dim=FLAGS.gp_hidden_dim,
        gp_scale=FLAGS.gp_scale,
        gp_bias=FLAGS.gp_bias,
        gp_input_normalization=FLAGS.gp_input_normalization,
        gp_cov_discount_factor=FLAGS.gp_cov_discount_factor,
        gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty,
        gp_output_imagenet_initializer=FLAGS.gp_output_imagenet_initializer,
        use_spec_norm=FLAGS.use_spec_norm,
        spec_norm_iteration=FLAGS.spec_norm_iteration,
        spec_norm_bound=FLAGS.spec_norm_bound,
        input_spec_norm=FLAGS.input_spec_norm,
        initializer=FLAGS.initializer)
    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())
    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate * batch_size / 256
    learning_rate = utils.LearningRateSchedule(steps_per_epoch,
                                               base_lr,
                                               FLAGS.train_epochs,
                                               _LR_SCHEDULE)
    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                        momentum=0.9,
                                        nesterov=True)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/kl': tf.keras.metrics.Mean(),
        'train/kl_scale': tf.keras.metrics.Mean(),
        'train/elbo': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/negative_log_likelihood': tf.keras.metrics.Mean(),
        'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'test/kl': tf.keras.metrics.Mean(),
        'test/elbo': tf.keras.metrics.Mean(),
        'test/ece': um.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        'test/stddev': tf.keras.metrics.Mean(),
        'test/member_accuracy_mean': (
            tf.keras.metrics.SparseCategoricalAccuracy()),
        'test/member_ece_mean': um.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins)

    }

    test_diversity = {
        'test/disagreement': tf.keras.metrics.Mean(),
        'test/average_kl': tf.keras.metrics.Mean(),
        'test/cosine_similarity': tf.keras.metrics.Mean(),
    }
    train_diversity = {
        'train/disagreement': tf.keras.metrics.Mean(),
        'train/average_kl': tf.keras.metrics.Mean(),
        'train/cosine_similarity': tf.keras.metrics.Mean(),
    }

    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, max_intensity + 1):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics['test/nll_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/kl_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/elbo_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/accuracy_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/ece_{}'.format(dataset_name)] = (
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))
          corrupt_metrics['test/stddev_{}'.format(dataset_name)] = (
              tf.keras.metrics.Mean())
          corrupt_metrics['test/member_acc_mean_{}'.format(dataset_name)] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics['test/member_ece_mean_{}'.format(dataset_name)] = (
              um.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

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

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  # TODO(ghassen): move this to rank1_bnn_utils.
  def compute_l2_loss(model):
    filtered_variables = []
    for var in model.trainable_variables:
      # Apply l2 on the BN parameters and bias terms. This
      # excludes only fast weight approximate posterior/prior parameters,
      # but pay caution to their naming scheme.
      if ('kernel' in var.name or
          'batch_norm' in var.name or
          'bias' in var.name):
        filtered_variables.append(tf.reshape(var, (-1,)))
    l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
        tf.concat(filtered_variables, axis=0))
    return l2_loss

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      labels = tf.tile(labels, [FLAGS.ensemble_size])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)

        if isinstance(logits, tuple):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        probs = tf.nn.softmax(logits)
        if FLAGS.ensemble_size > 1:
          per_probs = tf.reshape(
              probs, tf.concat([[FLAGS.ensemble_size, -1], probs.shape[1:]], 0))
          diversity_results = um.average_pairwise_diversity(
              per_probs, FLAGS.ensemble_size)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the weights. This excludes BN parameters and biases, but
          # pay caution to their naming scheme.
          if 'kernel' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = compute_l2_loss(model)
        print('> model.losses = {}'.format(model.losses))
        kl = sum(model.losses) / APPROX_IMAGENET_TRAIN_IMAGES
        kl_scale = tf.cast(optimizer.iterations + 1, kl.dtype)
        kl_scale /= steps_per_epoch * FLAGS.kl_annealing_epochs
        kl_scale = tf.minimum(1., kl_scale)
        kl_loss = kl_scale * kl

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss + kl_loss
        scaled_loss = loss / strategy.num_replicas_in_sync
        elbo = -(negative_log_likelihood + l2_loss + kl)

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

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].update_state(labels, probs)
      metrics['train/loss'].update_state(loss)
      metrics['train/kl'].update_state(kl)
      metrics['train/kl_scale'].update_state(kl_scale)
      metrics['train/elbo'].update_state(elbo)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)
      for k, v in diversity_results.items():
        train_diversity['train/' + k].update_state(v)

    strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_name):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images, labels = inputs
      images = tf.tile(images,
                       [FLAGS.ensemble_size * FLAGS.num_eval_samples, 1, 1, 1])
      logits = model(images, training=False)

      if isinstance(logits, tuple):
        # If model returns a tuple of (logits, covmat), extract both
        logits, covmat = logits
      else:
        covmat = tf.eye(FLAGS.per_core_batch_size)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)

      # Mean-field correction along the batch-dim axis.
      logits = sngp.mean_field_logits(
          logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
      stddev = tf.sqrt(tf.linalg.diag_part(covmat))
      print('OG Logits shape = {}'.format(logits.shape))
      print('OG covmat shape = {}'.format(covmat.shape))
      print('OG stddev shape = {}'.format(stddev.shape))

      # Reshaping.
      per_logits_tensor = tf.reshape(
          logits,
          [FLAGS.num_eval_samples, FLAGS.ensemble_size, -1, NUM_CLASSES])
      per_stddev_tensor = tf.reshape(
          stddev,
          tf.concat([[FLAGS.num_eval_samples, FLAGS.ensemble_size, -1],
                     stddev.shape[1:]], 0))
      per_probs_tensor = tf.nn.softmax(per_logits_tensor)
      print(' > per_logits_tensor.shape = {}'.format(per_logits_tensor.shape))
      print(' > per_stddev_tensor.shape = {}'.format(per_stddev_tensor.shape))
      print(' > per_probs_tensor.shape = {}'.format(per_probs_tensor.shape))

      # Averaging over the eval_samples dimension.
      ensemble_logits_tensor = tf.math.reduce_mean(per_logits_tensor, axis=0)
      ensemble_probs_tensor = tf.math.reduce_mean(per_probs_tensor, axis=0)
      ensemble_stddev_tensor = tf.math.reduce_mean(per_stddev_tensor, axis=0)
      print(' > ensemble_logits_tensor.shape = {}'.format(
          ensemble_logits_tensor.shape))
      print(' > ensemble_probs_tensor.shape = {}'.format(
          ensemble_probs_tensor.shape))
      print(' > ensemble_stddevs_tensor.shape = {}'.format(
          ensemble_stddev_tensor.shape))

      # Averaging across ensemble members.
      logits = tf.math.reduce_mean(per_logits_tensor, axis=[0, 1])
      probs = tf.math.reduce_mean(per_probs_tensor, axis=[0, 1])
      stddevs = tf.math.reduce_mean(per_probs_tensor, axis=[0, 1])
      print(' > logits.shape = {}'.format(logits.shape))
      print(' > probs.shape = {}'.format(probs.shape))
      print(' > stddevs.shape = {}'.format(stddevs.shape))

      if dataset_name == 'clean':
        # Compute diversity based on probs.
        diversity_results = um.average_pairwise_diversity(
            ensemble_probs_tensor, FLAGS.ensemble_size)
        for k, v in diversity_results.items():
          test_diversity['test/' + k].update_state(v)

        for i in range(FLAGS.ensemble_size):
          member_probs = ensemble_probs_tensor[i]
          print('> individual member probs shape = {}'.format(
              member_probs.shape))
          member_loss = tf.keras.losses.sparse_categorical_crossentropy(
              labels, member_probs)
          metrics['test/nll_member_{}'.format(i)].update_state(member_loss)
          metrics['test/accuracy_member_{}'.format(i)].update_state(
              labels, member_probs)
          metrics['test/member_accuracy_mean'].update_state(
              labels, member_probs)
          metrics['test/member_ece_mean'].update_state(labels, member_probs)

      labels_broadcasted = tf.broadcast_to(
          labels,
          [FLAGS.num_eval_samples, FLAGS.ensemble_size, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0, 1]) +
          tf.math.log(float(FLAGS.num_eval_samples * FLAGS.ensemble_size)))

      l2_loss = compute_l2_loss(model)
      kl = sum(model.losses) / IMAGENET_VALIDATION_IMAGES
      elbo = -(negative_log_likelihood + l2_loss + kl)

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].update_state(labels, probs)
        metrics['test/kl'].update_state(kl)
        metrics['test/elbo'].update_state(elbo)
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
    with summary_writer.as_default():
      for name, result in total_results.items():
        tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = checkpoint.save(os.path.join(
          FLAGS.output_dir, 'checkpoint'))
      logging.info('Saved checkpoint to %s', checkpoint_name)

  # TODO(jereliu): Convert to use SavedModel after fixing the graph-mode
  # execution bug in SpectralNormalizationConv2D which blocks the model.save()
  # functionality.
  final_checkpoint_name = checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)

if __name__ == '__main__':
  app.run(main)
