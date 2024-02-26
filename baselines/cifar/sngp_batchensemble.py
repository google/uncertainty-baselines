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

"""SNGP+BatchEnsemble Wide ResNet 28-10 on CIFAR-10 and CIFAR-100.

Spectral-normalized neural GP (SNGP) [1] is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

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
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import ood_utils  # local file import from baselines.cifar
import utils  # local file import from baselines.cifar
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_integer(
    'total_batch_size',
    256,
    'The total train (and test) batch size, split across all devices.')

# BatchEnsemble flags.
flags.DEFINE_integer('ensemble_size', 4, 'Size of ensemble.')
flags.DEFINE_float('random_sign_init', -0.5,
                   'Use random sign init for fast weights. See [3] for detail')
flags.DEFINE_float('fast_weight_lr_multiplier', 1.0,
                   'fast weights lr multiplier.')

# Spectral normalization flags.
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
flags.DEFINE_float('gp_cov_ridge_penalty', 1e-3,
                   'Ridge penalty parameter for GP posterior covariance.')
flags.DEFINE_float(
    'gp_cov_discount_factor', 0.999,
    'The discount factor to compute the moving average of precision matrix.')
flags.DEFINE_float(
    'gp_mean_field_factor', 0.001,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')

# OOD flags.
flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', False,
                  'Whether to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
                  'list of OOD datasets to evaluate on.')
flags.DEFINE_integer(
    'ood_interval', -1, 'Number of epochs between evaluating on OOD metrics.'
    ' Use -1 to never evaluate.')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
                  'Wheter to use DempsterShafer Uncertainty score.')

# Redefining default values
flags.FLAGS.set_default('lr_decay_epochs', ['80', '160', '180'])
flags.FLAGS.set_default('l2', 3e-4)
flags.FLAGS.set_default('train_epochs', 250)
FLAGS = flags.FLAGS


def mean_field_logits(logits, covmat, mean_field_factor=1.):
  """Adjust the predictive logits so its softmax approximates posterior mean."""
  # TODO(jereliu): Maybe move to ed2 library or ed2.experimental.sngp.
  logits_scale = tf.sqrt(1. + tf.linalg.diag_part(covmat) * mean_field_factor)
  if mean_field_factor > 0:
    logits = logits / tf.expand_dims(logits_scale, axis=-1)

  return logits


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  ds_info = tfds.builder(FLAGS.dataset).info
  batch_size = FLAGS.total_batch_size // FLAGS.ensemble_size
  # Train_proportion is a float so need to convert steps_per_epoch to int.
  steps_per_epoch = int((ds_info.splits['train'].num_examples *
                         FLAGS.train_proportion) // batch_size)
  steps_per_eval = ds_info.splits['test'].num_examples // batch_size
  num_classes = ds_info.features['label'].num_classes

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

  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TRAIN,
      validation_percent=1. - FLAGS.train_proportion)
  train_dataset = train_builder.load(batch_size=batch_size)
  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:
    validation_builder = ub.datasets.get(
        FLAGS.dataset,
        data_dir=data_dir,
        download_data=FLAGS.download_data,
        split=tfds.Split.VALIDATION,
        validation_percent=1. - FLAGS.train_proportion,
        drop_remainder=FLAGS.drop_remainder_for_eval)
    validation_dataset = validation_builder.load(batch_size=batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // batch_size
  clean_test_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=tfds.Split.TEST,
      drop_remainder=FLAGS.drop_remainder_for_eval)
  clean_test_dataset = clean_test_builder.load(batch_size=batch_size)
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }
  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_builder.num_examples // batch_size
  num_classes = 100 if FLAGS.dataset == 'cifar100' else 10

  if FLAGS.eval_on_ood:
    ood_dataset_names = FLAGS.ood_dataset
    ood_ds, steps_per_ood = ood_utils.load_ood_datasets(
        ood_dataset_names,
        clean_test_builder,
        1 - FLAGS.train_proportion,
        batch_size,
        drop_remainder=FLAGS.drop_remainder_for_eval)
    ood_datasets = {
        name: strategy.experimental_distribute_dataset(ds)
        for name, ds in ood_ds.items()
    }

  if FLAGS.corruptions_interval > 0:
    extra_kwargs = {}
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
            split=tfds.Split.TEST,
            drop_remainder=FLAGS.drop_remainder_for_eval**extra_kwargs).load(
                batch_size=batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras model')
    model = ub.models.wide_resnet_sngp_be(
        input_shape=(32, 32, 3),
        batch_size=batch_size,
        depth=28,
        width_multiplier=10,
        num_classes=num_classes,
        ensemble_size=FLAGS.ensemble_size,
        random_sign_init=FLAGS.random_sign_init,
        l2=FLAGS.l2,
        use_gp_layer=FLAGS.use_gp_layer,
        gp_input_dim=FLAGS.gp_input_dim,
        gp_hidden_dim=FLAGS.gp_hidden_dim,
        gp_scale=FLAGS.gp_scale,
        gp_bias=FLAGS.gp_bias,
        gp_input_normalization=FLAGS.gp_input_normalization,
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
                                        momentum=1.0 - FLAGS.one_minus_momentum,
                                        nesterov=True)
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
        'test/stddev': tf.keras.metrics.Mean(),
    }
    eval_dataset_splits = ['test']
    if validation_dataset:
      metrics.update({
          'validation/negative_log_likelihood': tf.keras.metrics.Mean(),
          'validation/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
          'validation/ece': rm.metrics.ExpectedCalibrationError(
              num_bins=FLAGS.num_bins),
          'validation/stddev': tf.keras.metrics.Mean(),
      })
      eval_dataset_splits += ['validation']
    if FLAGS.eval_on_ood:
      ood_metrics = ood_utils.create_ood_metrics(
          ood_dataset_names, tpr_list=FLAGS.ood_tpr_threshold)
      metrics.update(ood_metrics)
    for i in range(FLAGS.ensemble_size):
      for dataset_split in eval_dataset_splits:
        metrics[f'{dataset_split}/nll_member_{i}'] = tf.keras.metrics.Mean()
        metrics[f'{dataset_split}/accuracy_member_{i}'] = (
            tf.keras.metrics.SparseCategoricalAccuracy())
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
    if FLAGS.saved_model_dir:
      logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
      latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
    if FLAGS.eval_only:
      initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval

  @tf.function
  def train_step(iterator):
    """Training StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']
      images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
      labels = tf.tile(labels, [FLAGS.ensemble_size])

      with tf.GradientTape() as tape:
        logits = model(images, training=True)
        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(labels,
                                                            logits,
                                                            from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weight approximate
          # posterior/prior parameters. This is excludes BN and slow weights,
          # but pay caution to the naming scheme.
          if ('batch_norm' not in var.name and 'kernel' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      probs = tf.nn.softmax(logits)
      metrics['train/ece'].add_batch(probs, label=labels)
      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, logits)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""
    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      logits_list = []
      stddev_list = []

      for i in range(FLAGS.ensemble_size):
        logits = model(images, training=False)
        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract both
          logits, covmat = logits
          logits = mean_field_logits(
              logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
        else:
          covmat = tf.eye(logits.shape[0])

        stddev = tf.sqrt(tf.linalg.diag_part(covmat))

        stddev_list.append(stddev)
        logits_list.append(logits)

        member_probs = tf.nn.softmax(logits)
        member_loss = tf.keras.losses.sparse_categorical_crossentropy(
            labels, member_probs)
        metrics[f'{dataset_split}/nll_member_{i}'].update_state(member_loss)
        metrics[f'{dataset_split}/accuracy_member_{i}'].update_state(
            labels, member_probs)
      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      stddev_list = tf.stack(stddev_list, axis=0)

      stddev = tf.reduce_mean(stddev_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.ensemble_size, tf.shape(labels)[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.ensemble_size)))

      if dataset_name == 'clean':
        metrics[f'{dataset_split}/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics[f'{dataset_split}/accuracy'].update_state(labels, probs)
        metrics[f'{dataset_split}/ece'].add_batch(probs, label=labels)
        metrics[f'{dataset_split}/stddev'].update_state(stddev)
      elif dataset_name.startswith('ood'):
        ood_labels = 1 - inputs['is_in_distribution']
        if FLAGS.dempster_shafer_ood:
          ood_scores = ood_utils.DempsterShaferUncertainty(logits)
        else:
          ood_scores = 1 - tf.reduce_max(probs, axis=-1)

        # Edgecase for if dataset_name contains underscores
        ood_dataset_name = '_'.join(dataset_name.split('_')[1:])
        for name, metric in metrics.items():
          if ood_dataset_name in name:
            metric.update_state(ood_labels, ood_scores)
      elif FLAGS.corruptions_interval > 0:
        corrupt_metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        corrupt_metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        corrupt_metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)
        corrupt_metrics['test/stddev_{}'.format(dataset_name)].update_state(
            stddev)

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
                   current_step / max_steps,
                   epoch + 1,
                   FLAGS.train_epochs,
                   steps_per_sec,
                   eta_seconds / 60,
                   time_elapsed / 60))
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

    if FLAGS.eval_on_ood:
      for dataset_name in ood_dataset_names:
        ood_iterator = iter(ood_datasets['ood_{}'.format(dataset_name)])
        logging.info('Calculating OOD on dataset %s', dataset_name)
        logging.info('Running OOD eval at epoch: %s', epoch)
        test_step(ood_iterator, 'test', 'ood_{}'.format(dataset_name),
                  steps_per_ood[dataset_name])

        logging.info('Done with OOD eval on %s', dataset_name)

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
    for i in range(FLAGS.ensemble_size):
      logging.info('Member %d Test Loss: %.4f, Accuracy: %.2f%%',
                   i, metrics['test/nll_member_{}'.format(i)].result(),
                   metrics['test/accuracy_member_{}'.format(i)].result() * 100)
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

    if FLAGS.corruptions_interval > 0:
      for metric in corrupt_metrics.values():
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
        'gp_mean_field_factor': FLAGS.gp_mean_field_factor,
        'gp_input_dim': FLAGS.gp_input_dim,
        'gp_scale': FLAGS.gp_scale,
        'gp_hidden_dim': FLAGS.gp_hidden_dim,
        'fast_weight_lr_multiplier': FLAGS.fast_weight_lr_multiplier,
        'random_sign_init': FLAGS.random_sign_init,
    })


if __name__ == '__main__':
  app.run(main)
