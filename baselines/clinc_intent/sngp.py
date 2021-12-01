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

"""Bidirectional encoder representations from transformers (BERT) with SNGP.

Spectral-normalized neural Gaussian process (SNGP) [1] is a simple method to
improve a deterministic neural network's uncertainty. It simply applies spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

## Note:

Different from the paper, this implementation computes the posterior using the
Laplace approximation based on the Gaussian likelihood (i.e., squared loss)
rather than that based on cross-entropy loss. As a result, the logits for all
classes share the same covariance. In the experiments, this approach is shown to
perform better and computationally more scalable when the number of output
classes are large.

## References:

[1]: Jeremiah Liu et al. Simple and Principled Uncertainty Estimation with
     Deterministic Deep Learning via Distance Awareness.
     _arXiv preprint arXiv:2006.10108_, 2020.
     https://arxiv.org/abs/2006.10108
[2]: Zhiyun Lu, Eugene Ie, Fei Sha. Uncertainty Estimation with Infinitesimal
     Jackknife.  _arXiv preprint arXiv:2006.07584_, 2020.
     https://arxiv.org/abs/2006.07584
"""

import os
import time
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import robustness_metrics as rm
import tensorflow as tf

import uncertainty_baselines as ub
import bert_utils  # local file import from baselines.clinc_intent
from tensorboard.plugins.hparams import api as hp

# Data flags
flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the TFRecord datasets and the tokenizer for Clinc '
    'Intent Detection Data.')

# Checkpoint flags
flags.DEFINE_string(
    'bert_dir', None,
    'Directory to BERT pre-trained checkpoints and config files.')
flags.DEFINE_string(
    'bert_ckpt_dir', None, 'Directory to BERT pre-trained checkpoints. '
    'If None then then default to {bert_dir}/bert_model.ckpt.')
flags.DEFINE_string(
    'bert_config_dir', None, 'Directory to BERT config files. '
    'If None then then default to {bert_dir}/bert_config.json.')

# Normalization flags.
flags.DEFINE_bool(
    'use_layer_norm_att', True,
    'Whether to apply layer normalization to the self-attention layers.')
flags.DEFINE_bool(
    'use_layer_norm_ffn', True,
    'Whether to apply layer normalization to the feedforward layers.')
flags.DEFINE_bool(
    'use_spec_norm_att', False,
    'Whether to apply spectral normalization to the self-attention layers.')
flags.DEFINE_bool(
    'use_spec_norm_ffn', False,
    'Whether to apply spectral normalization to the feedforward layers.')
flags.DEFINE_bool(
    'use_spec_norm_plr', True,
    'Whether to apply spectral normalization to the final CLS pooler layer.')
flags.DEFINE_integer(
    'spec_norm_iteration', 1,
    'Number of power iterations to perform for estimating '
    'the spectral norm of weight matrices.')
flags.DEFINE_float('spec_norm_bound', .95,
                   'Upper bound to spectral norm of weight matrices.')

# Gaussian process flags.
flags.DEFINE_bool('use_gp_layer', True,
                  'Whether to use Gaussian process as the output layer.')
flags.DEFINE_float('gp_bias', 0., 'The bias term for GP layer.')
flags.DEFINE_float(
    'gp_scale', 2.,
    'The length-scale parameter for the RBF kernel of the GP layer.')
flags.DEFINE_integer(
    'gp_hidden_dim', 2048,
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
    'gp_mean_field_factor', 1e-1,
    'The tunable multiplicative factor used in the mean-field approximation '
    'for the posterior mean of softmax Gaussian process. If -1 then use '
    'posterior mode instead of posterior mean. See [2] for detail.')


# Optimization and evaluation flags
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('eval_batch_size', 512, 'Batch size for CPU evaluation.')
flags.DEFINE_float(
    'base_learning_rate', 5e-5,
    'Base learning rate when total batch size is 128. It is '
    'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'checkpoint_interval', 40,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 2,
                     'Number of epochs between evaluation.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/clinc_intent', 'Output directory.')
flags.DEFINE_integer('train_epochs', 40, 'Number of training epochs.')
flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')
flags.DEFINE_integer(
    'num_mc_samples', 1,
    'Number of Monte Carlo forward passes to collect for ensemble prediction.'
    'Currently can only be 1 since the model is deterministic.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS

# TODO(jereliu): Add support for Monte Carlo Dropout.


def resolve_bert_ckpt_and_config_dir(bert_dir, bert_config_dir, bert_ckpt_dir):
  """Resolves BERT checkpoint and config file directories."""

  missing_ckpt_or_config_dir = not (bert_ckpt_dir and bert_config_dir)
  if missing_ckpt_or_config_dir:
    if not bert_dir:
      raise ValueError('bert_dir cannot be empty.')

    if not bert_config_dir:
      bert_config_dir = os.path.join(bert_dir, 'bert_config.json')

    if not bert_ckpt_dir:
      bert_ckpt_dir = os.path.join(bert_dir, 'bert_model.ckpt')
  return bert_config_dir, bert_ckpt_dir


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

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores
  train_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='train',
      data_dir=FLAGS.data_dir,
      data_mode='ind')
  ind_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='ind')
  ood_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='ood')
  all_dataset_builder = ub.datasets.ClincIntentDetectionDataset(
      split='test',
      data_dir=FLAGS.data_dir,
      data_mode='all')

  dataset_builders = {
      'clean': ind_dataset_builder,
      'ood': ood_dataset_builder,
      'all': all_dataset_builder
  }

  train_dataset = train_dataset_builder.load(
      batch_size=FLAGS.per_core_batch_size)

  ds_info = train_dataset_builder.tfds_info
  feature_size = ds_info.metadata['feature_size']
  # num_classes is number of valid intents plus out-of-scope intent
  num_classes = ds_info.features['intent_label'].num_classes + 1

  steps_per_epoch = train_dataset_builder.num_examples // batch_size

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=FLAGS.eval_batch_size)
    steps_per_eval[dataset_name] = (
        dataset_builder.num_examples // FLAGS.eval_batch_size)

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building BERT model')
    logging.info('use_gp_layer=%s', FLAGS.use_gp_layer)
    logging.info('use_spec_norm_att=%s', FLAGS.use_spec_norm_att)
    logging.info('use_spec_norm_ffn=%s', FLAGS.use_spec_norm_ffn)
    logging.info('use_layer_norm_att=%s', FLAGS.use_layer_norm_att)
    logging.info('use_layer_norm_ffn=%s', FLAGS.use_layer_norm_ffn)

    bert_config_dir, bert_ckpt_dir = resolve_bert_ckpt_and_config_dir(
        FLAGS.bert_dir, FLAGS.bert_config_dir, FLAGS.bert_ckpt_dir)
    bert_config = bert_utils.create_config(bert_config_dir)

    gp_layer_kwargs = dict(
        num_inducing=FLAGS.gp_hidden_dim,
        gp_kernel_scale=FLAGS.gp_scale,
        gp_output_bias=FLAGS.gp_bias,
        normalize_input=FLAGS.gp_input_normalization,
        gp_cov_momentum=FLAGS.gp_cov_discount_factor,
        gp_cov_ridge_penalty=FLAGS.gp_cov_ridge_penalty)
    spec_norm_kwargs = dict(
        iteration=FLAGS.spec_norm_iteration,
        norm_multiplier=FLAGS.spec_norm_bound)

    model, bert_encoder = ub.models.bert_sngp_model(
        num_classes=num_classes,
        bert_config=bert_config,
        gp_layer_kwargs=gp_layer_kwargs,
        spec_norm_kwargs=spec_norm_kwargs,
        use_gp_layer=FLAGS.use_gp_layer,
        use_spec_norm_att=FLAGS.use_spec_norm_att,
        use_spec_norm_ffn=FLAGS.use_spec_norm_ffn,
        use_layer_norm_att=FLAGS.use_layer_norm_att,
        use_layer_norm_ffn=FLAGS.use_layer_norm_ffn,
        use_spec_norm_plr=FLAGS.use_spec_norm_plr)
    # Create an AdamW optimizer with beta_2=0.999, epsilon=1e-6.
    optimizer = bert_utils.create_optimizer(
        FLAGS.base_learning_rate,
        steps_per_epoch=steps_per_epoch,
        epochs=FLAGS.train_epochs,
        warmup_proportion=FLAGS.warmup_proportion,
        beta_1=1.0 - FLAGS.one_minus_momentum)

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/ece': rm.metrics.ExpectedCalibrationError(
            num_bins=FLAGS.num_bins),
    }

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
    else:
      # load BERT from initial checkpoint
      bert_encoder, _, _ = bert_utils.load_bert_weight_from_ckpt(
          bert_model=bert_encoder,
          bert_ckpt_dir=bert_ckpt_dir,
          repl_patterns=ub.models.bert_sngp.CHECKPOINT_REPL_PATTERNS)
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

  # Finally, define test metrics outside the accelerator scope for CPU eval.
  metrics.update({
      'test/negative_log_likelihood': tf.keras.metrics.Mean(),
      'test/accuracy': tf.keras.metrics.SparseCategoricalAccuracy(),
      'test/ece': rm.metrics.ExpectedCalibrationError(
          num_bins=FLAGS.num_bins),
      'test/stddev': tf.keras.metrics.Mean(),
  })
  for dataset_name, test_dataset in test_datasets.items():
    if dataset_name != 'clean':
      metrics.update({
          'test/nll_{}'.format(dataset_name):
              tf.keras.metrics.Mean(),
          'test/accuracy_{}'.format(dataset_name):
              tf.keras.metrics.SparseCategoricalAccuracy(),
          'test/ece_{}'.format(dataset_name):
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
          'test/stddev_{}'.format(dataset_name):
              tf.keras.metrics.Mean(),
      })
  metrics.update({
      'test/auroc_all': tf.keras.metrics.AUC(curve='ROC'),
      'test/auprc_all': tf.keras.metrics.AUC(curve='PR')
  })

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels = bert_utils.create_feature_and_label(
          inputs, feature_size)

      with tf.GradientTape() as tape:
        # Set learning phase to enable dropout etc during training.
        logits = model(features, training=True)

        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract logits
          logits, _ = logits
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits, from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
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
  def test_step(iterator, dataset_name, num_steps):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels = bert_utils.create_feature_and_label(
          inputs, feature_size)

      # Compute ensemble prediction over Monte Carlo forward-pass samples.
      logits_list = []
      stddev_list = []
      for _ in range(FLAGS.num_mc_samples):
        logits = model(features, training=False)

        if isinstance(logits, (list, tuple)):
          # If model returns a tuple of (logits, covmat), extract both.
          logits, covmat = logits
        else:
          covmat = tf.eye(FLAGS.eval_batch_size)

        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)
          covmat = tf.cast(covmat, tf.float32)

        logits = ed.layers.utils.mean_field_logits(
            logits, covmat, mean_field_factor=FLAGS.gp_mean_field_factor)
        stddev = tf.sqrt(tf.linalg.diag_part(covmat))

        logits_list.append(logits)
        stddev_list.append(stddev)

      # Logits dimension is (num_samples, batch_size, num_classes).
      logits_list = tf.stack(logits_list, axis=0)
      stddev_list = tf.stack(stddev_list, axis=0)

      stddev = tf.reduce_mean(stddev_list, axis=0)
      probs_list = tf.nn.softmax(logits_list)
      probs = tf.reduce_mean(probs_list, axis=0)

      labels_broadcasted = tf.broadcast_to(
          labels, [FLAGS.num_mc_samples, labels.shape[0]])
      log_likelihoods = -tf.keras.losses.sparse_categorical_crossentropy(
          labels_broadcasted, logits_list, from_logits=True)
      negative_log_likelihood = tf.reduce_mean(
          -tf.reduce_logsumexp(log_likelihoods, axis=[0]) +
          tf.math.log(float(FLAGS.num_mc_samples)))

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)
        metrics['test/stddev'].update_state(stddev)
      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)
        metrics['test/stddev_{}'.format(dataset_name)].update_state(stddev)

      if dataset_name == 'all':
        ood_labels = tf.cast(labels == 150, labels.dtype)
        ood_probs = 1. - tf.reduce_max(probs, axis=-1)
        metrics['test/auroc_{}'.format(dataset_name)].update_state(
            ood_labels, ood_probs)
        metrics['test/auprc_{}'.format(dataset_name)].update_state(
            ood_labels, ood_probs)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      step_fn(next(iterator))

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

    if epoch % FLAGS.evaluation_interval == 0:
      for dataset_name, test_dataset in test_datasets.items():
        test_iterator = iter(test_dataset)
        logging.info('Testing on dataset %s', dataset_name)
        logging.info('Starting to run eval at epoch: %s', epoch)
        test_step(test_iterator, dataset_name, steps_per_eval[dataset_name])
        logging.info('Done with testing on %s', dataset_name)

      logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                   metrics['train/loss'].result(),
                   metrics['train/accuracy'].result() * 100)
      logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                   metrics['test/negative_log_likelihood'].result(),
                   metrics['test/accuracy'].result() * 100)
      total_results = {
          name: metric.result() for name, metric in metrics.items()
      }
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
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'gp_mean_field_factor': FLAGS.gp_mean_field_factor,
    })


if __name__ == '__main__':
  app.run(main)
