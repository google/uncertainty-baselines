# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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

"""Rank-1 BNN ResNet-50 on on Kaggle's Diabetic Retinopathy Detection.

A Rank-1 Bayesian neural net (Rank-1 BNN) [1] is an efficient and scalable
approach to variational BNNs that posits prior distributions on rank-1 factors
of the weights and optimizes global mixture variational posterior distributions.
References:
  [1]: Michael W. Dusenberry*, Ghassen Jerfel*, Yeming Wen, Yian Ma, Jasper
       Snoek, Katherine Heller, Balaji Lakshminarayanan, Dustin Tran. Efficient
       and Scalable Bayesian Neural Nets with Rank-1 Factors. In Proc. of
       International Conference on Machine Learning (ICML) 2020.
       https://arxiv.org/abs/2005.07186

TODO(nband): update uncertainty estimation code for Rank 1
  Mixture of Gaussian ensembling, in addition to
  Deep Ensemble--style ensembling.
"""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import datetime
import os
import pathlib
import pprint
import time

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf
import uncertainty_baselines as ub
import utils  # local file import from baselines.diabetic_retinopathy_detection
import wandb
from tensorboard.plugins.hparams import api as hp

DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/rank1_bnn',
    'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')
flags.DEFINE_bool('use_test', False, 'Whether to use a test split.')
flags.DEFINE_string('preproc_builder_config', 'btgraham-300', (
    'Determines the preprocessing procedure for the images. Supported options: '
    '{btgraham-300, blur-3-btgraham-300, blur-5-btgraham-300, '
    'blur-10-btgraham-300, blur-20-btgraham-300}.'))
flags.DEFINE_string(
    'dr_decision_threshold', 'moderate',
    ('specifies where to binarize the labels {0, 1, 2, 3, 4} to create the '
     'binary classification task. Only affects the APTOS dataset partitioning. '
     "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
     "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))
flags.DEFINE_bool('load_from_checkpoint', False,
                  'Attempt to load from checkpoint')
flags.DEFINE_string('checkpoint_dir', None, 'Path to load Keras checkpoints.')
flags.DEFINE_bool('cache_eval_datasets', False, 'Caches eval datasets.')

# Logging and hyperparameter tuning.
flags.DEFINE_bool('use_wandb', False, 'Use wandb for logging.')
flags.DEFINE_string('wandb_dir', 'wandb', 'Directory where wandb logs go.')
flags.DEFINE_string('project', 'ub-debug', 'Wandb project name.')
flags.DEFINE_string('exp_name', None, 'Give experiment a name.')
flags.DEFINE_string('exp_group', None, 'Give experiment a group name.')

# OOD flags.
flags.DEFINE_string(
    'distribution_shift', None,
    ('Specifies distribution shift to use, if any.'
     'aptos: loads APTOS (India) OOD validation and test datasets. '
     '  Kaggle/EyePACS in-domain datasets are unchanged.'
     'severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision '
     '  of the Kaggle/EyePACS dataset to hold out clinical severity labels '
     '  as OOD.'))
flags.DEFINE_bool(
    'load_train_split', True,
    'Should always be enabled - required to load train split of the dataset.')

# Learning rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 0.032299, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.066501, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60'],
                  'Epochs to decay learning rate by.')
flags.DEFINE_float('fast_weight_lr_multiplier', 1.2390,
                   'fast weights lr multiplier.')

# Rank-1 BNN flags.
flags.DEFINE_integer('num_mc_samples_train', 1,
                     'Number of MC samples used during training.')
flags.DEFINE_integer('num_mc_samples_eval', 5,
                     'Number of MC samples to use for prediction.')
flags.DEFINE_integer('kl_annealing_epochs', 200,
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
flags.DEFINE_float(
    'dropout_rate', 1e-3,
    'Dropout rate. Only used if alpha/gamma initializers are, '
    'e.g., trainable normal with a fixed stddev.')
flags.DEFINE_float(
    'prior_stddev', 0.05,
    'Prior stddev. Sort of like a prior on dropout rate, where '
    'it encourages defaulting/shrinking to this value.')
flags.DEFINE_float('random_sign_init', 0.75,
                   'Use random sign init for fast weights.')
flags.DEFINE_bool('use_ensemble_bn', False, 'Whether to use ensemble bn.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,'
    'test respectively). `None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. `minibatch` will use the proportions of each minibatch to '
    'reweight the loss.')
flags.DEFINE_float('l2', 0.000022417, 'L2 coefficient.')
flags.DEFINE_integer('ensemble_size', 1, 'Size of ensemble.')
flags.DEFINE_integer('per_core_batch_size', 16, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer(
    'checkpoint_interval', 25,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE computation.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU')
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.'
    'Specify `read-from-file` to retrieve the name from tpu_name.txt.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.random.set_seed(FLAGS.seed)

  # Wandb Setup
  if FLAGS.use_wandb:
    pathlib.Path(FLAGS.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb_args = dict(
        project=FLAGS.project,
        entity='uncertainty-baselines',
        dir=FLAGS.wandb_dir,
        reinit=True,
        name=FLAGS.exp_name,
        group=FLAGS.exp_group)
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(FLAGS, allow_val_change=True)
    output_dir = str(
        os.path.join(FLAGS.output_dir,
                     datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
  else:
    wandb_run = None
    output_dir = FLAGS.output_dir

  tf.io.gfile.makedirs(output_dir)
  logging.info('Saving checkpoints at %s', output_dir)

  # Log Run Hypers
  hypers_dict = {
      'per_core_batch_size': FLAGS.per_core_batch_size,
      'base_learning_rate': FLAGS.base_learning_rate,
      'fast_weight_lr_multiplier': FLAGS.fast_weight_lr_multiplier,
      'one_minus_momentum': FLAGS.one_minus_momentum,
      'l2': FLAGS.l2
  }
  logging.info('Hypers:')
  logging.info(pprint.pformat(hypers_dict))

  # Initialize distribution strategy on flag-specified accelerator
  strategy = utils.init_distribution_strategy(FLAGS.force_use_cpu,
                                              FLAGS.use_gpu, FLAGS.tpu)
  use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  per_core_batch_size = FLAGS.per_core_batch_size // FLAGS.ensemble_size
  batch_size = per_core_batch_size * FLAGS.num_cores

  # Reweighting loss for class imbalance
  class_reweight_mode = FLAGS.class_reweight_mode
  if class_reweight_mode == 'constant':
    class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  else:
    class_weights = None

  # Load in datasets.
  datasets, steps = utils.load_dataset(
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      flags=FLAGS,
      strategy=strategy)
  available_splits = list(datasets.keys())
  test_splits = [split for split in available_splits if 'test' in split]
  eval_splits = [
      split for split in available_splits
      if 'validation' in split or 'test' in split
  ]

  # Iterate eval datasets
  eval_datasets = {split: iter(datasets[split]) for split in eval_splits}
  dataset_train = datasets['train']
  train_steps_per_epoch = steps['train']
  train_dataset_size = train_steps_per_epoch * batch_size

  if FLAGS.use_bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  summary_writer = tf.summary.create_file_writer(
      os.path.join(output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 Rank-1 BNN model.')
    model = None
    if FLAGS.load_from_checkpoint:
      initial_epoch, model = utils.load_keras_checkpoints(
          FLAGS.checkpoint_dir, load_ensemble=False, return_epoch=True)
    else:
      initial_epoch = 0
      model = ub.models.resnet50_rank1(
          input_shape=utils.load_input_shape(dataset_train),
          num_classes=1,  # binary classification task
          alpha_initializer=FLAGS.alpha_initializer,
          gamma_initializer=FLAGS.gamma_initializer,
          alpha_regularizer=FLAGS.alpha_regularizer,
          gamma_regularizer=FLAGS.gamma_regularizer,
          use_additive_perturbation=FLAGS.use_additive_perturbation,
          ensemble_size=FLAGS.ensemble_size,
          random_sign_init=FLAGS.random_sign_init,
          dropout_rate=FLAGS.dropout_rate,
          prior_stddev=FLAGS.prior_stddev,
          use_tpu=use_tpu,
          use_ensemble_bn=FLAGS.use_ensemble_bn)
      utils.log_model_init_info(model=model)

    # Scale learning rate and decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate
    lr_decay_epochs = [
        (int(start_epoch_str) * FLAGS.train_epochs) // DEFAULT_NUM_EPOCHS
        for start_epoch_str in FLAGS.lr_decay_epochs
    ]
    learning_rate = ub.schedules.WarmUpPiecewiseConstantSchedule(
        steps_per_epoch=train_steps_per_epoch,
        base_learning_rate=base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        learning_rate=learning_rate,
        momentum=1.0 - FLAGS.one_minus_momentum,
        nesterov=True)
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=use_tpu,
        num_bins=FLAGS.num_bins,
        use_validation=FLAGS.use_validation,
        available_splits=available_splits)

    # Rank-1 specific metrics
    metrics.update({
        'train/kl': tf.keras.metrics.Mean(),
        'train/kl_scale': tf.keras.metrics.Mean()
    })

    # TODO(nband): debug or remove
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # latest_checkpoint = tf.train.latest_checkpoint(output_dir)
    # if latest_checkpoint:
    #   # checkpoint.restore must be within a strategy.scope() so that optimizer
    #   # slot variables are mirrored.
    #   checkpoint.restore(latest_checkpoint)
    #   logging.info('Loaded checkpoint %s', latest_checkpoint)
    #   initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  # Define metrics outside the accelerator scope for CPU eval.
  # This will cause an error on TPU.
  if not use_tpu:
    metrics.update(
        utils.get_diabetic_retinopathy_cpu_metrics(
            available_splits=available_splits,
            use_validation=FLAGS.use_validation))

  for test_split in test_splits:
    metrics.update({f'{test_split}/ms_per_example': tf.keras.metrics.Mean()})

  # Initialize loss function based on class reweighting setting
  loss_fn = utils.get_diabetic_retinopathy_loss_fn(
      class_reweight_mode=class_reweight_mode, class_weights=class_weights)

  # * Prepare for Evaluation *

  # Get the wrapper function which will produce uncertainty estimates for
  # our choice of method and Y/N ensembling.
  uncertainty_estimator_fn = utils.get_uncertainty_estimator(
      'rank1', use_ensemble=False, use_tf=True)

  # Wrap our estimator to predict probabilities (apply sigmoid on logits)
  eval_estimator = utils.wrap_retinopathy_estimator(
      model, use_mixed_precision=FLAGS.use_bfloat16, numpy_outputs=False)

  estimator_args = {'num_samples': FLAGS.num_mc_samples_eval}

  def compute_l2_loss(model):
    filtered_variables = []
    for var in model.trainable_variables:
      # Apply l2 on the BN parameters and bias terms. This
      # excludes only fast weight approximate posterior/prior parameters,
      # but pay caution to their naming scheme.
      if ('kernel' in var.name or 'batch_norm' in var.name or
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
      images = inputs['features']
      labels = inputs['labels']
      if FLAGS.ensemble_size > 1:
        images = tf.tile(images, [FLAGS.ensemble_size, 1, 1, 1])
        labels = tf.tile(labels, [FLAGS.ensemble_size])

      # For minibatch class reweighting, initialize per-batch loss function
      if class_reweight_mode == 'minibatch':
        batch_loss_fn = utils.get_minibatch_reweighted_loss_fn(labels=labels)
      else:
        batch_loss_fn = loss_fn

      with tf.GradientTape() as tape:
        if FLAGS.num_mc_samples_train > 1:
          # Pythonic Implem
          logits_list = []
          for _ in range(FLAGS.num_mc_samples_train):
            logits = model(images, training=True)
            logits = tf.squeeze(logits, axis=-1)
            if FLAGS.use_bfloat16:
              logits = tf.cast(logits, tf.float32)

            logits_list.append(logits)

          # Logits dimension is (num_samples, batch_size).
          logits_list = tf.stack(logits_list, axis=0)

          probs_list = tf.nn.sigmoid(logits_list)
          probs = tf.reduce_mean(probs_list, axis=0)
          negative_log_likelihood = tf.reduce_mean(
              batch_loss_fn(
                  y_true=tf.expand_dims(labels, axis=-1),
                  y_pred=probs,
                  from_logits=False))
        else:
          # Single train step
          logits = model(images, training=True)
          if FLAGS.use_bfloat16:
            logits = tf.cast(logits, tf.float32)

          negative_log_likelihood = tf.reduce_mean(
              batch_loss_fn(
                  y_true=tf.expand_dims(labels, axis=-1),
                  y_pred=logits,
                  from_logits=True))
          probs = tf.squeeze(tf.nn.sigmoid(logits))

        l2_loss = compute_l2_loss(model)
        kl = sum(model.losses) / train_dataset_size
        kl_scale = tf.cast(optimizer.iterations + 1, kl.dtype)
        kl_scale /= train_steps_per_epoch * FLAGS.kl_annealing_epochs
        kl_scale = tf.minimum(1., kl_scale)
        kl_loss = kl_scale * kl

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        loss = negative_log_likelihood + l2_loss + kl_loss
        scaled_loss = loss / strategy.num_replicas_in_sync
        # elbo = -(negative_log_likelihood + l2_loss + kl)

      grads = tape.gradient(scaled_loss, model.trainable_variables)

      # Separate learning rate implementation.
      if FLAGS.fast_weight_lr_multiplier != 1.0:
        grads_and_vars = []
        for grad, var in zip(grads, model.trainable_variables):
          # Apply different learning rate on the fast weights. This excludes BN
          # and slow weights, but pay caution to the naming scheme.
          if ('batch_norm' not in var.name and 'kernel' not in var.name):
            grads_and_vars.append((grad * FLAGS.fast_weight_lr_multiplier, var))
          else:
            grads_and_vars.append((grad, var))
        optimizer.apply_gradients(grads_and_vars)
      else:
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/kl'].update_state(kl)
      metrics['train/kl_scale'].update_state(kl_scale)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/auprc'].update_state(labels, probs)
      metrics['train/auroc'].update_state(labels, probs)

      if not use_tpu:
        metrics['train/ece'].add_batch(probs, label=labels)

    for _ in tf.range(tf.cast(train_steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  start_time = time.time()

  train_iterator = iter(dataset_train)
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    train_step(train_iterator)

    current_step = (epoch + 1) * train_steps_per_epoch
    max_steps = train_steps_per_epoch * FLAGS.train_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    # Run evaluation on all evaluation datasets, and compute metrics
    per_pred_results, total_results = utils.evaluate_model_and_compute_metrics(
        strategy,
        eval_datasets,
        steps,
        metrics,
        eval_estimator,
        uncertainty_estimator_fn,
        batch_size,
        available_splits,
        estimator_args=estimator_args,
        call_dataset_iter=False,
        is_deterministic=False,
        num_bins=FLAGS.num_bins,
        use_tpu=use_tpu,
        return_per_pred_results=True)

    # Optionally log to wandb
    if FLAGS.use_wandb:
      wandb.log(total_results, step=epoch)

    with summary_writer.as_default():
      for name, result in total_results.items():
        if result is not None:
          tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      # checkpoint_name = checkpoint.save(os.path.join(
      #     output_dir, 'checkpoint'))
      # logging.info('Saved checkpoint to %s', checkpoint_name)

      # TODO(nband): debug checkpointing
      # Also save Keras model, due to checkpoint.save issue.
      keras_model_name = os.path.join(output_dir, f'keras_model_{epoch + 1}')
      model.save(keras_model_name)
      logging.info('Saved keras model to %s', keras_model_name)

      # Save per-prediction metrics
      utils.save_per_prediction_results(
          output_dir, epoch + 1, per_pred_results, verbose=False)

  # final_checkpoint_name = checkpoint.save(
  #     os.path.join(output_dir, 'checkpoint'))
  # logging.info('Saved last checkpoint to %s', final_checkpoint_name)

  keras_model_name = os.path.join(output_dir,
                                  f'keras_model_{FLAGS.train_epochs}')
  model.save(keras_model_name)
  logging.info('Saved keras model to %s', keras_model_name)

  # Save per-prediction metrics
  utils.save_per_prediction_results(
      output_dir, FLAGS.train_epochs, per_pred_results, verbose=False)

  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
        'fast_weight_lr_multiplier': FLAGS.fast_weight_lr_multiplier,
        'num_mc_samples_eval': FLAGS.num_mc_samples_eval,
    })

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
