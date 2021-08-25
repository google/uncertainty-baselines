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

"""ResNet50 Radial BNN trained on Kaggle's Diabetic Retinopathy Detection.

This script performs variational inference with a Radial posterior,
as introduced in Radial Bayesian Neural Networks (Farquhar et al. 2020).

We use a few additional, noteworthy techniques in training:

1. Normal prior whose mean is tied at the variational posterior's.
    This makes the KL penalty only penalize the weight posterior's standard
    deviation and not its mean.
    The prior's standard deviation can be fixed as a hyperparameter, but is by
    default set to the He initializer stddev: sqrt(2 / fan_in) (Neal 1995).
2. Flipout for lower-variance gradients in convolutional layers and the final
    dense layer (Wen et al., 2018).
3. KL annealing (Bowman et al., 2015).
"""

import os
import time
from pprint import pformat

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorboard.plugins.hparams import api as hp

import uncertainty_baselines as ub
import utils  # local file import

import wandb

import pathlib
from datetime import datetime
from pprint import pformat


DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/radial',
    'The directory where the model weights and '
    'training/evaluation summaries are stored.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')
flags.DEFINE_bool('use_test', False, 'Whether to use a test split.')
flags.DEFINE_string(
  'dr_decision_threshold', 'moderate',
  ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
   "binary classification task. Only affects the APTOS dataset partitioning. "
   "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
   "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))
flags.DEFINE_bool(
  'load_from_checkpoint', False, "Attempt to load from checkpoint")
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
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.DEFINE_bool(
  'load_train_split', True,
  "Should always be enabled - required to load train split of the dataset.")

# Learning rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 4e-4, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60'],
                  'Epochs to decay learning rate by.')

# Radial BNN flags.
flags.DEFINE_integer('num_mc_samples_train', 1,
                     'Number of MC samples used during training.')
flags.DEFINE_integer('num_mc_samples_eval', 5,
                     'Number of MC samples to use for prediction.')
flags.DEFINE_bool(
    'tied_mean_prior', True,
    'If True, fix the mean of the prior to that of the variational posterior. '
    'This causes the KL to only penalize the standard deviation of the weight '
    'posterior, and not its mean.')
flags.DEFINE_integer('kl_annealing_epochs', 200,
                     'Number of epochs over which to anneal the KL term to 1.')
flags.DEFINE_float(
    'prior_stddev', None, 'Sets a fixed stddev for weight prior. '
    'If None, defaults to the He initializer stddev: sqrt(2 / fan_in).')
flags.DEFINE_float(
    'stddev_mean_init', 1e-3,
    'Initializes the mean of the TruncatedNormal from which we sample the '
    'initial posterior standard deviation: '
    'mean = np.log(np.expm1(stddev_mean_init)).')
flags.DEFINE_float(
    'stddev_stddev_init', 0.1,
    'Standard deviation of the TruncatedNormal from which we sample the '
    'initial posterior standard deviation.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,'
    'test respectively). `None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. `minibatch` will use the proportions of each minibatch to '
    'reweight the loss.')
flags.DEFINE_float('l2', 5e-5, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('batch_size', DEFAULT_BATCH_SIZE,
                     'The per-core training/validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

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
      entity="uncertainty-baselines",
      dir=FLAGS.wandb_dir,
      reinit=True,
      name=FLAGS.exp_name,
      group=FLAGS.exp_group)
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(FLAGS, allow_val_change=True)
    output_dir = str(os.path.join(
      FLAGS.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
  else:
    wandb_run = None
    output_dir = FLAGS.output_dir

  tf.io.gfile.makedirs(output_dir)
  logging.info('Saving checkpoints at %s', output_dir)

  # Log Run Hypers
  hypers_dict = {
    'batch_size': FLAGS.batch_size,
    'base_learning_rate': FLAGS.base_learning_rate,
    'one_minus_momentum': FLAGS.one_minus_momentum,
    'l2': FLAGS.l2,
    'stddev_mean_init': FLAGS.stddev_mean_init,
    'stddev_stddev_init': FLAGS.stddev_stddev_init,
  }
  logging.info('Hypers:')
  logging.info(pformat(hypers_dict))

  # Initialize distribution strategy on flag-specified accelerator
  strategy = utils.init_distribution_strategy(
    FLAGS.force_use_cpu, FLAGS.use_gpu, FLAGS.tpu)
  use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  # Only permit use of L2 regularization with a tied mean prior
  if FLAGS.l2 is not None and FLAGS.l2 > 0 and not FLAGS.tied_mean_prior:
    raise NotImplementedError(
        'For a principled objective, L2 regularization should not be used '
        'when the prior mean is untied from the posterior mean.')

  batch_size = FLAGS.batch_size * FLAGS.num_cores

  # Reweighting loss for class imbalance
  class_reweight_mode = FLAGS.class_reweight_mode
  if class_reweight_mode == 'constant':
    class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  else:
    class_weights = None

  # Load in datasets.
  datasets, steps = utils.load_dataset(
    train_batch_size=batch_size, eval_batch_size=batch_size,
    flags=FLAGS, strategy=strategy)
  available_splits = list(datasets.keys())
  test_splits = [split for split in available_splits if 'test' in split]
  eval_splits = [split for split in available_splits
                 if 'validation' in split or 'test' in split]

  # Iterate eval datasets
  eval_datasets = {split: iter(datasets[split]) for split in eval_splits}
  dataset_train = datasets['train']
  train_steps_per_epoch = steps['train']
  train_dataset_size = train_steps_per_epoch * batch_size

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  if FLAGS.prior_stddev is None:
    logging.info(
      'A fixed prior stddev was not supplied. Computing a prior stddev = '
      'sqrt(2 / fan_in) for each layer. This is recommended over providing '
      'a fixed prior stddev.')

  with strategy.scope():
    logging.info('Building Keras ResNet-50 Radial model.')
    model = None
    if FLAGS.load_from_checkpoint:
      initial_epoch, model = utils.load_keras_checkpoints(
        FLAGS.checkpoint_dir, load_ensemble=False, return_epoch=True)
    else:
      initial_epoch = 0
      model = ub.models.resnet50_radial(
        input_shape=utils.load_input_shape(dataset_train),
        num_classes=1,  # binary classification task
        prior_stddev=FLAGS.prior_stddev,
        dataset_size=train_dataset_size,
        stddev_mean_init=FLAGS.stddev_mean_init,
        stddev_stddev_init=FLAGS.stddev_stddev_init,
        tied_mean_prior=FLAGS.tied_mean_prior)
    utils.log_model_init_info(model=model)

    # Linearly scale learning rate and the decay epochs by vanilla settings.
    base_lr = FLAGS.base_learning_rate
    lr_decay_epochs = [
        (int(start_epoch_str) * FLAGS.train_epochs) // DEFAULT_NUM_EPOCHS
        for start_epoch_str in FLAGS.lr_decay_epochs
    ]
    lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
        train_steps_per_epoch,
        base_lr,
        decay_ratio=FLAGS.lr_decay_ratio,
        decay_epochs=lr_decay_epochs,
        warmup_epochs=FLAGS.lr_warmup_epochs)
    optimizer = tf.keras.optimizers.SGD(
        lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=use_tpu,
        num_bins=FLAGS.num_bins,
        use_validation=FLAGS.use_validation,
        available_splits=available_splits)

    # Radial specific metrics
    metrics.update({
        'train/kl': tf.keras.metrics.Mean(),
        'train/kl_scale': tf.keras.metrics.Mean()
    })

    # TODO: debug or remove
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # latest_checkpoint = tf.train.latest_checkpoint(output_dir)
    # if latest_checkpoint:
    #   # checkpoint.restore must be within a strategy.scope()
    #   # so that optimizer slot variables are mirrored.
    #   checkpoint.restore(latest_checkpoint)
    #   logging.info('Loaded checkpoint %s', latest_checkpoint)
    #   initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

  # Define OOD metrics outside the accelerator scope for CPU eval.
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
    'radial', use_ensemble=False, use_tf=True)

  # Wrap our estimator to predict probabilities (apply sigmoid on logits)
  eval_estimator = utils.wrap_retinopathy_estimator(
    model, use_mixed_precision=FLAGS.use_bfloat16, numpy_outputs=False)

  estimator_args = {
    'num_samples': FLAGS.num_mc_samples_eval
  }

  @tf.function
  def train_step(iterator):
    """Training step function."""

    def step_fn(inputs):
      """Per-replica step function."""
      images = inputs['features']
      labels = inputs['labels']

      # For minibatch class reweighting, initialize per-batch loss function
      if class_reweight_mode == 'minibatch':
        batch_loss_fn = utils.get_minibatch_reweighted_loss_fn(labels=labels)
      else:
        batch_loss_fn = loss_fn

      with tf.GradientTape() as tape:
        # TODO: TPU-friendly implem
        # logits_arr = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        #
        # for _ in tf.range(FLAGS.num_mc_samples_train):
        #   logits = model(images, training=True)
        #   # logits = tf.squeeze(logits, axis=-1)
        #   # if FLAGS.use_bfloat16:
        #   #   logits = tf.cast(logits, tf.float32)
        #
        #   logits_arr = logits_arr.write(logits_arr.size(), logits)
        #
        # logits_list = logits_arr.stack()

        # # Pythonic Implem
        if FLAGS.num_mc_samples_train > 1:
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

        filtered_variables = []
        for var in model.trainable_variables:
          # Apply l2 on the BN parameters and bias terms. This
          # excludes only fast weight approximate posterior/prior parameters,
          # but pay caution to their naming scheme.
          if 'bn' in var.name or 'bias' in var.name:
            filtered_variables.append(tf.reshape(var, (-1,)))

        l2_loss = FLAGS.l2 * 2 * tf.nn.l2_loss(
            tf.concat(filtered_variables, axis=0))
        kl = sum(model.losses)
        kl_scale = tf.cast(optimizer.iterations + 1, kl.dtype)
        kl_scale /= train_steps_per_epoch * FLAGS.kl_annealing_epochs
        kl_scale = tf.minimum(1., kl_scale)
        kl_loss = kl_scale * kl

        loss = negative_log_likelihood + l2_loss + kl_loss

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
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
    logging.info('Starting to run epoch: %s', epoch + 1)
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
      strategy, eval_datasets, steps, metrics, eval_estimator,
      uncertainty_estimator_fn, batch_size, available_splits,
      estimator_args=estimator_args, call_dataset_iter=False,
      is_deterministic=False, num_bins=FLAGS.num_bins, use_tpu=use_tpu,
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
      # checkpoint_name = checkpoint.save(
      #     os.path.join(output_dir, 'checkpoint'))
      # logging.info('Saved checkpoint to %s', checkpoint_name)

      # TODO(nband): debug checkpointing
      # Also save Keras model, due to checkpoint.save issue
      keras_model_name = os.path.join(output_dir,
                                      f'keras_model_{epoch + 1}')
      model.save(keras_model_name)
      logging.info('Saved keras model to %s', keras_model_name)

      # Save per-prediction metrics
      utils.save_per_prediction_results(
        output_dir, epoch + 1, per_pred_results, verbose=False)

  # final_checkpoint_name = checkpoint.save(
  #     os.path.join(output_dir, 'checkpoint'),)
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
        'stddev_mean_init': FLAGS.stddev_mean_init,
        'stddev_stddev_init': FLAGS.stddev_stddev_init,
    })

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
