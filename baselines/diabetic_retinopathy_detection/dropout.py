# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""ResNet50 with Monte Carlo Dropout (Gal and Ghahramani 2016) on Kaggle's Diabetic Retinopathy Detection dataset."""

import os
import time

import tensorflow as tf
from absl import app
from absl import flags
from absl import logging
from tensorboard.plugins.hparams import api as hp
import wandb

import uncertainty_baselines as ub
import utils
from pprint import pformat
from datetime import datetime
import pathlib

DEFAULT_TRAIN_BATCH_SIZE = 16
DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/dropout',
    'The directory where the model weights and training/evaluation summaries are '
    'stored. If you aim to use these as trained models for dropoutensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')
flags.DEFINE_bool('use_test', False, 'Whether to use a test split.')
flags.DEFINE_string(
  'preproc_builder_config', 'btgraham-300',
  ("Determines the preprocessing procedure for the images. Supported options: "
   "{btgraham-300, blur-3-btgraham-300, blur-5-btgraham-300, "
   "blur-10-btgraham-300, blur-20-btgraham-300}."))
flags.DEFINE_string(
  'dr_decision_threshold', 'moderate',
  ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
   "binary classification task. Only affects the APTOS dataset partitioning. "
   "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
   "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))
flags.DEFINE_bool(
  'load_from_checkpoint', False, "Attempt to load from checkpoint")
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
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.DEFINE_bool(
  'load_train_split', True,
  "Should always be enabled - required to load train split of the dataset.")

# Learning Rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 0.027250, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum',  0.035193, 'Optimizer momentum.')
flags.DEFINE_integer(
    'lr_warmup_epochs', 1,
    'Number of epochs for a linear warmup to the initial '
    'learning rate. Use 0 to do no warmup.')
flags.DEFINE_float('lr_decay_ratio', 0.2, 'Amount to decay learning rate.')
flags.DEFINE_list('lr_decay_epochs', ['30', '60'],
                  'Epochs at which we decay learning rate.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,'
    'test respectively). `None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. `minibatch` will use the proportions of each minibatch to '
    'reweight the loss.')
flags.DEFINE_float('l2', 0.000014128, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('per_core_batch_size', 16,
                     'The per-core batch size for both training '
                     'and evaluation.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Dropout-related flags.
flags.DEFINE_float('dropout_rate', 0.17798, 'Dropout rate, between [0.0, 1.0).')
flags.DEFINE_integer('num_dropout_samples_eval', 5,
                     'Number of dropout samples to use for prediction.')
flags.DEFINE_bool(
    'filterwise_dropout', False,
    'Dropout whole convolutional filters instead of '
    'individual values in the feature map.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('force_use_cpu', False, 'If True, force usage of CPU.')
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string(
    'tpu', None,
    'Name of the TPU. Only used if force_use_cpu and use_gpu are both False.')
FLAGS = flags.FLAGS


# Load from checkpoint

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
    'per_core_batch_size': FLAGS.per_core_batch_size,
    'base_learning_rate': FLAGS.base_learning_rate,
    'one_minus_momentum': FLAGS.one_minus_momentum,
    'dropout_rate': FLAGS.dropout_rate,
    'l2': FLAGS.l2,
  }
  logging.info('Hypers:')
  logging.info(pformat(hypers_dict))

  # Initialize distribution strategy on flag-specified accelerator
  strategy = utils.init_distribution_strategy(
    FLAGS.force_use_cpu, FLAGS.use_gpu, FLAGS.tpu)
  use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)

  per_core_batch_size = (FLAGS.per_core_batch_size * FLAGS.num_cores)

  # Reweighting loss for class imbalance
  class_reweight_mode = FLAGS.class_reweight_mode
  if class_reweight_mode == 'constant':
    class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  else:
    class_weights = None

  # Load in datasets.
  datasets, steps = utils.load_dataset(
    train_batch_size=per_core_batch_size, eval_batch_size=per_core_batch_size,
    flags=FLAGS, strategy=strategy)
  available_splits = list(datasets.keys())
  test_splits = [split for split in available_splits if 'test' in split]
  eval_splits = [split for split in available_splits
                 if 'validation' in split or 'test' in split]

  # Iterate eval datasets
  eval_datasets = {split: iter(datasets[split]) for split in eval_splits}
  dataset_train = datasets['train']
  train_steps_per_epoch = steps['train']

  if FLAGS.use_bfloat16:
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    tf.keras.mixed_precision.experimental.set_policy(policy)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building Keras ResNet-50 MC Dropout model.')
    model = None
    if FLAGS.load_from_checkpoint:
      initial_epoch, model = utils.load_keras_checkpoints(
        FLAGS.checkpoint_dir, load_ensemble=False, return_epoch=True)
    else:
      initial_epoch = 0
      model = ub.models.resnet50_dropout(
          input_shape=utils.load_input_shape(dataset_train),
          num_classes=1,  # binary classification task
          dropout_rate=FLAGS.dropout_rate,
          filterwise_dropout=FLAGS.filterwise_dropout)
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

    # TODO: debug or remove
    # checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    # latest_checkpoint = tf.train.latest_checkpoint(output_dir)
    # if latest_checkpoint:
    #   # checkpoint.restore must be within a strategy.scope()
    #   # so that optimizer slot variables are mirrored.
    #   checkpoint.restore(latest_checkpoint)
    #   logging.info('Loaded checkpoint %s', latest_checkpoint)
    #   initial_epoch = optimizer.iterations.numpy() // train_steps_per_epoch

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
    'dropout', use_ensemble=False, use_tf=True)

  # Wrap our estimator to predict probabilities (apply sigmoid on logits)
  eval_estimator = utils.wrap_retinopathy_estimator(
    model, use_mixed_precision=FLAGS.use_bfloat16, numpy_outputs=False)

  estimator_args = {
    'num_samples': FLAGS.num_dropout_samples_eval
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
        logits = model(images, training=True)
        if FLAGS.use_bfloat16:
          logits = tf.cast(logits, tf.float32)

        negative_log_likelihood = tf.reduce_mean(
            batch_loss_fn(
                y_true=tf.expand_dims(labels, axis=-1),
                y_pred=logits,
                from_logits=True))
        l2_loss = sum(model.losses)
        loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, model.trainable_variables)
      optimizer.apply_gradients(zip(grads, model.trainable_variables))
      probs = tf.nn.sigmoid(logits)

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
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
      uncertainty_estimator_fn, per_core_batch_size, available_splits,
      estimator_args=estimator_args, call_dataset_iter=False,
      is_deterministic=False, num_bins=FLAGS.num_bins,
      use_tpu=use_tpu, return_per_pred_results=True)

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
      'per_core_batch_size': FLAGS.per_core_batch_size,
      'base_learning_rate': FLAGS.base_learning_rate,
      'one_minus_momentum': FLAGS.one_minus_momentum,
      'dropout_rate': FLAGS.dropout_rate,
      'l2': FLAGS.l2,
    })

  if wandb_run is not None:
    wandb_run.finish()


if __name__ == '__main__':
  app.run(main)
