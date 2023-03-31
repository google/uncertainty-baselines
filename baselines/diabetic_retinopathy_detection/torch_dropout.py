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

"""PyTorch ResNet50 with Monte Carlo Dropout (Gal and Ghahramani 2016) on Kaggle's Diabetic Retinopathy Detection dataset."""

import os
import time

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

import torch_utils  # local file import
import uncertainty_baselines as ub
import utils  # local file import
from tensorboard.plugins.hparams import api as hp

# Hide any GPUs form TensorFlow. Otherwise TF might reserve memory and make
# PyTorch crash with CUDA OoM error.
tf.config.experimental.set_visible_devices([], 'GPU')

DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/dropout_torch',
    'The directory where the model weights and training/evaluation summaries are '
    'stored. If you aim to use these as trained models for dropoutensemble.py, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')

# Learning Rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 4e-4, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_float(
    'lr_warmup_epochs', 20,
    'Number of epochs to the first LR peak using cosine annealing with warm '
    'restarts LR scheduler.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('l2', 5e-5, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('train_batch_size', 16,
                     'The per-core training batch size.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The per-core validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

# Dropout-related flags.
flags.DEFINE_float('dropout_rate', 0.1, 'Dropout rate, between [0.0, 1.0).')
flags.DEFINE_integer('num_dropout_samples_eval', 10,
                     'Number of dropout samples to use for prediction.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True,
                  'Whether to run on (a single) GPU or otherwise CPU.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)

  # Set seeds
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  # Resolve CUDA device(s)
  if FLAGS.use_gpu and torch.cuda.is_available():
    print('Running model with CUDA.')
    device = 'cuda:0'
  else:
    print('Running model on CPU.')
    device = 'cpu'

  train_batch_size = FLAGS.train_batch_size
  eval_batch_size = FLAGS.eval_batch_size // FLAGS.num_dropout_samples_eval

  # As per the Kaggle challenge, we have split sizes:
  # train: 35,126
  # validation: 10,906
  # test: 42,670
  ds_info = tfds.builder('diabetic_retinopathy_detection').info
  steps_per_epoch = ds_info.splits['train'].num_examples // train_batch_size
  steps_per_validation_eval = (
      ds_info.splits['validation'].num_examples // eval_batch_size)
  steps_per_test_eval = ds_info.splits['test'].num_examples // eval_batch_size

  data_dir = FLAGS.data_dir

  dataset_train_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='train', data_dir=data_dir)
  dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

  dataset_validation_builder = ub.datasets.get(
      'diabetic_retinopathy_detection',
      split='validation',
      data_dir=data_dir,
      is_training=not FLAGS.use_validation)
  validation_batch_size = (
      eval_batch_size if FLAGS.use_validation else train_batch_size)
  dataset_validation = dataset_validation_builder.load(
      batch_size=validation_batch_size)
  if not FLAGS.use_validation:
    # Note that this will not create any mixed batches of train and validation
    # images.
    dataset_train = dataset_train.concatenate(dataset_validation)

  dataset_test_builder = ub.datasets.get(
      'diabetic_retinopathy_detection', split='test', data_dir=data_dir)
  dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  # MC Dropout ResNet50 based on PyTorch Vision implementation
  logging.info('Building Torch ResNet-50 MC Dropout model.')
  model = ub.models.resnet50_dropout_torch(
      num_classes=1, dropout_rate=FLAGS.dropout_rate)
  logging.info('Model number of weights: %s',
               torch_utils.count_parameters(model))

  # Linearly scale learning rate and the decay epochs by vanilla settings.
  base_lr = FLAGS.base_learning_rate
  optimizer = torch.optim.SGD(
      model.parameters(),
      lr=base_lr,
      momentum=1.0 - FLAGS.one_minus_momentum,
      nesterov=True)
  steps_to_lr_peak = int(steps_per_epoch * FLAGS.lr_warmup_epochs)
  scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer, steps_to_lr_peak, T_mult=2)

  model = model.to(device)

  metrics = utils.get_diabetic_retinopathy_base_metrics(
      use_tpu=False,
      num_bins=FLAGS.num_bins,
      use_validation=FLAGS.use_validation)

  # Define additional metrics that would fail in a TF TPU implementation.
  metrics.update(
      utils.get_diabetic_retinopathy_cpu_metrics(
          use_validation=FLAGS.use_validation))

  # Initialize loss function based on class reweighting setting
  loss_fn = torch.nn.BCELoss()
  sigmoid = torch.nn.Sigmoid()
  max_steps = steps_per_epoch * FLAGS.train_epochs
  image_h = 512
  image_w = 512

  def run_train_epoch(iterator):

    def train_step(inputs):
      images = inputs['features']
      labels = inputs['labels']
      images = torch.from_numpy(images._numpy()).view(train_batch_size, 3,  # pylint: disable=protected-access
                                                      image_h,
                                                      image_w).to(device)
      labels = torch.from_numpy(labels._numpy()).to(device).float()  # pylint: disable=protected-access

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward
      logits = model(images)
      probs = sigmoid(logits).squeeze(-1)

      # Add L2 regularization loss to NLL
      negative_log_likelihood = loss_fn(probs, labels)
      l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
      loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

      # Backward/optimizer
      loss.backward()
      optimizer.step()

      # Convert to NumPy for metrics updates
      loss = loss.detach()
      negative_log_likelihood = negative_log_likelihood.detach()
      labels = labels.detach()
      probs = probs.detach()

      if device != 'cpu':
        loss = loss.cpu()
        negative_log_likelihood = negative_log_likelihood.cpu()
        labels = labels.cpu()
        probs = probs.cpu()

      loss = loss.numpy()
      negative_log_likelihood = negative_log_likelihood.numpy()
      labels = labels.numpy()
      probs = probs.numpy()

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
          negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/auprc'].update_state(labels, probs)
      metrics['train/auroc'].update_state(labels, probs)
      metrics['train/ece'].add_batch(probs, label=labels)

    for step in range(steps_per_epoch):
      train_step(next(iterator))

      if step % 100 == 0:
        current_step = (epoch + 1) * step
        time_elapsed = time.time() - start_time
        steps_per_sec = float(current_step) / time_elapsed
        eta_seconds = (max_steps -
                       current_step) / steps_per_sec if steps_per_sec else 0
        message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
                   'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                       current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                       steps_per_sec, eta_seconds / 60, time_elapsed / 60))
        logging.info(message)

  def run_eval_epoch(iterator, dataset_split, num_steps):

    def eval_step(inputs, model):
      images = inputs['features']
      labels = inputs['labels']
      images = torch.from_numpy(images._numpy()).view(eval_batch_size, 3,  # pylint: disable=protected-access
                                                      image_h,
                                                      image_w).to(device)
      labels = torch.from_numpy(
          labels._numpy()).to(device).float().unsqueeze(-1)  # pylint: disable=protected-access

      with torch.no_grad():
        logits = torch.stack(
            [model(images) for _ in range(FLAGS.num_dropout_samples_eval)],
            dim=-1)

      # Logits dimension is (batch_size, 1, num_dropout_samples).
      logits = logits.squeeze()

      # It is now (batch_size, num_dropout_samples).
      probs = sigmoid(logits)

      # labels_tiled shape is (batch_size, num_dropout_samples).
      labels_tiled = torch.tile(labels, (1, FLAGS.num_dropout_samples_eval))

      log_likelihoods = -loss_fn(probs, labels_tiled)
      negative_log_likelihood = torch.mean(
          -torch.logsumexp(log_likelihoods, dim=-1) +
          torch.log(torch.tensor(float(FLAGS.num_dropout_samples_eval))))

      probs = torch.mean(probs, dim=-1)

      # Convert to NumPy for metrics updates
      negative_log_likelihood = negative_log_likelihood.detach()
      labels = labels.detach()
      probs = probs.detach()

      if device != 'cpu':
        negative_log_likelihood = negative_log_likelihood.cpu()
        labels = labels.cpu()
        probs = probs.cpu()

      negative_log_likelihood = negative_log_likelihood.numpy()
      labels = labels.numpy()
      probs = probs.numpy()

      metrics[dataset_split +
              '/negative_log_likelihood'].update_state(negative_log_likelihood)
      metrics[dataset_split + '/accuracy'].update_state(labels, probs)
      metrics[dataset_split + '/auprc'].update_state(labels, probs)
      metrics[dataset_split + '/auroc'].update_state(labels, probs)
      metrics[dataset_split + '/ece'].add_batch(probs, label=labels)

    for _ in range(num_steps):
      eval_step(next(iterator), model=model)

  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})
  start_time = time.time()
  initial_epoch = 0
  train_iterator = iter(dataset_train)
  model.train()
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch + 1)

    run_train_epoch(train_iterator)

    if FLAGS.use_validation:
      validation_iterator = iter(dataset_validation)
      logging.info('Starting to run validation eval at epoch: %s', epoch + 1)
      run_eval_epoch(validation_iterator, 'validation',
                     steps_per_validation_eval)

    test_iterator = iter(dataset_test)
    logging.info('Starting to run test eval at epoch: %s', epoch + 1)
    test_start_time = time.time()
    run_eval_epoch(test_iterator, 'test', steps_per_test_eval)
    ms_per_example = (time.time() - test_start_time) * 1e6 / eval_batch_size
    metrics['test/ms_per_example'].update_state(ms_per_example)

    # Step scheduler
    scheduler.step()

    # Log and write to summary the epoch metrics
    utils.log_epoch_metrics(metrics=metrics, use_tpu=False)
    total_results = {name: metric.result() for name, metric in metrics.items()}
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

      checkpoint_path = os.path.join(FLAGS.output_dir, f'model_{epoch + 1}.pt')
      torch_utils.checkpoint_torch_model(
          model=model,
          optimizer=optimizer,
          epoch=epoch + 1,
          checkpoint_path=checkpoint_path)
      logging.info('Saved Torch checkpoint to %s', checkpoint_path)

  final_checkpoint_path = os.path.join(FLAGS.output_dir,
                                       f'model_{FLAGS.train_epochs}.pt')
  torch_utils.checkpoint_torch_model(
      model=model,
      optimizer=optimizer,
      epoch=FLAGS.train_epochs,
      checkpoint_path=final_checkpoint_path)
  logging.info('Saved last checkpoint to %s', final_checkpoint_path)

  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'dropout_rate': FLAGS.dropout_rate,
        'l2': FLAGS.l2,
        'lr_warmup_epochs': FLAGS.lr_warmup_epochs
    })


if __name__ == '__main__':
  app.run(main)
