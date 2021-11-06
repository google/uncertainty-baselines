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

"""Binary to train a deterministic MPNN model."""
import collections
import dataclasses
import json
import logging
import os
import time
from typing import Any, Dict, Optional, Sequence

from absl import app
from absl import flags
import robustness_metrics as rm
import tensorflow as tf
from tensorflow_addons import losses as tfa_losses
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
from uncertainty_baselines.datasets.drug_cardiotoxicity import DrugCardiotoxicityDataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the TFRecord datasets for Drug Cardiotoxicity.')
flags.DEFINE_integer('num_heads', 2, 'Number of classification heads.')
flags.DEFINE_string('output_dir', None, 'Output directory.')
flags.DEFINE_string('job_base_dir', None,
                    'Output directory for the umbrella job, which may have '
                    'multilple models from ensemble strategy or Vizier.'
                    'It can be set to None when model_dir is specified.'
                    'When we use Vizier and there is no model_dir specified, '
                    'this flag should be specified.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs.')
flags.DEFINE_boolean('use_gpu', True, 'If True, uses GPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', '', 'TPU master.')
# Parameter flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('num_layers', 2, 'Number of message passing layers.')
flags.DEFINE_integer('message_layer_size', 32,
                     'Number of units in message layers.')
flags.DEFINE_integer('readout_layer_size', 32,
                     'Number of units in the readout layer.')

# Loss type.
flags.DEFINE_enum('loss_type', 'xent', ['xent', 'focal'],
                  'Type of loss function to use.')


# TODO(kehanghan): Factor this out to a shared `utils.py`.
@dataclasses.dataclass(frozen=True)
class ModelParameters:
  """Model Parameters used in MPNN architecture.

  Attributes:
    num_heads: Int, number of output classes.
    num_layers: Int, number of Message Passing layers.
    message_layer_size: Int, dimension of message representation.
    readout_layer_size: Int, dimension of graph level readout representation.
    use_gp_layer: Bool, whether to use Gaussian Process layer as classifier.
    learning_rate: Float, learning rate.
    num_epochs: Int, number of epoch for the entire training process.
    steps_per_epoch: Int, number of training batches to take in one epoch.
  """
  num_heads: int = 2
  num_layers: int = 2
  message_layer_size: int = 32
  readout_layer_size: int = 32
  use_gp_layer: bool = False
  learning_rate: float = 0.001
  num_epochs: int = 100
  steps_per_epoch: Optional[int] = None


def get_tpu_strategy(master: str) -> tf.distribute.TPUStrategy:
  """Builds a TPU distribution strategy."""
  logging.info('TPU master: %s', master)
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(master)
  tf.config.experimental_connect_to_cluster(resolver)
  tf.tpu.experimental.initialize_tpu_system(resolver)
  return tf.distribute.TPUStrategy(resolver)


def write_params(params: Any, filename: str):
  """Writes a dataclass to disk."""
  tf.io.gfile.makedirs(os.path.dirname(filename))
  with tf.io.gfile.GFile(filename, 'w') as f:
    json.dump(params, f, indent=2)


@tf.function
def train_step(model, strategy, iterator, steps_per_epoch, optimizer, metrics,
               loss_type):
  """Training StepFn."""

  def step_fn(inputs, metrics, model, strategy, optimizer, loss_type):
    """Per-Replica StepFn."""
    if len(inputs) == 3:
      features, labels, sample_weights = inputs
    else:
      features, labels = inputs
      sample_weights = 1

    with tf.GradientTape() as tape:
      probs = model(features, training=True)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.categorical_crossentropy(labels, probs) *
          sample_weights)

      l2_loss = sum(model.losses)
      if loss_type == 'focal':
        focal_loss_fn = tfa_losses.SigmoidFocalCrossEntropy()
        focal_loss = tf.reduce_mean(
            focal_loss_fn(labels, probs) * sample_weights)
        loss = focal_loss + l2_loss
      else:
        loss = negative_log_likelihood + l2_loss
      # Scale the loss given the tf.distribute.Strategy will reduce sum all
      # gradients. See details in
      # https://www.tensorflow.org/tutorials/distribute/custom_training#define_the_loss_function
      scaled_loss = loss / strategy.num_replicas_in_sync

    grads = tape.gradient(scaled_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    metrics['train/loss'].update_state(loss)
    metrics['train/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics['train/accuracy'].update_state(labels, probs)
    metrics['train/roc_auc'].update_state(labels[:, 1], probs[:, 1])

  for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
    strategy.run(
        step_fn,
        args=(next(iterator), metrics, model, strategy, optimizer, loss_type))


@tf.function
def eval_step(model, strategy, iterator, dataset_name, num_steps, metrics):
  """Evaluation StepFn."""

  def step_fn(model, inputs, dataset_name, metrics):
    """Per-Replica StepFn."""
    if len(inputs) == 3:
      features, labels, _ = inputs
    else:
      features, labels = inputs

    probs = model(features, training=False)
    negative_log_likelihood = tf.reduce_mean(
        tf.keras.losses.categorical_crossentropy(labels, probs))

    metrics[f'{dataset_name}/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics[f'{dataset_name}/accuracy'].update_state(labels, probs)
    metrics[f'{dataset_name}/roc_auc'].update_state(
        labels[:, 1], probs[:, 1])
    metrics[f'{dataset_name}/ece'].add_batch(probs[:, 1], label=labels[:, 1])
    metrics[f'{dataset_name}/brier'].add_batch(probs, label=labels[:, 1])

  for _ in tf.range(tf.cast(num_steps, tf.int32)):
    strategy.run(
        step_fn,
        args=(model, next(iterator), dataset_name, metrics))


def get_metric_result_value(metric):
  """Gets the value of the input metric current result."""
  result = metric.result()
  if isinstance(metric, tf.keras.metrics.Metric):
    return result.numpy()
  elif isinstance(metric, rm.metrics.Metric):
    return list(result.values())[0]
  else:
    raise ValueError(f'Metric type {type(metric)} not supported.')


def run(train_dataset: tf.data.Dataset, eval_datasets: Dict[str,
                                                            tf.data.Dataset],
        steps_per_eval: Dict[str, int], params: ModelParameters, model_dir: str,
        strategy: tf.distribute.Strategy,
        summary_writer: tf.summary.SummaryWriter,
        loss_type: str):
  """Trains and evaluates the model."""
  with strategy.scope():
    model = ub.models.mpnn(
        nodes_shape=train_dataset.element_spec[0]['atoms'].shape[1:],
        edges_shape=train_dataset.element_spec[0]['pairs'].shape[1:],
        num_heads=params.num_heads,
        num_layers=params.num_layers,
        message_layer_size=params.message_layer_size,
        readout_layer_size=params.readout_layer_size,
        use_gp_layer=params.use_gp_layer)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=params.learning_rate)
    metrics = {
        'train/negative_log_likelihood': tf.keras.metrics.Mean(),
        'train/accuracy': tf.keras.metrics.CategoricalAccuracy(),
        'train/loss': tf.keras.metrics.Mean(),
        'train/roc_auc': tf.keras.metrics.AUC(),
    }

    for dataset_name in eval_datasets:
      metrics[
          f'{dataset_name}/accuracy'] = tf.keras.metrics.CategoricalAccuracy()
      metrics[f'{dataset_name}/roc_auc'] = tf.keras.metrics.AUC()
      metrics[
          f'{dataset_name}/negative_log_likelihood'] = tf.keras.metrics.Mean()
      if dataset_name == 'test2':
        ece_num_bins = 5
      else:
        ece_num_bins = 10
      metrics[f'{dataset_name}/ece'] = rm.metrics.ExpectedCalibrationError(
          num_bins=ece_num_bins)
      metrics[f'{dataset_name}/brier'] = rm.metrics.Brier()

  # Makes datasets into distributed version.
  train_dataset = strategy.experimental_distribute_dataset(train_dataset)
  eval_datasets = {
      ds_name: strategy.experimental_distribute_dataset(ds)
      for ds_name, ds in eval_datasets.items()
  }
  logging.info('Number of replicas in sync: %s', strategy.num_replicas_in_sync)

  train_iterator = iter(train_dataset)
  start_time = time.time()
  metrics_history = collections.defaultdict(list)
  for epoch in range(params.num_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    train_step(model, strategy, train_iterator, params.steps_per_epoch,
               optimizer, metrics, loss_type)

    current_step = (epoch + 1) * params.steps_per_epoch
    max_steps = params.steps_per_epoch * params.num_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, params.num_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    # Start evaluation.
    logging.info('Starting to run eval at epoch: %s', epoch)
    for dataset_name, eval_dataset in eval_datasets.items():
      eval_iterator = iter(eval_dataset)
      eval_step(model, strategy, eval_iterator, dataset_name,
                steps_per_eval[dataset_name], metrics)

    metrics_history['epoch'].append(epoch + 1)
    with summary_writer.as_default():
      for name, metric in metrics.items():
        result = get_metric_result_value(metric)
        tf.summary.scalar(name, result, step=epoch + 1)
        metrics_history[name].append(str(result))


    for metric in metrics.values():
      metric.reset_states()

    model.save(os.path.join(model_dir, f'model_{epoch + 1}'), overwrite=True)

  write_params(metrics_history,
               os.path.join(model_dir, 'metrics_history.json'))


def main(argv: Sequence[str]):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  if not FLAGS.use_gpu:
    logging.info('Using TPU for training.')
    strategy = get_tpu_strategy(FLAGS.tpu)
  else:
    logging.info('Using GPU for training.')
    strategy = tf.distribute.MirroredStrategy()

  train_dataset_builder = DrugCardiotoxicityDataset(
      split=tfds.Split.TRAIN,
      data_dir=FLAGS.data_dir)
  train_dataset = train_dataset_builder.load(
      batch_size=FLAGS.batch_size).map(lambda x: (x['features'], x['labels']))

  ds_info = train_dataset_builder.tfds_info
  max_nodes = ds_info.metadata['max_nodes']
  node_features = ds_info.metadata['node_features']
  edge_features = ds_info.metadata['edge_features']
  steps_per_epoch = train_dataset_builder.num_examples // FLAGS.batch_size

  eval_datasets = {}
  steps_per_eval = {}
  test_iid_builder = DrugCardiotoxicityDataset(
      split=tfds.Split.VALIDATION,
      data_dir=FLAGS.data_dir,
      drop_remainder=False)
  test_iid_dataset = test_iid_builder.load(
      batch_size=FLAGS.batch_size).map(lambda x: (x['features'], x['labels']))

  eval_datasets['tune'] = test_iid_dataset
  steps_per_eval['tune'] = 1 + test_iid_builder.num_examples//FLAGS.batch_size

  test_ood1_builder = DrugCardiotoxicityDataset(
      split=tfds.Split.TEST,
      data_dir=FLAGS.data_dir,
      drop_remainder=False)
  test_ood1_dataset = test_ood1_builder.load(
      batch_size=FLAGS.batch_size).map(lambda x: (x['features'], x['labels']))

  eval_datasets['test1'] = test_ood1_dataset
  steps_per_eval['test1'] = 1 + test_ood1_builder.num_examples//FLAGS.batch_size

  test_ood2_builder = DrugCardiotoxicityDataset(
      split=tfds.Split('test2'),
      data_dir=FLAGS.data_dir,
      is_training=False,
      drop_remainder=False)
  test_ood2_dataset = test_ood2_builder.load(
      batch_size=FLAGS.batch_size).map(lambda x: (x['features'], x['labels']))

  eval_datasets['test2'] = test_ood2_dataset
  steps_per_eval['test2'] = 1 + test_ood2_builder.num_examples//FLAGS.batch_size

  logging.info('Steps for eval datasets: %s', steps_per_eval)

  params = ModelParameters(
      num_heads=FLAGS.num_heads,
      num_layers=FLAGS.num_layers,
      message_layer_size=FLAGS.message_layer_size,
      readout_layer_size=FLAGS.readout_layer_size,
      use_gp_layer=False,
      learning_rate=FLAGS.learning_rate,
      num_epochs=FLAGS.num_epochs,
      steps_per_epoch=steps_per_epoch)

  model_dir = FLAGS.output_dir
  write_params(
      dataclasses.asdict(params), os.path.join(model_dir, 'params.json'))

  summary_writer = tf.summary.create_file_writer(
      os.path.join(model_dir, 'summaries'))
  run(train_dataset=train_dataset,
      eval_datasets=eval_datasets,
      steps_per_eval=steps_per_eval,
      params=params,
      model_dir=model_dir,
      strategy=strategy,
      summary_writer=summary_writer,
      loss_type=FLAGS.loss_type)


if __name__ == '__main__':
  app.run(main)
