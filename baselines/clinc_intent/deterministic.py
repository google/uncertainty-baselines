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

"""TextCNN trained with maximum likelihood."""

import os
import time
from absl import app
from absl import flags
from absl import logging

import numpy as np
import robustness_metrics as rm
import tensorflow as tf
import uncertainty_baselines as ub
import bert_utils  # local file import from baselines.clinc_intent
import deterministic_model_textcnn as cnn_model  # local file import from baselines.clinc_intent
from tensorboard.plugins.hparams import api as hp

# Data flags
flags.DEFINE_string(
    'data_dir', None,
    'Directory containing the TFRecord datasets and the tokenizer for Clinc '
    'Intent Detection Data.')

# Model flags
flags.DEFINE_string('model_family', 'bert',
                    'Types of model to use. Can be either TextCNN or BERT.')

# Model flags, TextCNN
flags.DEFINE_list('filter_sizes', ['3', '4', '5'],
                  'Comma-separated filter sizes.')
flags.DEFINE_integer('num_filters', 64, 'Number of filters per filter size.')
flags.DEFINE_integer('embedding_size', 300,
                     'Dimensionality of character embedding.')
flags.DEFINE_float('dropout_rate', 0.2,
                   'Fraction of units to drop in the convolutional layers.')
flags.DEFINE_float('l2', 1e-4,
                   'L2 regularization coefficient for the output layer.')
flags.DEFINE_string('word_embedding_dir', None,
                    'Directory to word embedding npy file.')

# Model flags, BERT.
flags.DEFINE_string(
    'bert_dir', None,
    'Directory to BERT pre-trained checkpoints and config files.')
flags.DEFINE_string(
    'bert_ckpt_dir', None, 'Directory to BERT pre-trained checkpoints. '
    'If None then then default to {bert_dir}/bert_model.ckpt.')
flags.DEFINE_string(
    'bert_config_dir', None, 'Directory to BERT config files. '
    'If None then then default to {bert_dir}/bert_config.json.')

# Optimization and evaluation flags
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_integer('per_core_batch_size', 64, 'Batch size per TPU core/GPU.')
flags.DEFINE_integer('eval_batch_size', 512, 'Batch size for CPU evaluation.')
flags.DEFINE_float(
    'base_learning_rate', 1e-4,
    'Base learning rate when total batch size is 128. It is '
    'scaled by the ratio of the total batch size to 128.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')
flags.DEFINE_integer(
    'checkpoint_interval', 150,
    'Number of epochs between saving checkpoints. Use -1 to '
    'never save checkpoints.')
flags.DEFINE_integer('evaluation_interval', 5,
                     'Number of epochs between evaluation.')
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')
flags.DEFINE_string('output_dir', '/tmp/clinc_intent', 'Output directory.')
flags.DEFINE_integer('train_epochs', 150, 'Number of training epochs.')
flags.DEFINE_float(
    'warmup_proportion', 0.1,
    'Proportion of training to perform linear learning rate warmup for. '
    'E.g., 0.1 = 10% of training.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_bool('use_bfloat16', False, 'Whether to use mixed precision.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
flags.DEFINE_string('tpu', None,
                    'Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS


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


def create_feature_and_label(inputs, feature_size, model_family):
  """Creates features and labels from model inputs."""
  if model_family.lower() == 'bert':
    features, labels = bert_utils.create_feature_and_label(inputs, feature_size)
  else:
    features = inputs['features']
    labels = inputs['labels']
  return features, labels


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
  # vocab_size is total number of valid tokens plus the out-of-vocabulary token.
  vocab_size = ind_dataset_builder.tokenizer.num_words + 1

  steps_per_epoch = train_dataset_builder.num_examples // batch_size

  test_datasets = {}
  steps_per_eval = {}
  for dataset_name, dataset_builder in dataset_builders.items():
    test_datasets[dataset_name] = dataset_builder.load(
        batch_size=FLAGS.eval_batch_size)
    steps_per_eval[dataset_name] = (
        dataset_builder.num_examples // FLAGS.eval_batch_size)

  if FLAGS.use_bfloat16:
    tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  premade_embedding_array = None
  if FLAGS.word_embedding_dir:
    with tf.io.gfile.GFile(FLAGS.word_embedding_dir, 'rb') as embedding_file:
      premade_embedding_array = np.load(embedding_file)

  with strategy.scope():
    logging.info('Building %s model', FLAGS.model_family)
    if FLAGS.model_family.lower() == 'textcnn':
      model = cnn_model.textcnn(
          filter_sizes=[int(x) for x in FLAGS.filter_sizes],
          num_filters=FLAGS.num_filters,
          num_classes=num_classes,
          feature_size=feature_size,
          vocab_size=vocab_size,
          embed_size=FLAGS.embedding_size,
          dropout_rate=FLAGS.dropout_rate,
          l2=FLAGS.l2,
          premade_embedding_arr=premade_embedding_array)
      optimizer = tf.keras.optimizers.Adam(
          FLAGS.base_learning_rate, beta_1=1.0 - FLAGS.one_minus_momentum)
    elif FLAGS.model_family.lower() == 'bert':
      bert_config_dir, bert_ckpt_dir = resolve_bert_ckpt_and_config_dir(
          FLAGS.bert_dir, FLAGS.bert_config_dir, FLAGS.bert_ckpt_dir)
      bert_config = bert_utils.create_config(bert_config_dir)
      model, bert_encoder = ub.models.bert_model(
          num_classes=num_classes,
          max_seq_length=feature_size,
          bert_config=bert_config)
      # Create an AdamW optimizer with beta_2=0.999, epsilon=1e-6.
      optimizer = bert_utils.create_optimizer(
          FLAGS.base_learning_rate,
          steps_per_epoch=steps_per_epoch,
          epochs=FLAGS.train_epochs,
          warmup_proportion=FLAGS.warmup_proportion,
          beta_1=1.0 - FLAGS.one_minus_momentum)
    else:
      raise ValueError('model_family ({}) can only be TextCNN or BERT.'.format(
          FLAGS.model_family))

    logging.info('Model input shape: %s', model.input_shape)
    logging.info('Model output shape: %s', model.output_shape)
    logging.info('Model number of weights: %s', model.count_params())

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
    }

    for dataset_name, test_dataset in test_datasets.items():
      if dataset_name != 'clean':
        metrics.update({
            'test/nll_{}'.format(dataset_name):
                tf.keras.metrics.Mean(),
            'test/accuracy_{}'.format(dataset_name):
                tf.keras.metrics.SparseCategoricalAccuracy(),
            'test/ece_{}'.format(dataset_name):
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins)
        })

    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    latest_checkpoint = tf.train.latest_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
      # checkpoint.restore must be within a strategy.scope() so that optimizer
      # slot variables are mirrored.
      checkpoint.restore(latest_checkpoint)
      logging.info('Loaded checkpoint %s', latest_checkpoint)
      initial_epoch = optimizer.iterations.numpy() // steps_per_epoch
    elif FLAGS.model_family.lower() == 'bert':
      # load BERT from initial checkpoint
      bert_checkpoint = tf.train.Checkpoint(model=bert_encoder)
      bert_checkpoint.restore(
          bert_ckpt_dir).assert_existing_objects_matched()
      logging.info('Loaded BERT checkpoint %s', bert_ckpt_dir)

  # Finally, define OOD metrics outside the accelerator scope for CPU eval.
  metrics.update({'test/auroc_all': tf.keras.metrics.AUC(curve='ROC'),
                  'test/auprc_all': tf.keras.metrics.AUC(curve='PR')})

  @tf.function
  def train_step(iterator):
    """Training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      features, labels = create_feature_and_label(
          inputs, feature_size, model_family=FLAGS.model_family)

      with tf.GradientTape() as tape:
        # Set learning phase to enable dropout etc during training.
        logits = model(features, training=True)
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
      features, labels = create_feature_and_label(
          inputs, feature_size, model_family=FLAGS.model_family)

      # Set learning phase to disable dropout etc during eval.
      logits = model(features, training=False)
      if FLAGS.use_bfloat16:
        logits = tf.cast(logits, tf.float32)
      probs = tf.nn.softmax(logits)
      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        metrics['test/negative_log_likelihood'].update_state(
            negative_log_likelihood)
        metrics['test/accuracy'].update_state(labels, probs)
        metrics['test/ece'].add_batch(probs, label=labels)
      else:
        metrics['test/nll_{}'.format(dataset_name)].update_state(
            negative_log_likelihood)
        metrics['test/accuracy_{}'.format(dataset_name)].update_state(
            labels, probs)
        metrics['test/ece_{}'.format(dataset_name)].add_batch(
            probs, label=labels)

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
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
