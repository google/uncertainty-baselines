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

"""LeNet-5 on (Fashion) MNIST."""

import os
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import uncertainty_baselines as ub
import utils  # local file import from baselines.mnist

flags.DEFINE_enum('dataset', 'mnist',
                  enum_values=['mnist', 'fashion_mnist'],
                  help='Name of the image dataset.')
flags.DEFINE_integer('ensemble_size', 1, 'Number of ensemble members.')
flags.DEFINE_boolean('bootstrap', False,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 5000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/det_training',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


def lenet5(input_shape, num_classes):
  """Builds LeNet5."""
  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = tf.keras.layers.Conv2D(6,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation='relu')(inputs)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv1)
  conv2 = tf.keras.layers.Conv2D(16,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation='relu')(pool1)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
  conv3 = tf.keras.layers.Conv2D(120,
                                 kernel_size=5,
                                 padding='SAME',
                                 activation=tf.nn.relu)(pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = tf.keras.layers.Dense(84, activation=tf.nn.relu)(flatten)
  logits = tf.keras.layers.Dense(num_classes)(dense1)
  outputs = tf.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(logits)
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  if FLAGS.dataset == 'mnist':
    dataset_builder_class = ub.datasets.MnistDataset
  else:
    dataset_builder_class = ub.datasets.FashionMnistDataset
  n_train = 50000
  train_dataset = next(dataset_builder_class(
      'train').load(batch_size=n_train).as_numpy_iterator())
  x_train = train_dataset['features']
  y_train = train_dataset['labels']
  test_dataset = next(dataset_builder_class(
      'test').load(batch_size=10000).as_numpy_iterator())
  x_test = test_dataset['features']
  y_test = test_dataset['labels']
  num_classes = int(np.amax(y_train)) + 1

  # Note that we need to disable v2 behavior after we load the data.
  tf1.disable_v2_behavior()

  ensemble_filenames = []
  for i in range(FLAGS.ensemble_size):
    # TODO(trandustin): We re-build the graph for each ensemble member. This
    # is due to an unknown bug where the variables are otherwise not
    # re-initialized to be random. While this is inefficient in graph mode, I'm
    # keeping this for now as we'd like to move to eager mode anyways.
    model = lenet5(x_train.shape[1:], num_classes)

    def negative_log_likelihood(y, rv_y):
      del rv_y  # unused arg
      return -model.output.distribution.log_prob(tf.squeeze(y))  # pylint: disable=cell-var-from-loop

    def accuracy(y_true, y_sample):
      del y_sample  # unused arg
      return tf.equal(
          tf.argmax(input=model.output.distribution.logits, axis=1),  # pylint: disable=cell-var-from-loop
          tf.cast(tf.squeeze(y_true), tf.int64))

    def log_likelihood(y_true, y_sample):
      del y_sample  # unused arg
      return model.output.distribution.log_prob(tf.squeeze(y_true))  # pylint: disable=cell-var-from-loop

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
        loss=negative_log_likelihood,
        metrics=[log_likelihood, accuracy])
    member_dir = os.path.join(FLAGS.output_dir, 'member_' + str(i))
    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=member_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)

    if FLAGS.bootstrap:
      inds = np.random.choice(n_train, n_train, replace=True)
      x_sampled = x_train[inds]
      y_sampled = y_train[inds]

    model.fit(
        x=x_train if not FLAGS.bootstrap else x_sampled,
        y=y_train if not FLAGS.bootstrap else y_sampled,
        batch_size=FLAGS.batch_size,
        epochs=(FLAGS.batch_size * FLAGS.training_steps) // n_train,
        validation_data=(x_test, y_test),
        validation_freq=max(
            (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
        verbose=1,
        callbacks=[tensorboard])

    member_filename = os.path.join(member_dir, 'model.weights')
    ensemble_filenames.append(member_filename)
    model.save_weights(member_filename)

  labels = tf.keras.layers.Input(shape=y_train.shape[1:])
  ll = tf.keras.backend.function([model.input, labels], [
      model.output.distribution.log_prob(tf.squeeze(labels)),
      model.output.distribution.logits,
  ])

  ensemble_metrics_vals = {
      'train': utils.ensemble_metrics(
          x_train, y_train, model, ll, weight_files=ensemble_filenames),
      'test': utils.ensemble_metrics(
          x_test, y_test, model, ll, weight_files=ensemble_filenames),
  }

  for split, metrics in ensemble_metrics_vals.items():
    logging.info(split)
    for metric_name in metrics:
      logging.info('%s: %s', metric_name, metrics[metric_name])

if __name__ == '__main__':
  app.run(main)
