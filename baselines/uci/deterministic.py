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

"""MLP on UCI data trained with maximum likelihood and gradient descent."""

import os
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import numpy as np
import tensorflow as tf
import utils  # local file import from baselines.uci

flags.DEFINE_enum('dataset', 'boston_housing',
                  enum_values=['boston_housing',
                               'concrete_strength',
                               'energy_efficiency',
                               'naval_propulsion',
                               'kin8nm',
                               'power_plant',
                               'protein_structure',
                               'wine',
                               'yacht_hydrodynamics'],
                  help='Name of the UCI dataset.')
flags.DEFINE_integer('ensemble_size', 1, 'Number of ensemble members.')
flags.DEFINE_boolean('bootstrap', False,
                     'Sample the training set for bootstrapping.')
flags.DEFINE_integer('training_steps', 2500, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('epsilon', 0.,
                   'Epsilon for adversarial training. It is given as a ratio '
                   'of the input range (e.g the adjustment is 2.55 if input '
                   'range is [0,255]). Set to 0. for no adversarial training.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/uci',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0,
                     'Random seed. Note train/test splits are random and also '
                     'based on this seed.')
flags.DEFINE_bool('use_gpu', False, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 8, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


# TODO(trandustin): Change to act like InputLayer or just swap to a Normal
# custom layer with a scale tf.Variable.
class VariableInputLayer(tf.keras.layers.Layer):
  """Layer as an entry point into a Model, and which is learnable."""

  def __init__(self,
               input_shape,
               initializer='zeros',
               regularizer=None,
               constraint=None,
               **kwargs):
    self.shape = input_shape
    self.initializer = ed.initializers.get(initializer)
    self.regularizer = ed.regularizers.get(regularizer)
    self.constraint = ed.constraints.get(constraint)
    super().__init__(**kwargs)

  def build(self, input_shape):
    del input_shape  # unused arg
    self.variable = self.add_weight(shape=self.shape,
                                    name='variable',
                                    initializer=self.initializer,
                                    regularizer=self.regularizer,
                                    constraint=None)
    self.built = True

  def call(self, inputs):
    del inputs  # unused arg
    variable = tf.convert_to_tensor(self.variable)
    if self.constraint is not None:
      variable = self.constraint(variable)
    return variable

  def get_config(self):
    config = {
        'input_shape': self.shape,
        'initializer': ed.initializers.serialize(self.initializer),
        'regularizer': ed.regularizers.serialize(self.regularizer),
        'constraint': ed.constraints.serialize(self.constraint),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def multilayer_perceptron(input_shape, output_scaler=1.):
  """Builds a single hidden layer feedforward network.

  Args:
    input_shape: tf.TensorShape.
    output_scaler: Float to scale mean predictions. Training is faster and more
      stable when both the inputs and outputs are normalized. To not affect
      metrics such as RMSE and NLL, the outputs need to be scaled back
      (de-normalized, but the mean doesn't matter), using output_scaler.

  Returns:
    tf.keras.Model.
  """
  inputs = tf.keras.layers.Input(shape=input_shape)
  hidden = tf.keras.layers.Dense(50, activation='relu')(inputs)
  loc = tf.keras.layers.Dense(1, activation=None)(hidden)
  loc = tf.keras.layers.Lambda(lambda x: x * output_scaler)(loc)
  # The variable layer must depend on a symbolic input tensor.
  scale = VariableInputLayer((), constraint='softplus')(inputs)
  outputs = tf.keras.layers.Lambda(lambda x: ed.Normal(loc=x[0], scale=x[1]))(
      (loc, scale))
  return tf.keras.Model(inputs=inputs, outputs=outputs)


def make_adversarial_loss_fn(model):
  """Returns loss function with adversarial training."""
  def loss_fn(x, y):
    """Loss function with adversarial training."""
    with tf.GradientTape() as tape:
      tape.watch(x)
      predictions = model(x)
      loss = -tf.reduce_mean(predictions.distribution.log_prob(y))
    dx = tape.gradient(loss, x)
    # Assume the training data is normalized.
    adv_inputs = x + FLAGS.epsilon * tf.math.sign(tf.stop_gradient(dx))
    adv_predictions = model(adv_inputs)
    adv_loss = -tf.reduce_mean(adv_predictions.distribution.log_prob(y))
    return 0.5 * loss + 0.5 * adv_loss
  return loss_fn


def main(argv):
  del argv  # unused arg
  if not FLAGS.use_gpu:
    raise ValueError('Only GPU is currently supported.')
  if FLAGS.num_cores > 1:
    raise ValueError('Only a single accelerator is currently supported.')
  np.random.seed(FLAGS.seed)
  tf.random.set_seed(FLAGS.seed)
  tf.io.gfile.makedirs(FLAGS.output_dir)

  x_train, y_train, x_test, y_test = utils.load(FLAGS.dataset)
  n_train = x_train.shape[0]
  ensemble_filenames = []
  for i in range(FLAGS.ensemble_size):
    model = multilayer_perceptron(
        x_train.shape[1:], np.std(y_train, axis=0) + tf.keras.backend.epsilon())
    if FLAGS.epsilon:
      loss_fn = make_adversarial_loss_fn(model)
      optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    else:
      def negative_log_likelihood(y_true, y_pred):
        return -y_pred.distribution.log_prob(y_true)
      model.compile(optimizer=tf.keras.optimizers.Adam(lr=FLAGS.learning_rate),
                    loss=negative_log_likelihood)

    member_dir = os.path.join(FLAGS.output_dir, 'member_' + str(i))
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=member_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)
    if FLAGS.epsilon:
      for epoch in range((FLAGS.batch_size * FLAGS.training_steps) // n_train):
        logging.info('Epoch %s', epoch)
        for j in range(n_train // FLAGS.batch_size):
          perm = np.random.permutation(n_train)
          with tf.GradientTape() as tape:
            loss = loss_fn(x_train[perm[j:j + FLAGS.batch_size]],
                           y_train[perm[j:j + FLAGS.batch_size]])
          grads = tape.gradient(loss, model.trainable_weights)
          optimizer.apply_gradients(zip(grads, model.trainable_weights))
    else:
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
          verbose=0,
          callbacks=[tensorboard])

    member_filename = os.path.join(member_dir, 'model.weights')
    ensemble_filenames.append(member_filename)
    model.save_weights(member_filename)

  # TODO(trandustin): Move this into utils.ensemble_metrics. It's currently
  # separate so that VI can use utils.ensemble_metrics while in TF1.
  def ll(arg):
    features, labels = arg
    predictions = model(features)
    log_prob = predictions.distribution.log_prob(labels)
    error = predictions.distribution.loc - labels
    return [log_prob, error]

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
