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

"""Variational inference for LeNet5 on (Fashion) MNIST."""

import os
from absl import app
from absl import flags
from absl import logging

import edward2 as ed
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
import tensorflow_probability as tfp
import uncertainty_baselines as ub
import utils  # local file import from baselines.mnist

flags.DEFINE_enum('dataset', 'mnist',
                  enum_values=['mnist', 'fashion_mnist'],
                  help='Name of the image dataset.')
flags.DEFINE_integer('training_steps', 30000, 'Training steps.')
flags.DEFINE_integer('batch_size', 256, 'Batch size.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_float('learning_rate_for_sampling', 0.00001, 'Learning rate.')
flags.DEFINE_integer('auxiliary_sampling_frequency', 100,
                     'Steps between sampling auxiliary variables.')
flags.DEFINE_float('auxiliary_variance_ratio', 0.7,
                   'Variance ratio of the auxiliary variables wrt the prior.')
flags.DEFINE_integer('n_auxiliary_variables', 5,
                     'Number of auxiliary variables.')
flags.DEFINE_float('mean_field_init_untransformed_scale', -7,
                   'Initial scale (before softplus) for mean field.')
flags.DEFINE_integer('ensemble_size', 10, 'Number of ensemble components.')
flags.DEFINE_integer('validation_freq', 5, 'Validation frequency in steps.')
flags.DEFINE_string('output_dir', '/tmp/uci',
                    'The directory where the model weights and '
                    'training/evaluation summaries are stored.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_bool('use_gpu', True, 'Whether to run on GPU or otherwise TPU.')
flags.DEFINE_integer('num_cores', 1, 'Number of TPU cores or number of GPUs.')
FLAGS = flags.FLAGS


# TODO(trandustin): Remove need for this boilerplate code.
def mean_field_fn(empirical_bayes=False,
                  initializer=tf1.initializers.he_normal()):
  """Constructors for Gaussian prior and posterior distributions.

  Args:
    empirical_bayes (bool): Whether to train the variance of the prior or not.
    initializer (tf1.initializer): Initializer for the posterior means.
  Returns:
    prior, posterior (tfp.distribution): prior and posterior
    to be fed into a Bayesian Layer.
  """

  def prior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the prior distribution (tfp.distributions.Independent)."""
    softplus_inverse_scale = np.log(np.exp(1.) - 1.)

    istrainable = add_variable_fn(
        name=name + '_istrainable',
        shape=(),
        initializer=tf1.constant_initializer(1.),
        dtype=dtype,
        trainable=False)

    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=(),
        initializer=tf1.constant_initializer(softplus_inverse_scale),
        dtype=dtype,
        trainable=empirical_bayes and trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale * istrainable + (1. - istrainable) *
                       tf1.stop_gradient(untransformed_scale)))
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=tf1.constant_initializer(0.),
        dtype=dtype,
        trainable=False)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.istrainable = istrainable
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  def posterior(dtype, shape, name, trainable, add_variable_fn):
    """Returns the posterior distribution (tfp.distributions.Independent)."""
    untransformed_scale = add_variable_fn(
        name=name + '_untransformed_scale',
        shape=shape,
        initializer=tf1.initializers.random_normal(
            mean=FLAGS.mean_field_init_untransformed_scale, stddev=0.1),
        dtype=dtype,
        trainable=trainable)
    scale = (
        np.finfo(dtype.as_numpy_dtype).eps +
        tf.nn.softplus(untransformed_scale))
    loc = add_variable_fn(
        name=name + '_loc',
        shape=shape,
        initializer=initializer,
        dtype=dtype,
        trainable=trainable)
    dist = tfp.distributions.Normal(loc=loc, scale=scale)
    dist.untransformed_scale = untransformed_scale
    batch_ndims = tf1.size(input=dist.batch_shape_tensor())
    return tfp.distributions.Independent(dist,
                                         reinterpreted_batch_ndims=batch_ndims)

  return prior, posterior


def sample_auxiliary_op(prior, posterior, aux_variance_ratio):
  r"""Sample the auxiliary variable and calculate the conditionals.

  Given a gaussian prior $$\mathcal{N}(\mu_z, \sigma^2_z)$$
  Define auxiliary variables $$z=a_1+a_2$$ with $$a_1=\mathcal{N}(0,
  \sigma_{a_1}^2)$$ and $$a_2=\mathcal{N}(\mu_z, \sigma_{a_2}^2)$$ with
  $$\frac{\sigma_{a_1}^2}{\sigma^2_z}=$$aux_variance_ratio and
  $$\sigma_{a_1}^2+\sigma_{a_2}^2=\sigma_z^2$$.
  From this, we can calculate the posterior of a1 and the conditional of z.

  Conditional:
  $$p(a_1|z) =  \mathcal{N}(z \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_z^2})$$

  Posterior of $$a_1$$:
  $$q(a_1) =\mathcal{N}(\mu_{q(z)} \frac{\sigma_{a_1}^2}{\sigma_{z}^2},
  \frac{\sigma_{q(z)}^2\sigma_{a_1}^4}{\sigma_{z}^4} +
  \frac{\sigma_{a_1}^2\sigma_{a_2}^2}{\sigma_{z}^2})$$

  Conditional posterior:
  $$q(z|a_1)=\frac{q(a_1|z)q(z)}{q(a_1)}$$

  $$q(z|a_1)=\mathcal{N}(\frac{a_1\sigma^2_{q(z)}\sigma^2_{z} +
  \mu_{q(z)}\sigma^2_{a_2}\sigma^2_{z}}{\sigma^2_{q(z)}\sigma^2_{a_1} +
  \sigma^2_z\sigma^2_{a_2}},
  \frac{\sigma^2_{q(z)}\sigma^2_z\sigma^2_{a_2}}{\sigma^2_{a_1}\sigma^2_{q(z)} +
  \sigma^2_{z}\sigma^2_{a_2}})$$.

  Args:
    prior: The prior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    posterior: The posterior distribution. Must be parameterized by loc and
      untransformed_scale, with the transformation being the softplus function.
    aux_variance_ratio: Ratio of the variance of the auxiliary variable and the
      prior. The mean of the auxiliary variable is at 0.

  Returns:
    sampling_op: Tensorflow operation that executes the sampling.
    log_density_ratio: Tensor containing the density ratio of the auxiliary
    variable.
  """
  if aux_variance_ratio > 1. or aux_variance_ratio < 0.:
    raise ValueError(
        'The ratio of the variance of the auxiliary variable must be between 0 '
        'and 1.'
    )

  p_a1_loc = tf.zeros_like(prior.loc)
  p_a1_scale = tf.math.sqrt(prior.scale**2 * aux_variance_ratio)
  p_a1 = tfp.distributions.Normal(loc=p_a1_loc, scale=p_a1_scale)
  p_a2_loc = prior.loc
  p_a2_scale = tf.math.sqrt(prior.scale**2 - p_a1_scale**2)
  # q(a1)
  a1_loc = (posterior.loc - prior.loc) * p_a1_scale**2 / prior.scale**2
  a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a1_scale**2 / prior.scale**2 + p_a2_scale**2) *
      p_a1_scale**2 / prior.scale**2)
  q_a1 = tfp.distributions.Normal(loc=a1_loc, scale=a1_scale)
  a1 = q_a1.sample()

  # q(z|a1)
  z_a1_loc = prior.loc + (
      (posterior.loc - prior.loc) * p_a2_scale**2 * prior.scale**2 +
      a1 * posterior.scale**2 * prior.scale**2) / (
          prior.scale**2 * p_a2_scale**2 + posterior.scale**2 * p_a1_scale**2)
  z_a1_scale = tf.math.sqrt(
      (posterior.scale**2 * p_a2_scale**2 * prior.scale**2) /
      (prior.scale**2 * p_a2_scale**2 + p_a1_scale**2 * posterior.scale**2))

  with tf1.control_dependencies([
      q_a1.loc, q_a1.scale, p_a1.loc, p_a1.scale, a1, p_a2_loc, p_a2_scale,
      z_a1_loc, z_a1_scale
  ]):
    log_density_ratio = q_a1.log_prob(a1) - p_a1.log_prob(a1)
    prior_update = [
        prior.loc.assign(a1 + p_a2_loc),
        prior.untransformed_scale.assign(tfp.math.softplus_inverse(p_a2_scale))
    ]
    posterior_update = [
        posterior.loc.assign(z_a1_loc),
        posterior.untransformed_scale.assign(
            tfp.math.softplus_inverse(z_a1_scale))
    ]
  return [prior_update, posterior_update], tf.reduce_sum(log_density_ratio)


def lenet5(n_examples, input_shape, num_classes):
  """Builds Bayesian LeNet5."""
  p_fn, q_fn = mean_field_fn(empirical_bayes=True)
  def normalized_kl_fn(q, p, _):
    return q.kl_divergence(p) / tf.cast(n_examples, tf.float32)

  inputs = tf.keras.layers.Input(shape=input_shape)
  conv1 = tfp.layers.Convolution2DFlipout(
      6,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(inputs)
  pool1 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv1)
  conv2 = tfp.layers.Convolution2DFlipout(
      16,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(pool1)
  pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2],
                                       strides=[2, 2],
                                       padding='SAME')(conv2)
  conv3 = tfp.layers.Convolution2DFlipout(
      120,
      kernel_size=5,
      padding='SAME',
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(pool2)
  flatten = tf.keras.layers.Flatten()(conv3)
  dense1 = tfp.layers.DenseLocalReparameterization(
      84,
      activation=tf.nn.relu,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(flatten)
  dense2 = tfp.layers.DenseLocalReparameterization(
      num_classes,
      kernel_prior_fn=p_fn,
      kernel_posterior_fn=q_fn,
      bias_prior_fn=p_fn,
      bias_posterior_fn=q_fn,
      kernel_divergence_fn=normalized_kl_fn,
      bias_divergence_fn=normalized_kl_fn)(dense1)
  outputs = tf.keras.layers.Lambda(lambda x: ed.Categorical(logits=x))(dense2)
  return tf.keras.models.Model(inputs=inputs, outputs=outputs)


def get_losses_and_metrics(model, n_train):
  """Define the losses and metrics for the model."""

  def negative_log_likelihood(y, rv_y):
    del rv_y  # unused arg
    return -model.output.distribution.log_prob(tf.squeeze(y))

  def accuracy(y_true, y_sample):
    del y_sample  # unused arg
    return tf.equal(
        tf.argmax(input=model.output.distribution.logits, axis=1),
        tf.cast(tf.squeeze(y_true), tf.int64))

  def log_likelihood(y_true, y_sample):
    """Expected conditional log-likelihood."""
    del y_sample  # unused arg
    return model.output.distribution.log_prob(tf.squeeze(y_true))

  def kl(y_true, y_sample):
    """KL-divergence."""
    del y_true  # unused arg
    del y_sample  # unused arg
    sampling_cost = sum(
        [l.kl_cost_weight + l.kl_cost_bias for l in model.layers])
    return sum(model.losses) * n_train + sampling_cost

  def elbo(y_true, y_sample):
    return log_likelihood(y_true, y_sample) * n_train - kl(y_true, y_sample)

  return negative_log_likelihood, accuracy, log_likelihood, kl, elbo


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

  session = tf1.Session()
  with session.as_default():
    n_train = x_train.shape[0]

    num_classes = int(np.amax(y_train)) + 1
    model = lenet5(n_train, x_train.shape[1:], num_classes)
    for l in model.layers:
      l.kl_cost_weight = l.add_weight(
          name='kl_cost_weight',
          shape=(),
          initializer=tf.constant_initializer(0.),
          trainable=False)
      l.kl_cost_bias = l.add_variable(
          name='kl_cost_bias',
          shape=(),
          initializer=tf.constant_initializer(0.),
          trainable=False)

    [negative_log_likelihood,
     accuracy,
     log_likelihood,
     kl,
     elbo] = get_losses_and_metrics(model, n_train)
    metrics = [elbo, log_likelihood, kl, accuracy]
    tensorboard = tf1.keras.callbacks.TensorBoard(
        log_dir=FLAGS.output_dir,
        update_freq=FLAGS.batch_size * FLAGS.validation_freq)

    def fit_fn(model,
               steps,
               initial_epoch):
      return model.fit(
          x=x_train,
          y=y_train,
          batch_size=FLAGS.batch_size,
          epochs=initial_epoch + (FLAGS.batch_size * steps) // n_train,
          initial_epoch=initial_epoch,
          validation_data=(x_test, y_test),
          validation_freq=max(
              (FLAGS.validation_freq * FLAGS.batch_size) // n_train, 1),
          verbose=1,
          callbacks=[tensorboard])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=float(FLAGS.learning_rate)),
        loss=negative_log_likelihood,
        metrics=metrics)
    session.run(tf1.initialize_all_variables())

    train_epochs = (FLAGS.training_steps * FLAGS.batch_size) // n_train
    fit_fn(model, FLAGS.training_steps, initial_epoch=0)

    labels = tf.keras.layers.Input(shape=y_train.shape[1:])
    ll = tf.keras.backend.function([model.input, labels], [
        model.output.distribution.log_prob(tf.squeeze(labels)),
        model.output.distribution.logits
    ])

    base_metrics = [
        utils.ensemble_metrics(x_train, y_train, model, ll, n_samples=10),
        utils.ensemble_metrics(x_test, y_test, model, ll, n_samples=10)
    ]
    model_dir = os.path.join(FLAGS.output_dir, 'models')
    tf.io.gfile.makedirs(model_dir)
    base_model_filename = os.path.join(model_dir, 'base_model.weights')
    model.save_weights(base_model_filename)

    # Train base model further for comparison.
    fit_fn(
        model,
        FLAGS.n_auxiliary_variables * FLAGS.auxiliary_sampling_frequency *
        FLAGS.ensemble_size,
        initial_epoch=train_epochs)

    overtrained_metrics = [
        utils.ensemble_metrics(x_train, y_train, model, ll, n_samples=10),
        utils.ensemble_metrics(x_test, y_test, model, ll, n_samples=10)
    ]

    # Perform refined VI.
    sample_op = []
    for l in model.layers:
      if isinstance(l, tfp.layers.DenseLocalReparameterization) or isinstance(
          l, tfp.layers.Convolution2DFlipout):
        weight_op, weight_cost = sample_auxiliary_op(
            l.kernel_prior.distribution, l.kernel_posterior.distribution,
            FLAGS.auxiliary_variance_ratio)
        sample_op.append(weight_op)
        sample_op.append(l.kl_cost_weight.assign_add(weight_cost))
        # Fix the variance of the prior
        session.run(l.kernel_prior.distribution.istrainable.assign(0.))
        if hasattr(l.bias_prior, 'distribution'):
          bias_op, bias_cost = sample_auxiliary_op(
              l.bias_prior.distribution, l.bias_posterior.distribution,
              FLAGS.auxiliary_variance_ratio)
          sample_op.append(bias_op)
          sample_op.append(l.kl_cost_bias.assign_add(bias_cost))
          # Fix the variance of the prior
          session.run(l.bias_prior.distribution.istrainable.assign(0.))

    ensemble_filenames = []
    for i in range(FLAGS.ensemble_size):
      model.load_weights(base_model_filename)
      for j in range(FLAGS.n_auxiliary_variables):
        session.run(sample_op)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                # The learning rate is proportional to the scale of the prior.
                lr=float(FLAGS.learning_rate_for_sampling *
                         np.sqrt(1. - FLAGS.auxiliary_variance_ratio)**j)),
            loss=negative_log_likelihood,
            metrics=metrics)
        fit_fn(
            model,
            FLAGS.auxiliary_sampling_frequency,
            initial_epoch=train_epochs)
      ensemble_filename = os.path.join(
          model_dir, 'ensemble_component_' + str(i) + '.weights')
      ensemble_filenames.append(ensemble_filename)
      model.save_weights(ensemble_filename)

    auxiliary_metrics = [
        utils.ensemble_metrics(
            x_train,
            y_train,
            model,
            ll,
            weight_files=ensemble_filenames, n_samples=10),
        utils.ensemble_metrics(
            x_test,
            y_test,
            model,
            ll,
            weight_files=ensemble_filenames, n_samples=10)
    ]

    for metrics, name in [(base_metrics, 'Base model'),
                          (overtrained_metrics, 'Overtrained model'),
                          (auxiliary_metrics, 'Auxiliary sampling')]:
      logging.info(name)
      for metrics_dict, split in [(metrics[0], 'train'),
                                  (metrics[1], 'test')]:
        logging.info(split)
        for metric_name in metrics_dict:
          logging.info('%s: %s', metric_name, metrics_dict[metric_name])


if __name__ == '__main__':
  app.run(main)
