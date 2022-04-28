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

"""Tests for the Rank-1 BNN ViT model script."""
import os
import pathlib
import tempfile

from absl import flags
from absl import logging
from absl.testing import flagsaver
from absl.testing import parameterized
import flax
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_probability.substrates.jax as tfp
import batchensemble_utils  # local file import from baselines.jft
import checkpoint_utils  # local file import from baselines.jft
import rank1_bnn  # local file import from baselines.jft
import test_utils  # local file import from baselines.jft

flags.adopt_module_key_flags(rank1_bnn)
FLAGS = flags.FLAGS


def get_config(dataset_name, classifier, representation_size):
  """Config."""
  config = test_utils.get_config(
      dataset_name=dataset_name,
      classifier=classifier,
      representation_size=representation_size,
      batch_size=2,
      total_steps=2)

  config.model.patches.size = [4, 4]

  # BatchEnsemble parameters
  config.model.transformer.num_layers = 1
  config.model.transformer.ens_size = 2
  config.model.transformer.random_sign_init = 0.5
  config.model.transformer.be_layers = (0,)
  config.fast_weight_lr_multiplier = 1.0
  config.weight_decay = 0.1

  # Rank-1 BNN parameters
  config.prior_mean = 1.
  config.prior_std = 0.05
  config.eval_samples = 3

  return config


class Rank1BNNTest(parameterized.TestCase, tf.test.TestCase):

  def setUp(self):
    super().setUp()
    baseline_root_dir = pathlib.Path(__file__).parents[1]
    data_dir = os.path.join(baseline_root_dir, 'testing_data')
    logging.info('data_dir contents: %s', os.listdir(data_dir))
    self.data_dir = data_dir

  def test_pointwise_to_loc_scale(self):
    key = jax.random.PRNGKey(42)
    key, x_key, params_key = jax.random.split(key, num=3)
    x = jax.random.normal(x_key, shape=(5,))
    params = rank1_bnn._pointwise_to_loc_scale(x, params_key)
    self.assertIsInstance(params, dict)
    self.assertEqual(list(params.keys()), ['loc', 'unconstrained_scale'])
    self.assertEqual(params['loc'].shape, x.shape)
    self.assertEqual(params['unconstrained_scale'].shape, x.shape)

  def test_sample_mean_field_gaussian(self):
    key = jax.random.PRNGKey(42)
    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    params = {'loc': loc, 'unconstrained_scale': unconstrained_scale}

    key, sample_key = jax.random.split(key)
    sample = rank1_bnn._sample_mean_field_gaussian(params, sample_key)
    self.assertEqual(sample.shape, loc.shape)
    self.assertTrue(jnp.all(jnp.isfinite(sample)))

  def test_sample_compute_gaussian_kl_divergence(self):
    key = jax.random.PRNGKey(42)
    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    params = {'loc': loc, 'unconstrained_scale': unconstrained_scale}

    prior_mean = 1.
    prior_std = 0.05
    kl = rank1_bnn._compute_gaussian_kl_divergence(
        params, prior_mean=prior_mean, prior_std=prior_std)
    self.assertEqual(kl.shape, ())

    scale = jax.nn.softplus(unconstrained_scale) + 1e-8
    posterior_dist = tfp.distributions.Independent(
        tfp.distributions.Normal(loc, scale),
        reinterpreted_batch_ndims=len(loc.shape))
    prior_dist = tfp.distributions.Independent(
        tfp.distributions.Normal(
            jnp.ones_like(loc) * prior_mean,
            jnp.ones_like(loc) * prior_std),
        reinterpreted_batch_ndims=len(loc.shape))
    correct_kl = tfp.distributions.kl_divergence(posterior_dist, prior_dist)
    self.assertAlmostEqual(kl, correct_kl)

  def test_create_key_tree(self):
    key = jax.random.PRNGKey(42)
    key, x_key = jax.random.split(key)
    x = jax.random.normal(x_key, shape=(5,))

    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    fast_weight_params = {
        'loc': loc,
        'unconstrained_scale': unconstrained_scale
    }

    rank1_weights_prefix = 'fast_weight'
    params = {
        'this': {
            'that': x,
            f'{rank1_weights_prefix}_alpha': fast_weight_params
        }
    }

    key, key_tree_key = jax.random.split(key)
    key_tree = rank1_bnn._create_key_tree(params, key_tree_key)
    self.assertEqual(
        flax.traverse_util.flatten_dict(key_tree).keys(),
        flax.traverse_util.flatten_dict(params).keys())
    self.assertIsInstance(key_tree['this'], dict)
    self.assertEqual(
        list(key_tree['this'].keys()),
        ['that', f'{rank1_weights_prefix}_alpha'])
    self.assertIsInstance(key_tree['this']['that'], type(key))
    self.assertIsInstance(key_tree['this'][f'{rank1_weights_prefix}_alpha'],
                          dict)
    self.assertEqual(key_tree['this'][f'{rank1_weights_prefix}_alpha'].keys(),
                     fast_weight_params.keys())
    self.assertIsInstance(
        key_tree['this'][f'{rank1_weights_prefix}_alpha']['loc'], type(key))
    self.assertIsInstance(
        key_tree['this'][f'{rank1_weights_prefix}_alpha']
        ['unconstrained_scale'], type(key))

    is_leaf = lambda prefix, _: rank1_weights_prefix in '/'.join(prefix)
    key_tree = rank1_bnn._create_key_tree(params, key_tree_key, is_leaf=is_leaf)
    self.assertEqual(key_tree.keys(), params.keys())
    self.assertIsInstance(key_tree['this'][f'{rank1_weights_prefix}_alpha'],
                          type(key))

  def test_get_rank1_params(self):
    key = jax.random.PRNGKey(42)
    key, x_key = jax.random.split(key)
    x = jax.random.normal(x_key, shape=(5,))

    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    fast_weight_params = {
        'loc': loc,
        'unconstrained_scale': unconstrained_scale
    }
    loc2 = loc * 2
    unconstrained_scale2 = unconstrained_scale * 2
    fast_weight_params2 = {
        'loc': loc2,
        'unconstrained_scale': unconstrained_scale2
    }

    rank1_weights_prefix = 'fast_weight'
    params = {
        'this': {
            'that': x,
            f'{rank1_weights_prefix}_alpha': fast_weight_params
        },
        f'{rank1_weights_prefix}_alpha': fast_weight_params2
    }

    rank1_params = rank1_bnn._get_rank1_params(params,
                                               [f'.*{rank1_weights_prefix}.*'])
    self.assertIsInstance(rank1_params, list)
    self.assertLen(rank1_params, 2)
    self.assertEqual(
        list(rank1_params[0].keys()), ['loc', 'unconstrained_scale'])
    self.assertEqual(
        list(rank1_params[1].keys()), ['loc', 'unconstrained_scale'])
    self.assertTrue(jnp.allclose(rank1_params[0]['loc'], loc))
    self.assertTrue(
        jnp.allclose(rank1_params[0]['unconstrained_scale'],
                     unconstrained_scale))
    self.assertTrue(jnp.allclose(rank1_params[1]['loc'], loc2))
    self.assertTrue(
        jnp.allclose(rank1_params[1]['unconstrained_scale'],
                     unconstrained_scale2))

  def test_init_gaussian_rank1(self):
    key = jax.random.PRNGKey(42)
    key, x_key, y_key = jax.random.split(key, num=3)
    x = jax.random.normal(x_key, shape=(5,))
    y = jax.random.normal(y_key, shape=(5,))
    rank1_weights_prefix = 'fast_weight'
    params = {'this': {'that': x, f'{rank1_weights_prefix}_alpha': y}}

    key, params_key = jax.random.split(key)
    rank1_params = rank1_bnn.init_gaussian_rank1(
        params,
        params_key,
        rank1_regex_patterns=[f'.*{rank1_weights_prefix}.*'])
    self.assertEqual(list(rank1_params.keys()), ['this'])
    self.assertEqual(
        list(rank1_params['this'].keys()),
        ['that', f'{rank1_weights_prefix}_alpha'])
    self.assertIsInstance(rank1_params['this'][f'{rank1_weights_prefix}_alpha'],
                          dict)
    self.assertEqual(
        list(rank1_params['this'][f'{rank1_weights_prefix}_alpha'].keys()),
        ['loc', 'unconstrained_scale'])

  def test_sample_gaussian_rank1(self):
    key = jax.random.PRNGKey(42)
    key, x_key = jax.random.split(key)
    x = jax.random.normal(x_key, shape=(5,))

    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    fast_weight_params = {
        'loc': loc,
        'unconstrained_scale': unconstrained_scale
    }

    rank1_weights_prefix = 'fast_weight'
    params = {
        'this': {
            'that': x,
            f'{rank1_weights_prefix}_alpha': fast_weight_params
        }
    }

    key, sample_key = jax.random.split(key)
    sampled_params = rank1_bnn.sample_gaussian_rank1(
        params,
        sample_key,
        rank1_regex_patterns=[f'.*{rank1_weights_prefix}.*'])
    self.assertEqual(list(sampled_params.keys()), ['this'])
    self.assertEqual(
        list(sampled_params['this'].keys()),
        ['that', f'{rank1_weights_prefix}_alpha'])
    self.assertIsInstance(
        sampled_params['this'][f'{rank1_weights_prefix}_alpha'], type(loc))
    self.assertEqual(
        sampled_params['this'][f'{rank1_weights_prefix}_alpha'].shape,
        loc.shape)

  def test_gaussian_rank1_kl_divergence(self):
    key = jax.random.PRNGKey(42)
    key, x_key = jax.random.split(key)
    x = jax.random.normal(x_key, shape=(5,))

    key, loc_key, scale_key = jax.random.split(key, num=3)
    loc = jax.random.normal(loc_key, shape=(5,))
    unconstrained_scale = -3. + 0.1 * jax.random.truncated_normal(
        scale_key, lower=-2, upper=2, shape=loc.shape)
    fast_weight_params = {
        'loc': loc,
        'unconstrained_scale': unconstrained_scale
    }

    rank1_weights_prefix = 'fast_weight'
    params = {
        'this': {
            'that': x,
            f'{rank1_weights_prefix}_alpha': fast_weight_params
        }
    }

    prior_mean = 1.
    prior_std = 0.05
    kl = rank1_bnn.gaussian_rank1_kl_divergence(
        params,
        prior_mean=prior_mean,
        prior_std=prior_std,
        rank1_regex_patterns=[f'.*{rank1_weights_prefix}.*'])
    self.assertEqual(kl.shape, ())

    scale = jax.nn.softplus(unconstrained_scale) + 1e-8
    posterior_dist = tfp.distributions.Independent(
        tfp.distributions.Normal(loc, scale),
        reinterpreted_batch_ndims=len(loc.shape))
    prior_dist = tfp.distributions.Independent(
        tfp.distributions.Normal(
            jnp.ones_like(loc) * prior_mean,
            jnp.ones_like(loc) * prior_std),
        reinterpreted_batch_ndims=len(loc.shape))
    correct_kl = tfp.distributions.kl_divergence(posterior_dist, prior_dist)
    self.assertAlmostEqual(kl, correct_kl)

    params = {
        'this': {
            'that': x,
            f'{rank1_weights_prefix}_alpha': fast_weight_params
        },
        f'{rank1_weights_prefix}_alpha': fast_weight_params
    }
    kl = rank1_bnn.gaussian_rank1_kl_divergence(
        params,
        prior_mean=prior_mean,
        prior_std=prior_std,
        rank1_regex_patterns=[f'.*{rank1_weights_prefix}.*'])

    self.assertEqual(kl.shape, ())
    self.assertAlmostEqual(kl, 2 * correct_kl, msg=kl)

  @parameterized.parameters(1, 3, 5)
  def test_log_average_probs(self, ensemble_size):
    batch_size, num_classes = 16, 3
    logits_shape = (ensemble_size, batch_size, num_classes)
    np.random.seed(42)
    ensemble_logits = jnp.asarray(np.random.normal(size=logits_shape))

    actual_logits = batchensemble_utils.log_average_softmax_probs(
        ensemble_logits)
    self.assertEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.softmax(ensemble_logits), axis=0)
    np.testing.assert_allclose(
        jax.nn.softmax(actual_logits), expected_probs, rtol=1e-06, atol=1e-06)

    actual_logits = batchensemble_utils.log_average_sigmoid_probs(
        ensemble_logits)
    self.assertEqual(actual_logits.shape, (batch_size, num_classes))

    expected_probs = jnp.mean(jax.nn.sigmoid(ensemble_logits), axis=0)
    np.testing.assert_allclose(
        jax.nn.sigmoid(actual_logits), expected_probs, rtol=1e-06, atol=1e-06)

  @parameterized.parameters(
      ('imagenet2012', 'token', 2, 841510.44, 620.3136291503906, False),
      ('imagenet2012', 'token', 2, 841510.44, 620.3136291503906, True),
      ('imagenet2012', 'token', None, 839909.8, 550.0570678710938, False),
      ('imagenet2012', 'gap', 2, 841508.9, 626.0960388183594, False),
      ('imagenet2012', 'gap', None, 839909.25, 609.67138671875, False),
  )
  @flagsaver.flagsaver
  def test_rank1_bnn_script(self, dataset_name, classifier, representation_size,
                            correct_train_loss, correct_val_loss,
                            simulate_failure):
    data_dir = self.data_dir
    config = get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = data_dir
    num_examples = config.batch_size * config.total_steps

    if not simulate_failure:
      # Check for any errors.
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        # TODO(dusenberrymw): Test the fewshot results once deterministic.
        train_loss, val_loss, _ = rank1_bnn.main(config, output_dir)
    else:
      # Check for the ability to restart from a previous checkpoint (after
      # failure, etc.).
      # NOTE: Use this flag to simulate failing at a certain step.
      config.testing_failure_step = config.total_steps - 1
      config.checkpoint_steps = config.testing_failure_step
      config.keep_checkpoint_steps = config.checkpoint_steps
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        rank1_bnn.main(config, output_dir)

      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))
      checkpoint = checkpoint_utils.load_checkpoint(None, checkpoint_path)
      self.assertEqual(
          int(checkpoint['opt']['state']['step']), config.testing_failure_step)

      # This should resume from the failed step.
      del config.testing_failure_step
      with tfds.testing.mock_data(num_examples=num_examples, data_dir=data_dir):
        train_loss, val_loss, _ = rank1_bnn.main(config, output_dir)

    # Check for reproducibility.
    logging.info('(train_loss, val_loss) = %s, %s', train_loss, val_loss['val'])
    np.testing.assert_allclose(
        train_loss, correct_train_loss, rtol=1e-06, atol=1e-06)
    np.testing.assert_allclose(
        val_loss['val'], correct_val_loss, rtol=1e-06, atol=1e-06)

    checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
    self.assertTrue(os.path.exists(checkpoint_path))

    # Ensure that the model contains rank-1 distributions.
    params = checkpoint_utils.load_checkpoint(None, checkpoint_path)
    flat_keys = flax.traverse_util.flatten_dict(params, sep='/').keys()
    self.assertTrue(any('fast_weight_alpha/loc' in k for k in flat_keys))
    self.assertTrue(
        any('fast_weight_alpha/unconstrained_scale' in k for k in flat_keys))

  @parameterized.parameters(
      ('imagenet2012', 'token', 2),
      ('imagenet2012', 'token', None),
      ('imagenet2012', 'gap', 2),
      ('imagenet2012', 'gap', None),
  )
  @flagsaver.flagsaver
  def test_load_model(self, dataset_name, classifier, representation_size):
    config = get_config(
        dataset_name=dataset_name,
        classifier=classifier,
        representation_size=representation_size)
    output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
    config.dataset_dir = self.data_dir
    config.total_steps = 2
    num_examples = config.batch_size * config.total_steps

    with tfds.testing.mock_data(
        num_examples=num_examples, data_dir=self.data_dir):
      _, val_loss, _ = rank1_bnn.main(config, output_dir)
      checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')
      self.assertTrue(os.path.exists(checkpoint_path))

      # Set different output directory so that the logic doesn't think we are
      # resuming from a previous checkpoint.
      output_dir = tempfile.mkdtemp(dir=self.get_temp_dir())
      config.model_init = checkpoint_path
      # Reload model from checkpoint.
      # Currently, we don't have a standalone evaluation function, so we check
      # that the loaded model has the same performance as the saved model by
      # running training with a learning rate of 0 to obtain the train and eval
      # metrics.
      # TODO(zmariet, dusenberrymw): write standalone eval function.
      config.lr.base = 0.0
      config.lr.linear_end = 0.0
      config.lr.warmup_steps = 0
      config.model_reinit_params = []

      _, loaded_val_loss, _ = rank1_bnn.main(config, output_dir)

    # We can't compare training losses, since `rank1_bnn.main()` reports the
    # loss *before* applying the last SGD update: the reported training loss is
    # different from the loss of the checkpointed model.
    self.assertEqual(val_loss['val'], loaded_val_loss['val'])


if __name__ == '__main__':
  tf.test.main()
