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

"""
Train vit model on cityscapes.

Step 1: aim to train model on cityscapes for 1 step
# Runs with

"""
import functools

from functools import partial  # pylint: disable=g-importing-member so standard
import itertools
import multiprocessing
import numbers
import os
import sys
#%%
from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
#%%
import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
from ml_collections.config_flags import config_flags
import numpy as np
import robustness_metrics as rm
#%%
import tensorflow as tf
#import train_utils  # local file import
import uncertainty_baselines as ub

# scenic dependencies for debugging
from scenic.train_lib import lr_schedules
from scenic.train_lib import optimizers
from scenic.train_lib import train_utils
from flax import jax_utils

import custom_models
#%%
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_string('dataset_service_address', None,
                    'Address of the tf.data service')
FLAGS = flags.FLAGS


def write_note(note):
    if jax.process_index() == 0:
        logging.info('NOTE: %s', note)
#%%
def main(config, output_dir):
  seed = config.get('rng_seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  write_note('Initializing...')

  # Train dataset configs
  data_rng, rng = jax.random.split(rng)

  # ----------------------
  # Load dataset
  # ----------------------
  # set resource limit to debug in mac osx (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
    import resource
    low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))
  write_note('Loading dataset...')

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  # ----------------------
  # Define model
  # ----------------------
  write_note('Creating model...')
  model = ub.models.segmenter_transformer(
      num_classes=config.num_classes,
      patches=config.patches,
      backbone_configs=config.backbone_configs,
      decoder_configs=config.decoder_configs
  )
  # ----------------------
  # Initialize model
  # ----------------------
  # Here we follow the scenic/model_lib/base_models/segmentation_model.py
  from scenic.train_lib.train_utils import initialize_model
  """
  #TODO(kellybuchanan): update local_batch_size according to train_utils
  local_batch_size = 1
  @partial(jax.jit, backend='cpu')
  def init(rng):
    #image_size = tuple(train_ds.element_spec['image'].shape[2:])
    image_size = config.dataset_configs.target_size + (3,)
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input,
                                           train=False))['params']

    return params
  rng, init_rng = jax.random.split(rng)
  params_cpu = init(init_rng)
  """
  rng, init_rng = jax.random.split(rng)
  (params, model_state, num_trainable_params,
   gflops) = train_utils.initialize_model(
      model_def=model, #.flax_model,
      input_spec=[(dataset.meta_data['input_shape'],
                   dataset.meta_data.get('input_dtype', jnp.float32))],
      config=config,
      rngs=init_rng)

  # ----------------------
  # Create optimizer
  # ----------------------
  """
  # Load the optimizer from flax.
  opt_name = config.get('optimizer')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optimizer_configs', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)
  """
  optimizer = jax.jit(
      optimizers.get_optimizer(config).create, backend='cpu')(
          params)
  rng, train_rng = jax.random.split(rng)
  train_state = train_utils.TrainState(
      global_step=0,
      optimizer=optimizer,
      model_state=model_state,
      rng=train_rng,
      accum_train_time=0)

  start_step = train_state.global_step

  if config.checkpoint:
    train_state, start_step = train_utils.restore_checkpoint(
        workdir, train_state)
  # Replicate the optimzier, state, and rng.
  train_state = jax_utils.replicate(train_state)
  del params  # Do not keep a copy of the initial params.

  # Calculate the total number of training steps.
  total_steps, steps_per_epoch = train_utils.get_num_training_steps(
      config, dataset.meta_data)
  # Get learning rate scheduler.
  learning_rate_fn = lr_schedules.get_learning_rate_fn(config)

  # --- STOP ---
  # TODO: debug train_step in scenic/train_lib/segmentation_trainer.py
  # import pdb; pdb.set_trace()

  train_step_pmapped = jax.pmap(
      functools.partial(
          train_step,
          flax_model=model,#.flax_model,
          learning_rate_fn=learning_rate_fn,
          loss_fn=model.loss_function,
          metrics_fn=model.get_metrics_fn('train'),
          config=config,
          debug=config.debug_train),
      axis_name='batch',
      # We can donate both buffers of train_state and train_batch.
      donate_argnums=(0, 1),
  )


  #dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)

  #inputs = jnp.ones([num_examples, img_h, img_w, 3], jnp.float32)
  #model = ub.models.segmenter_transformer(**config)
  #key = jax.random.PRNGKey(0)
  #variables = model.init(key, inputs, train=False)
  #logits, outputs = model.apply(variables, inputs, train=False)
  #variables = model.init(rng, inputs, train=False)
  #logits, outputs = model.apply(variables, inputs, train=False)

  return


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.config_with_absl()

  # TODO(dusenberrymw): Refactor `main` such that there is a `train_eval`
  # function that returns values for tests and does not directly access flags,
  # and then have `main` return None.

  def _main(unused_argv):
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.