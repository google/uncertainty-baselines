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

Evaluate ensemble
"""

import os
import sys

# %%
import jax
import flax
import numpy as np
import jax.numpy as jnp
from flax.training import checkpoints

# %%
import tensorflow as tf
# %%
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags
from tensorflow.io import gfile

import custom_models
import custom_segmentation_eval
# scenic dependencies for debugging
from scenic.train_lib import train_utils
from scenic.model_lib.base_models import model_utils
from scenic.model_lib.base_models.segmentation_model import num_pixels

# import train_utils  # local file import

#%%
config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_string('checkpoint_dir', default=None, help='Checkpoint directory.')

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


from clu import metric_writers



def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints."""
  paths = []
  subdirectories = tf.io.gfile.glob(os.path.join(checkpoint_dir, '*'))
  #is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  is_checkpoint = lambda f: ('checkpoint' in f)

  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = flax.training.checkpoints.latest_checkpoint(path)
        paths.append(latest_checkpoint_without_suffix)
        break
  return paths


def run(config, workdir):
    """Prepares model, and dataset for training.

    This creates summary directories, summary writers, model definition, and
    builds datasets to be sent to the main training script.

    Args:
      config:  ConfigDict; Hyper parameters.
      workdir: string; Root directory for the experiment.

    Returns:
      The outputs of trainer.train(), which are train_state, train_summary, and
        eval_summary.
    """
    lead_host = jax.process_index() == 0
    # set up the train_dir and log_dir
    gfile.makedirs(workdir)
    #workdir = os.path.join(workdir, 'trial')
    #gfile.makedirs(workdir)

    summary_writer = None
    if lead_host and config.write_summary:
        tensorboard_dir = os.path.join(workdir, 'tb_summaries')
        gfile.makedirs(tensorboard_dir)
        # summary_writer = tensorboard.SummaryWriter(tensorboard_dir)
        summary_writer = metric_writers.SummaryWriter(tensorboard_dir)

    device_count = jax.device_count()
    logging.info('device_count: %d', device_count)
    logging.info('num_hosts : %d', jax.process_count())
    logging.info('host_id : %d', jax.process_index())

    rng = jax.random.PRNGKey(config.rng_seed)
    logging.info('rng: %s', rng)

    # ----------------------
    # Call Model
    # ----------------------

    model_cls = custom_models.SegmenterSegmentationModel

    # ----------------------
    # Load dataset
    # ----------------------
    data_rng, rng = jax.random.split(rng)
    # set resource limit to debug in mac osx (see https://github.com/tensorflow/datasets/issues/1441)
    if jax.process_index() == 0 and sys.platform == 'darwin':
        import resource
        low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (high, high))


    eval_dataset_name = config.get('eval_dataset_name', 'cityscapes_val')

    write_note('Loading dataset... {}'.format(eval_dataset_name))

    # TODO: update num_classes
    if eval_dataset_name  == 'cityscapes_val':
        dataset = train_utils.get_dataset(
            config, data_rng, dataset_service_address=FLAGS.dataset_service_address)


    return rng, model_cls, dataset, config, workdir, summary_writer


def main(config, output_dir,checkpoint_dir):

  print('config')
  print(config)
  seed = config.get('rng_seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  # ----------------------
  # Get dataset
  # ----------------------

  print('workdir ', output_dir)
  rng, model_cls, dataset, config, workdir, summary_writer = run(config, output_dir)

  num_eval_examples = dataset.meta_data['num_eval_examples']
  num_eval_steps = int(np.ceil(num_eval_examples / config.batch_size))
  assert config.batch_size == 1

  # ----------------------
  # Buils Model
  # ----------------------

  # Build dummy input
  input_shape = [1] + list(dataset.meta_data['input_shape'][1:])
  #input_shape = dataset.meta_data['input_shape']
  in_st = dataset.meta_data['input_dtype']

  dummy_input = jnp.zeros(input_shape, in_st.dtype)

  # Init model
  rng, init_rng = jax.random.split(rng)
  model = model_cls(config, dataset.meta_data) # extracting number of classes in meta_data
  flax_model = model.flax_model
  init_model_state, init_params = flax_model.init(
      init_rng, dummy_input, train=False, debug=False).pop('params')


  ensemble_filenames = parse_checkpoint_dir(checkpoint_dir)
  ensemble_size = len(ensemble_filenames)

  # ----------------------
  # Evaluate models
  # ----------------------
  num_eval_steps = 1
  dataset_name='trial'
  # dict_keys(['batch_mask', 'inputs', 'label'])

  # -------------------------------
  # Write Model Predictions to file
  # -------------------------------

  # TODO: reset iterator
  test_iterator = dataset.valid_iter
  #import pdb; pdb.set_trace()
  for m, ensemble_filename in enumerate(ensemble_filenames):
    #train_state = checkpoints.restore_checkpoint(ensemble_filename, init_model_state)

    variables = {'params': init_params, **init_model_state}

    # assume only one test_set
    #test_iterator = iter(test_dataset)
    for _ in range(num_eval_steps): # num_eval_steps
      eval_batch = next(dataset.valid_iter) #dict_keys(['batch_mask', 'inputs', 'label'])
      inputs = eval_batch['inputs'][0]
      logits, outs = flax_model.apply(variables, inputs, train=False, mutable=False)

      targets = eval_batch['label'][0]
      weights = eval_batch['batch_mask'][0]
      one_hot_targets = flax.training.common_utils.onehot(targets, dataset.meta_data['num_classes'])

      correct = model_utils.weighted_correctly_classified(logits, one_hot_targets, weights)

      number_pixels = num_pixels(logits,one_hot_targets,weights)

      accuracy = correct.sum()/number_pixels

      loss = model_utils.weighted_softmax_cross_entropy(logits, one_hot_targets, weights)


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
    checkpoint_dir = FLAGS.checkpoint_dir
    main(config, output_dir, checkpoint_dir)

  app.run(_main)  # Ignore the returned values from `main`.