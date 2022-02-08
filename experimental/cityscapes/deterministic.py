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
See run_pretrained.sh for an example
"""

import os
import sys

# %%
import jax
# %%
import tensorflow as tf
# %%
from absl import app
from absl import flags
from absl import logging
from ml_collections.config_flags import config_flags
from tensorflow.io import gfile

import custom_models
import custom_segmentation_trainer
# scenic dependencies for debugging
from scenic.train_lib import train_utils

# import train_utils  # local file import

import wandb
import pathlib
import datetime

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


from clu import metric_writers


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
    write_note('Loading dataset...')

    # TODO: update num_classes
    dataset = train_utils.get_dataset(
        config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

    return rng, model_cls, dataset, config, workdir, summary_writer


def main(config, output_dir):

  print('config')
  print(config)
  seed = config.get('rng_seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  # Wandb Setup
  if config.use_wandb:
      pathlib.Path(config.wandb_dir).mkdir(parents=True, exist_ok=True)
      wandb_args = dict(
          project=config.wandb_project,
          entity='ub_rdl_big_paper',
          dir=config.wandb_dir,
          reinit=True,
          name=config.wandb_exp_name,
          group=config.wandb_exp_group)
      wandb_run = wandb.init(**wandb_args)
      wandb.config.update(FLAGS, allow_val_change=True)
      output_dir = str(
          os.path.join(output_dir,
                       datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')))
  else:
      wandb_run = None
      #output_dir = FLAGS.output_dir

  print('workdir ', output_dir)
  rng, model_cls, dataset, config, workdir, summary_writer = run(config, output_dir)
  print('workdir ', workdir)

  # ----------------------
  # Train function
  # ----------------------
  train_fn = custom_segmentation_trainer.train

  train_state, train_summary, eval_summary = train_fn(rng=rng, model_cls=model_cls, dataset=dataset,
                                                      config=config,
                                                      workdir=output_dir, writer=summary_writer)

  print(train_summary)
  #import pdb; pdb.set_trace()
  if config.use_wandb:
      epoch = int(train_state.global_step)
      wandb.log(train_summary, step=epoch)
      wandb.log(eval_summary, step=epoch)

  if wandb_run is not None:
    wandb_run.finish()
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