# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

r"""Train vit model on cityscapes."""
import os

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
import jax
import ml_collections.config_flags
from scenic.train_lib_deprecated import train_utils
import tensorflow as tf
from tensorflow.io import gfile
import custom_models  # local file import from experimental.robust_segvit
import custom_segmentation_trainer  # local file import from experimental.robust_segvit

import resource
import sys
import wandb

ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=False, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')
flags.DEFINE_string('dataset_service_address', None,
                    'Address of the tf.data service')
FLAGS = flags.FLAGS


def write_note(note):
  if jax.process_index() == 0:
    logging.info('NOTE: %s', note)


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
  gfile.makedirs(workdir)
  tensorboard_dir = os.path.join(workdir, 'tb_summaries')

  summary_writer = None
  if config.write_summary:
    summary_writer = metric_writers.create_default_writer(
        tensorboard_dir, just_logging=jax.process_index() > 0)

  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  rng = jax.random.PRNGKey(config.rng_seed)
  logging.info('rng: %s', rng)

  if config.model.backbone.type == 'vit':
    if config.model.decoder.type == 'gp':
      model_cls = custom_models.SegmenterGPSegmentationModel
    elif config.model.decoder.type == 'het':
      model_cls = custom_models.SegmenterHetSegmentationModel
    else:
      model_cls = custom_models.SegmenterSegmentationModel
  elif config.model.backbone.type == 'vit_be':
    model_cls = custom_models.SegmenterBESegmentationModel
  else:
    raise NotImplementedError('Model is not implemented {}'.format(
        config.model.backbone.type))
  # ----------------------
  # Load dataset
  # ----------------------
  data_rng, rng = jax.random.split(rng)
  # set resource limit to debug in mac osx
  # (see https://github.com/tensorflow/datasets/issues/1441)
  if jax.process_index() == 0 and sys.platform == 'darwin':
   low, high = resource.getrlimit(resource.RLIMIT_NOFILE)
   resource.setrlimit(resource.RLIMIT_NOFILE, (low, high))
  write_note('Loading dataset...')

  dataset = train_utils.get_dataset(
      config, data_rng, dataset_service_address=FLAGS.dataset_service_address)

  return rng, model_cls, dataset, config, workdir, summary_writer


def main(config, output_dir):

  seed = config.get('rng_seed', 0)
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  # Wandb Setup
  if config.use_wandb:
   gfile.makedirs(config.wandb_dir)
   wandb_args = dict(
       project=config.wandb_project,
       entity=config.get('wandb_entity', 'ub_rdl_big_paper'),
       dir=config.wandb_dir,
       reinit=True,
       name=config.wandb_exp_name,
       group=config.wandb_exp_group,
       sync_tensorboard=True)
   wandb_run = wandb.init(**wandb_args)
   wandb.config.update(FLAGS, allow_val_change=True)
   output_dir = str(os.path.join(output_dir, config.wandb_exp_name))
  else:
   wandb_run = None
   # output_dir = FLAGS.output_dir

  rng, model_cls, dataset, config, _, summary_writer = run(
      config, output_dir)

  # ----------------------
  # Train or Eval function
  # ----------------------
  if config.get('eval_mode', False):
    train_fn = custom_segmentation_trainer.eval_ckpt
  else:
    train_fn = custom_segmentation_trainer.train

  train_fn(
      rng=rng,
      model_cls=model_cls,
      dataset=dataset,
      config=config,
      workdir=output_dir,
      writer=summary_writer)

  if wandb_run is not None:
   wandb_run.finish()

  return


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.config_with_absl()

  def _main(unused_argv):
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.
