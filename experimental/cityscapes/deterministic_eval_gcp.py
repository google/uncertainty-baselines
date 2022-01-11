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
Eval vit model on cityscapes.

Step 1: aim to train model on cityscapes for 1 step
# Runs with

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
import custom_segmentation_eval
# scenic dependencies for debugging
from scenic.train_lib import train_utils

# import train_utils  # local file import
import pandas as pd

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

  print('workdir ', output_dir)
  rng, model_cls, dataset, config, workdir, summary_writer = run(config, output_dir)
  print('workdir ', workdir)

  # ----------------------
  # Eval function
  # ----------------------
  eval_fn = custom_segmentation_eval.eval1
  
  # models
  for rng_seed in [0,1,2,3,4]:
    for train_mode in ["deterministic","scratch","gp"]:
      for train_split in [100,75, 50, 25, 10]:
        run_name="{}_split{}_seed{}".format(train_mode, train_split, rng_seed)
        tmp_workdir="gs://ub-ekb/segmenter/cityscapes/run_splits1/checkpoints/{}".format(run_name)
        print("temp directory", tmp_workdir)
        tmp_resultsdir="results/metrics/{}.csv".format(run_name)
        #import pdb; pdb.set_trace();
        train_state, train_summary, eval_summary = eval_fn(rng=rng, model_cls=model_cls, dataset=dataset,
                                                                  config=config,
                                                                  workdir=tmp_workdir, writer=summary_writer)
        print(eval_summary)
        #import pdb;pdb.set_trace()
        df = pd.DataFrame([eval_summary]) 
        df.to_csv (r'{}'.format(tmp_resultsdir), index = False, header=True)

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
