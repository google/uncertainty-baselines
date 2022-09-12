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

"""Fewshot learning via multinomial logistic regression.

This binary is designed to showcase how to carry out fewshot experiments
for the following model types: Det, GP, Het and BE. The model list can be
extended in the future, but users can use LogRegFewShotEvaluator in their own
binary (e.g., pretraining scripts).

"""

import functools
import multiprocessing

from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
import flax
import jax
import jax.numpy as jnp
import ml_collections.config_flags

import uncertainty_baselines as ub
import checkpoint_utils  # local file import from baselines.jft
import fewshot_utils  # local file import from baselines.jft


ml_collections.config_flags.DEFINE_config_file(
    'config', None, 'Training configuration.', lock_config=True)
flags.DEFINE_string('output_dir', default=None, help='Work unit directory.')
flags.DEFINE_integer(
    'num_cores', default=None, help='Unused. How many devices being used.')
flags.DEFINE_boolean(
    'use_gpu', default=None, help='Unused. Whether or not running on GPU.')
flags.DEFINE_string('tpu', None,
                    'Unused. Name of the TPU. Only used if use_gpu is False.')

FLAGS = flags.FLAGS


def main(config, output_dir):

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
      output_dir, just_logging=jax.process_index() > 0)
  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.process_index() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing model...')

  if config.model_family == 'single':
    model = ub.models.vision_transformer(
        num_classes=config.num_classes, **config.get('model', {}))
  else:
    model = ub.models.vision_transformer_be(
        num_classes=config.num_classes, **config.get('model', {}))

  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input,
                                           train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    if config.model_family == 'single':
      head_layer_name = 'head'
    else:
      head_layer_name = 'batchensemble_head'
    params[head_layer_name]['bias'] = jnp.full_like(
        params[head_layer_name]['bias'], config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params[head_layer_name]['kernel'] = jnp.full_like(
          params[head_layer_name]['kernel'], 0)

    return params

  batch_size = 4096
  batch_size_eval = batch_size
  local_batch_size = batch_size // jax.process_count()
  image_size = (224, 224, 3)

  seed = config.get('seed', 0)
  rng = jax.random.PRNGKey(seed)
  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  write_note('Loading checkpoint...')

  loader = lambda path: checkpoint_utils.load_checkpoint(tree=None, path=path)
  loaded_params = loader(config.model_init)
  if config.model_family == 'single':
    reinit_params = ('head/kernel', 'head/bias')
  else:
    reinit_params = ('batchensemble_head/bias', 'batchensemble_head/kernel',
                     'batchensemble_head/fast_weight_alpha',
                     'batchensemble_head/fast_weight_gamma')
  loaded = checkpoint_utils.restore_from_pretrained_params(
      params_cpu, loaded_params, config.model.representation_size,
      config.model.classifier, reinit_params)
  loaded_repl = flax.jax_utils.replicate(loaded)

  # Setup function for computing representation.
  @functools.partial(jax.pmap, axis_name='batch')
  def representation_fn(params, images, labels, mask):
    _, outputs = model.apply({'params': flax.core.freeze(params)},
                             images,
                             train=False)
    representation = outputs[config.fewshot.representation_layer]
    if config.model_family == 'batchensemble':
      representation = jnp.concatenate(
          jnp.split(representation, config.model.transformer.ens_size), axis=-1)
    representation = jax.lax.all_gather(representation, 'batch')
    labels = jax.lax.all_gather(labels, 'batch')
    mask = jax.lax.all_gather(mask, 'batch')
    return representation, labels, mask

  write_note('Running fewshot eval...')
  fewshotter = fewshot_utils.LogRegFewShotEvaluator(
      representation_fn, config.fewshot, batch_size_eval,
      config.model.transformer.ens_size,
      config.fewshot.l2_selection_scheme)

  fewshot_results, best_l2 = fewshotter.run_all(
      loaded_repl, config.fewshot.datasets)

  # TODO(dusenberrymw): Remove this once fewshot.py is updated.
  def make_writer_measure_fn(step):

    def writer_measure(name, value):
      writer.write_scalars(step, {name: value})

    return writer_measure

  fewshotter.walk_results(
      make_writer_measure_fn(0), fewshot_results, best_l2)


if __name__ == '__main__':
  # Adds jax flags to the program.
  jax.config.config_with_absl()

  # TODO(dusenberrymw): Refactor `main` such that there is a `train_eval`
  # function that returns values for tests and does not directly access flags,
  # and then have `main` return None.

  def _main(argv):
    del argv
    config = FLAGS.config
    output_dir = FLAGS.output_dir
    main(config, output_dir)

  app.run(_main)  # Ignore the returned values from `main`.

