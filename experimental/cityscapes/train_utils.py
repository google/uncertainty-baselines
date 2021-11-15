
"""
Load cityscapes dataset
"""
from absl import logging
from typing import Any, Callable, Dict, Tuple, Sequence, Optional, Mapping, Union
import jax
import jax.numpy as jnp
import ml_collections

PRNGKey = jnp.ndarray

from scenic.dataset_lib import datasets


def get_dataset(config: ml_collections.ConfigDict, data_rng: PRNGKey, *,
                dataset_service_address: Optional[str] = None):
  """Creates dataset from config.
  Edited from
  https://github.com/google-research/scenic/blob/c3ae6d7b5dc829fafe204a92522a5983959561a0/scenic/train_lib/train_utils.py#L145
  """
  device_count = jax.device_count()
  logging.info('device_count: %d', device_count)
  logging.info('num_hosts : %d', jax.process_count())
  logging.info('host_id : %d', jax.process_index())

  dataset_builder = datasets.get_dataset(config.dataset_name)

  batch_size = config.batch_size
  if batch_size % device_count > 0:
    raise ValueError(f'Batch size ({batch_size}) must be divisible by the '
                     f'number of devices ({device_count})')

  eval_batch_size = config.get('eval_batch_size', batch_size)
  if eval_batch_size % device_count > 0:
    raise ValueError(f'Eval batch size ({eval_batch_size}) must be divisible '
                     f'by the number of devices ({device_count})')

  local_batch_size = batch_size // jax.process_count()
  eval_local_batch_size = eval_batch_size // jax.process_count()
  device_batch_size = batch_size // device_count
  logging.info('local_batch_size : %d', local_batch_size)
  logging.info('device_batch_size : %d', device_batch_size)

  shuffle_seed = config.get('shuffle_seed', None)
  if dataset_service_address and shuffle_seed is not None:
    raise ValueError('Using dataset service with a random seed causes each '
                     'worker to produce exactly the same data. Add '
                     'config.shuffle_seed = None to your config if you want '
                     'to run with dataset service.')

  dataset = dataset_builder(
      batch_size=local_batch_size,
      eval_batch_size=eval_local_batch_size,
      num_shards=jax.local_device_count(),
      dtype_str=config.data_dtype_str,
      rng=data_rng,
      shuffle_seed=shuffle_seed,
      dataset_configs=config.get('dataset_configs'),
      dataset_service_address=dataset_service_address)

  return dataset