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

"""XManager launcher for both GPU and TPU jobs.

The launcher works with any Python binary with the following flags:

* `output_dir` is the directory for saving summaries and logs;
* `use_gpu` determines whether to run on GPU or otherwise TPU;
* `num_cores` is the number of TPU cores or GPUs;
* `tpu` is the TPU main address (flag not required if launching with GPU);
* `seed` is the experiment's random seed.

For binaries that support only certain accelerator settings, we recommend still
using these flags. Raise errors outside its support or rely on runtime errors.

To learn about experiment workflows, see
`third_party/py/uncertainty_baselines/baselines/README.md`.
"""

import dataclasses
import getpass
import importlib.util
import inspect
import json
import os
import random
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Text

from absl import app
from absl import flags
from absl import logging
from ml_collections.config_dict import config_dict
from xmanager import resource_selector as rs
from xmanager import xm
from xmanager import xm_local
from xmanager.contrib import copybara

# pylint: disable=g-import-not-at-top
try:
  from uncertainty_baselines import halton
except (ImportError, ModuleNotFoundError):
  logging.exception('Cannot import halton sequence generator.')
  halton = None

hyper = None


# pylint: enable=g-import-not-at-top

# Binary flags
flags.DEFINE_string(
    'binary',
    None,
    'Filepath to Python script to run. For external GCS experiments, it can be '
    'an absolute path to the binary, or a relative one with respect to the '
    'current folder.'
    'Alternatively, if `use_fragmented_python` is True, this can be a '
    'Fragmented Python build MPM such as'
    '"//third_party/py/uncertainty_baselines/baselines/jft:deterministic_fragmented_mpms".'
)
flags.mark_flag_as_required('binary')
flags.DEFINE_list(
    'args', [], 'Flag arguments to pass to binary. Follow the format '
    '--args=batch_size=64,train_epochs=300.')
flags.DEFINE_string(
    'config', None, 'Filepath to Python file with a function '
    'get_sweep(hyper) returning a hyperparameter sweep and/or '
    'a function get_config() returning a ConfigDict.')
flags.DEFINE_bool(
    'use_halton_generator', False,
    'Whether to use the open-sourced Halton generator or an internal generator '
    'to generate hyperparameter sweeps.')
flags.DEFINE_bool('launch_on_gcp', False, 'Whether or not to launch on GCS.')
flags.DEFINE_string(
    'cell', None,
    'Cloud region or cell for the worker (and coordinator if using TPU).')
flags.DEFINE_multi_string(
    'gin_file', default=None, help='Path to the gin configuration file.')

# Accelerator flags
flags.DEFINE_string('platform', None, 'Platform (e.g., tpu-v2, tpu-v3, gpu).')
flags.DEFINE_string(
    'topology', '2x2',
    'Accelerator topology in the format described by `xm.Topology. Topologies '
    'have a form of "NxM_suffix" where N & M are the number of accelerators '
    'across the dimension and suffix corresponds to a specific interconnect '
    'type. Number of dimensions may vary. Examples of valid topologies: "4" '
    'means 4 GPUs on one host. "4x4" means a 4x4 TPU grid. For TPU, {x}x{y} '
    'means x*x **chips**, and because the number of devices is the number of '
    'cores, we further multiply by 2 because there are 2 cores per chip. For '
    'example, 2x2 is equivalent to an 8 core TPU slice, 8x8 = 128 cores, etc.')
flags.DEFINE_string('gpu_type', 'p100',
                    'GPU type. Only used if platform is GPU.')
flags.DEFINE_integer('num_gpus', None,
                     'Number of GPUs. Only used if platform is GPU.')
flags.DEFINE_integer('num_cpus', None,
                     'Number of CPUs. Only used if launching on GCP.')
flags.DEFINE_integer(
    'memory', None, 'Amount of CPU memory in GB. Only used if launching on '
    'GCP.')
flags.DEFINE_string('experiment_name', None,
                    'Experiment name; defaults to timestamp.')
flags.DEFINE_integer('num_runs', 1,
                     'Number of runs each with a different seed.')
flags.DEFINE_bool(
    'use_jax', False, 'Whether or not the binary uses a JAX setup'
    'like baselines/jft.')
flags.DEFINE_bool(
    'use_t5', False, 'Whether or not the binary uses a T5 setup like '
    'baselines/t5.')
flags.DEFINE_string('notes', default='', help='Notes for the experiment.')
flags.DEFINE_list('tags', [], 'Experiment tags.')
flags.DEFINE_enum(
    'priority',
    'prod',
    enum_values=['prod', 'batch', 'freebie', 'auto'],
    help='Priority of the GPU/TPU worker.')


# TODO(dusenberrymw): Generalize these with flags in the v2 rewrite.
# Legacy hard-coding of subdirectories using JAX, config flags, etc.
DIRS_USING_JAX = [
    '/robust_segvit',
    '/diabetic_retinopathy_detection',
    '/jft',
    '/near_ood',
]

DIRS_USING_CONFIG = DIRS_USING_JAX + [
]

FLAGS = flags.FLAGS


@dataclasses.dataclass
class _JobMetadata:
  """XManager job metadata."""
  user: str
  cell: str
  platform: xm.ResourceType
  topology: xm.Topology
  num_cpus: int
  experiment_name: str
  memory: int
  notes: str
  tags: str
  priority: xm.ServiceTier


def _get_attr(config, name: str, default: Any = None) -> Optional[Any]:
  """Get a given attribute from the passed FLAGS or ConfigDict."""
  # Note that if a flag is passed with its default value, this will not override
  # a conflicting config value.
  has_flag_value = name in FLAGS and FLAGS[name].value != FLAGS[name].default
  if has_flag_value:
    return FLAGS[name].value
  elif config and name in config:
    return config[name]
  elif name in FLAGS:
    return FLAGS[name].default
  return default


def _build_binary_metadata(config):
  """Extracts job metadata and args from the given ConfigDict and/or FLAGS."""
  if FLAGS.binary[:2] == '//':
    # We assume the path will have at least two cmds split by '/' and
    # We will use the last two to name the experiment.
    # Ideally, the path will look like //.../{dataset}/{baseline}.py
    # but {dataset} and {baseline} can be any string in practice.
    command = FLAGS.binary.split('/')
    if len(command) >= 2:
      dataset = command[-2]
      baseline = command[-1]
      baseline = os.path.splitext(baseline)[0]
    else:
      dataset = None
      baseline = None
  else:
    pieces = FLAGS.binary.split('/')
    dataset = pieces[-2]
    baseline = pieces[-1]
    baseline = os.path.splitext(baseline)[0]

  if ':' in baseline:
    baseline = baseline.split(':')[0]

  if config:
    flag_args = _get_attr(config, 'args')
    if not isinstance(flag_args, (dict, config_dict.ConfigDict)):
      flag_args = dict(arg.split('=', 1) for arg in flag_args)
    experiment_name = _get_attr(config, 'experiment_name')
  else:
    flag_args = dict(arg.split('=', 1) for arg in FLAGS.args)
    experiment_name = FLAGS.experiment_name
  dataset = flag_args.get('dataset', dataset)

  if not experiment_name:
    # Set default experiment name to {dataset}-{baseline}, assuming both exist
    # and the name fits into a max character length of 40.
    experiment_name = 'uncertainty_baselines'
    if baseline is not None:
      name = baseline
      experiment_name = name if len(name) < 40 else experiment_name
    if dataset is not None:
      name = f'{dataset}-{experiment_name}'
      experiment_name = name if len(name) < 40 else experiment_name
    experiment_name = experiment_name.lower()

  user = _get_attr(config, 'user')
  metadata = _JobMetadata(
      user=user,
      cell=_get_attr(config, 'cell'),
      platform=xm.ResourceType[_get_attr(config, 'platform')],
      topology=xm.Topology(_get_attr(config, 'topology')),
      num_cpus=_get_attr(config, 'num_cpus'),
      memory=_get_attr(config, 'memory'),
      experiment_name=experiment_name,
      priority=xm.ServiceTier[_get_attr(config, 'priority')],
      notes=_get_attr(config, 'notes'),
      tags=_get_attr(config, 'tags'),
  )

  metadata.tags.append(metadata.priority.name.lower())

  use_gpu = (
      metadata.platform in xm.GpuType or
      metadata.platform == xm.ResourceType.CPU)

  if metadata.platform == xm.ResourceType.CPU:
    num_cores = 1
  elif metadata.platform in xm.GpuType:
    num_cores = metadata.topology.chip_count
  else:  # TPU
    num_cores = 2 * metadata.topology.chip_count  # 2 cores per TPU chip.

  if 'num_cores' in flag_args and flag_args['num_cores'] != num_cores:
    raise ValueError(
        '"num_cores" requested in binary incompatible with inferred number of '
        'cores based on topology and platform ({}!={} respectively)'
        .format(flag_args['num_cores'], num_cores))
  args = dict(num_cores=num_cores, use_gpu=use_gpu)
  args.update(flag_args)
  return args, metadata


  logging.info('Using %s as the project dir.', project_dir)

  # TODO(znado): support different caip regions, etc.?
  with xm_local.create_experiment(metadata.experiment_name) as experiment:
    # Note that we normally would need to append a "$@" in order to properly
    # forward the args passed to the job into the python command, but the XM
    # library already does this for us.
    run_cmd = f'python {binary_path}'
    # These images are necessary to get tf-nightly pre-installed.
    # Our lazy loading `__getattr__ = _lazy_import` in `__init__.py` requires
    # at least Python 3.7, so we use a base image that has Python 3.7.
    if metadata.platform_str == 'gpu':
      base_image = 'tensorflow/tensorflow:nightly-gpu'
      # base_image = 'gcr.io/deeplearning-platform-release/tf2-gpu.2-5'
    else:
      base_image = 'tensorflow/tensorflow:nightly'
      # base_image = 'gcr.io/deeplearning-platform-release/tf2-cpu.2-5'
    pip_cmd = 'pip --no-cache-dir install'
    spec = xm.PythonContainer(
        path=project_dir,
        base_image=base_image,
        entrypoint=xm.CommandList([run_cmd]),
        docker_instructions=[
            f'COPY {os.path.basename(project_dir)}/ uncertainty-baselines',
            'RUN apt-get update && apt-get install -y git netcat',
            'RUN python -m pip install --upgrade pip setuptools wheel',
            # # Uninstall TF2.5 so that the UB pip install will install nightly.
            # 'RUN python -m pip uninstall -y tensorflow tf-nightly',
            f'RUN {pip_cmd} google-cloud-storage',
            f'RUN {pip_cmd} ./uncertainty-baselines[experimental,models]',
            'WORKDIR uncertainty-baselines',
        ],
    )
    [executable] = experiment.package([
        xm.Packageable(
            executable_spec=spec,
            executor_spec=xm_local.Caip.Spec(),
        ),
    ])

    platform = {}
    if 'tpu' in metadata.platform_str:
      # To run on a tpu-v2-8, tpu_topology should be 2x2.
      pieces = map(int, metadata.topology.split('x'))
      num_tpus = pieces[0] * pieces[1] * 2  # 2 cores per TPU chip.
      platform = {metadata.platform_str.split('-')[-1]: num_tpus}
    elif metadata.platform_str == 'gpu':
      platform = {metadata.platform: metadata.topology}

    if metadata.num_cpus is not None:
      platform['cpu'] = metadata.num_cpus * xm.vCPU
    if metadata.memory is not None:
      platform['memory'] = metadata.memory * xm.GiB
    executor = xm_local.Caip(xm.JobRequirements(**platform))

    # Create one job per setting in the hyperparameter sweep. The default case
    # is a length 1 sweep with a single argument name "seed".
    for ji, sweep_args in enumerate(sweep):
      job_args = args.copy()
      if 'output_dir' in job_args:
        job_args['output_dir'] = os.path.join(job_args['output_dir'], str(ji))
      if 'data_dir' in job_args and job_args.get('download_data', False):
        job_args['data_dir'] = os.path.join(job_args['data_dir'], str(ji))
      # Overwrite any values in `args` with the `sweep_args`.
      job_args.update(sweep_args)
      logging.info('Launching job %d/%d with args %s.\n', ji + 1, len(sweep),
                   json.dumps(job_args, indent=4, sort_keys=True))
      job = xm.Job(
          executable=executable,
          executor=executor,
          args=job_args,
      )
      experiment.add(job)


def _generate_hyperparameter_sweep(
    config_module, config: config_dict.ConfigDict,
    project_dir: Optional[str]) -> List[Dict[Text, Any]]:
  """Generate the hyperparameter sweep."""
  config_use_halton_generator = (
      FLAGS.config and config is not None and
      'use_halton_generator' in config and config.use_halton_generator)
  use_halton_generator = (
      FLAGS.use_halton_generator or config_use_halton_generator or not hyper)
  hyper_module = halton if use_halton_generator else hyper
  if use_halton_generator and halton is None:
    logging.info(
        'Could not import Uncertainty Baselines but requested generating '
        'hyperparameter sweeps using the Halton sequence generator. Falling '
        'back to local copy.')
    # We should only need to do this on external GCS.
    halton_path = os.path.join(project_dir, 'uncertainty_baselines/halton.py')
    hyper_module_spec = importlib.util.spec_from_file_location(
        '', os.path.abspath(halton_path))
    hyper_module = importlib.util.module_from_spec(hyper_module_spec)
    hyper_module_spec.loader.exec_module(hyper_module)
  if FLAGS.config and 'get_sweep' in dir(config_module):
    if hyper_module is None:
      raise ValueError('Need a hyperparameter module to construct sweep.')
    if FLAGS.num_runs != 1:
      raise ValueError('FLAGS.num_runs not supported with config.get_sweep().')
    sweep = config_module.get_sweep(hyper_module)
  elif '/t5' in FLAGS.binary:
    num_runs = FLAGS.num_runs
    if num_runs > 1:
      random.seed(0)
      # NOTE: Seeds must be in [0, 2**31 - 1], a non-negative signed int32.
      train_seeds = [random.randint(0, 2**31 - 1) for _ in range(num_runs)]
      dataset_seeds = [random.randint(0, 2**31 - 1) for _ in range(num_runs)]
      train_sweep = hyper_module.sweep('_gin.train.random_seed', train_seeds)
      dataset_sweep = hyper_module.sweep(
          '_gin.train__.utils.DatasetConfig.seed', dataset_seeds)
      sweep = hyper_module.zipit([train_sweep, dataset_sweep])
    # If `num_runs == 1`, we will not use sweep.
    else:
      sweep = hyper_module.product([])
  else:
    sweep = [
        # NOTE: NumPy seeds must be in [0, 2**32 - 1], i.e., an unsigned int32.
        {
            'seed': seed + random.randint(0, 2**32 - FLAGS.num_runs)
        } for seed in range(FLAGS.num_runs)
    ]
  return sweep


def _load_config_helper(config_path, launch_on_gcp):
  """Get the ConfigDict from config_path:get_config()."""
  config_module_spec = importlib.util.spec_from_file_location(
      '', os.path.abspath(config_path))
  config_module = importlib.util.module_from_spec(config_module_spec)
  config_module_spec.loader.exec_module(config_module)
  config = None
  if 'get_config' in dir(config_module):
    # Check if get_config takes a parameter called launch_on_gcp, and if so then
    # pass in FLAGS.launch_on_gcp.
    get_config_inspect = inspect.getfullargspec(config_module.get_config)
    get_config_params = get_config_inspect.args
    if 'launch_on_gcp' in get_config_params:
      config = config_module.get_config(launch_on_gcp=launch_on_gcp)
    else:
      config = config_module.get_config()
  return config_module, config


def _load_config(config_path, launch_on_gcp):
  """Load the ConfigDict if one was passed in as FLAGS.config."""
  if config_path:
    config_module = None
    if not config_module:
      config_module, config = _load_config_helper(config_path, launch_on_gcp)
  else:
    config_module = None
    config = None
  return config_module, config


def main(argv):
  del argv  # unused arg
  if '/t5' in FLAGS.binary and not FLAGS.gin_file:
    raise ValueError('Missing `gin_file` flag for t5 experiment.')
  config_module, config = _load_config(FLAGS.config, FLAGS.launch_on_gcp)
  args, metadata = _build_binary_metadata(config)
  if FLAGS.launch_on_gcp:
    project_dir, binary_path = _split_path_to_ub(FLAGS.binary)
    sweep = _generate_hyperparameter_sweep(config_module, config, project_dir)
    return _launch_gcp_experiment(project_dir, binary_path, sweep, args,
                                  metadata)


if __name__ == '__main__':
  app.run(main)
