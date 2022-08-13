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

"""Launch bound tightening + final verification on several CPUs.

Example command:
$ gxm launch.py
"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags
import numpy as np
from xmanager import xm
from xmanager import xm_abc

from google3.learning.deepmind.xmanager import hyper

FLAGS = flags.FLAGS

flags.DEFINE_string('exp_name', '', '(optional) Experiment name.')
flags.DEFINE_integer('num_workers', 1000, 'Maximum number of workers.')
flags.DEFINE_string('cell', '', 'Cell (optional)')
flags.DEFINE_enum(
    'tier', None, [tier.name for tier in xm.ServiceTier],
    'Borg priority tier (optional).'
    ' `PROD` is high priority (200) but limited quota.'
    ' `BATCH` is medium priority (100-115) with more quota.'
    ' `FREEBIE` is low priority (25) with almost unlimited quota.'
    ' XManager may fine-tune these priorities.')
config_flags.DEFINE_config_file(
    'config', default='config.py', help_string='config file')
flags.DEFINE_string(
    'workdir', '/cns/li-d/home/{username}/shoshin/{xid}',
    'Working directory for this experiment. The following placeholders are '
    "available: 'username' and 'xid'.")


def _serialize_sweep(sweep, **kwargs):
  return [{**kwargs, **paramset} for paramset in sweep]


def main(unused_argv):
  bazel_args = xm_abc.bazel_args.cpu()
  hardware = {'v100': 1}
  # Share config with binary
  config_resource = xm_abc.Fileset(
      files={config_flags.get_config_filename(FLAGS['config']): 'config.py'})
  kwargs = {}
  kwargs['config'] = config_resource.get_path('config.py', xm_abc.Borg.Spec())
  for key, val in config_flags.get_override_values(FLAGS['config']).items():
    kwargs[f'config.{key}'] = val
  kwargs['workdir'] = FLAGS.workdir + '/{wid}'

  with xm_abc.create_experiment(
      experiment_title='Shoshin CV Ensemble',
      settings=xm_abc.ExecutionSettings(
          max_parallel_work_units=FLAGS.num_workers),
  ) as experiment:

    [executable] = experiment.package([
        xm.bazel_binary(
            label='//third_party/py/uncertainty_baselines/experimental/shoshin:cross_validated_ensemble_training',
            bazel_args=bazel_args,
            dependencies=[config_resource],
            executor_spec=xm_abc.Borg.Spec()),
    ])

    executor = xm_abc.Borg(
        requirements=xm.JobRequirements(
            location=(FLAGS.cell or None),
            service_tier=(xm.ServiceTier[FLAGS.tier] if FLAGS.tier else None),
            **hardware),
        logs_read_access_roles='research-users')

    # Sweep config
    sweep = hyper.product([
        hyper.sweep('config.index',
                    np.arange(10).tolist()),
    ])

    for paramset in sweep:
      args = {**kwargs, **paramset}
      experiment.add(xm.Job(executable, args=args, executor=executor))


if __name__ == '__main__':
  app.run(main)
