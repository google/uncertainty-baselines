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

"""Distribution / parallelism utils."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import logging
import os

import tensorflow as tf


def init_distribution_strategy(force_use_cpu: bool, use_gpu: bool,
                               tpu_name: str):
  """Initialize distribution/parallelization of training or inference.

  Args:
    force_use_cpu: bool, if True, force usage of CPU.
    use_gpu: bool, whether to run on GPU or otherwise TPU.
    tpu_name: str, name of the TPU. Only used if use_gpu is False.

  Returns:
    tf.distribute.Strategy
  """
  if force_use_cpu:
    logging.info('Use CPU')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  elif use_gpu:
    logging.info('Use GPU')

  if force_use_cpu or use_gpu:
    logging.info('Using MirroredStrategy.')
    strategy = tf.distribute.MirroredStrategy()
  else:
    if tpu_name == 'read-from-file':
      with open('tpu_name.txt', 'r') as f:
        tpu_name = f.readline().rstrip()

    logging.info('Use TPU at %s', tpu_name if tpu_name is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  return strategy
