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

"""Task specification for BIG-Bench Multple Choice Question Answer (BBMCQA) tasks.

BIG-Bench consists of a collection of tasks intended to probe large language
models and extrapolate their future capabilities [1]. The tasks are highly
varied and authored by a large set of contributors. This mixture compiles the
multiple choice question answer tasks.

## References
[1]: BIG-bench collaboration. Beyond the Imitation Game: Measuring and
     extrapolating the capabilities of language
     models. https://github.com/google/BIG-bench/.
"""
import json
import os
from typing import Optional

# Following import conventions from bigbench codebase.
import bigbench.bbseqio.task_api as bb_task_api
import bigbench.bbseqio.tasks as bb_tasks
import bigbench.bbseqio.vocabs as bb_vocabs

from tensorflow.io import gfile

_BIGBENCH_JSON_PATH = None
_BIGBENCH_TASK_FILE = "task.json"

# These are used as raw strings in the bigbench codebase.
_MULTIPLE_CHOICE_GRADE = "multiple_choice_grade"
_METRICS = "metrics"
_EXAMPLES = "examples"

# A task must have at least this many examples to be included.
_MINIMUM_EXAMPLE_COUNT = 100


# TODO(lyric): Filter out some tasks based on task size or similar criteria. For
# an example, see Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can
# Solve Them.
# TODO(lyric): Should there be a mixture per shot count or a single mixture with
# all shot counts?
def create_mixture_all_multiple_choice_tasks(
    mixture_name_prefix: str,
    num_shots: int = 0,
    vocab: bb_task_api.SeqIOVocabulary = bb_vocabs.T5_DEFAULT_VOCAB,
    max_task_count: Optional[int] = None,
    bigbench_benchmark_tasks_path: Optional[str] = _BIGBENCH_JSON_PATH,
) -> str:
  """Creates a mixture with BIG-Bench multiple choice tasks based on criteria.

  Args:
    mixture_name_prefix: Name to prepend to the created mixture. The full
      mixture name follows BIG-Bench conventions.
    num_shots: The number of shots to provide with each example.
    vocab: The vocab to use when preprocessing the task.
    max_task_count: The number of tasks to include in the mixture. Use None is
      signify all available tasks. Other values would primarily be used for
      faster debugging runs.
    bigbench_benchmark_tasks_path: The path to the bigbench/benchmark_tasks
      directory available from the bigbench github repository at
      https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks.
        The path is listed as Optional for compatibility reasons, but is
        required not to be None.

  Returns:
    The name of the created mixture constructed using the mixture_name_prefix
    and following BIG-Bench naming conventions.

  Raises:
    ValueError if bigbench_benchmark_tasks_path is None.
  """

  if bigbench_benchmark_tasks_path is None:
    raise ValueError("The bigbench_benchmark_tasks_path must be provided.")

  files = gfile.listdir(bigbench_benchmark_tasks_path)

  multiple_choice_tasks = []

  task_dirs = [
      file
      for file in files
      if gfile.isdir(os.path.join(bigbench_benchmark_tasks_path, file))
  ]
  for task_dir in task_dirs:
    task_path = os.path.join(
        bigbench_benchmark_tasks_path, task_dir, _BIGBENCH_TASK_FILE
    )
    if not gfile.exists(task_path):
      continue

    with gfile.GFile(task_path) as f:
      data = json.load(f)

    if _EXAMPLES not in data or len(data[_EXAMPLES]) < _MINIMUM_EXAMPLE_COUNT:
      continue

    if _METRICS in data and _MULTIPLE_CHOICE_GRADE in data[_METRICS]:
      multiple_choice_tasks.append(task_dir)

  count = (
      len(multiple_choice_tasks) if max_task_count is None else max_task_count
  )
  multiple_choice_tasks_mixture_name = (
      bb_tasks.register_mixture_over_bigbench_tasks(
          mixture_name_prefix, multiple_choice_tasks[:count], num_shots, vocab
      )
  )

  return multiple_choice_tasks_mixture_name
