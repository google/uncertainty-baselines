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

"""Hyperparameter sweeps with Halton sequences of quasi-random numbers.

Based off the algorithms described in https://arxiv.org/abs/1706.03200.
"""

import collections
import functools
import itertools
import math

from typing import Any, Callable, Dict, List, Sequence, Text, Tuple, Union
from numpy import random


_SweepSequence = List[Dict[Text, Any]]
_GeneratorFn = Callable[[float], Tuple[Text, float]]


def generate_primes(n: int):
  """Generate primes less than `n` (except 2) using the Sieve of Sundaram."""
  half_m1 = int((n - 2) / 2)
  sieve = [0] * (half_m1 + 1)
  for outer in range(1, half_m1 + 1):
    inner = outer
    while outer + inner + 2 * outer * inner <= half_m1:
      sieve[outer + inner + (2 * outer * inner)] = 1
      inner += 1
  return [2 * i + 1 for i in range(1, half_m1 + 1) if sieve[i] == 0]


def _is_prime(n: int):
  """Check if `n` is a prime number."""
  return all(n % i != 0 for i in range(2, int(n ** 0.5) + 1)) and n != 2


def _generate_dim(num_samples: int,
                  base: int,
                  per_dim_shift: bool,
                  shuffled_seed_sequence: List[int]):
  """Generate `num_samples` from a Van der Corput sequence with base `base`.

  Args:
    num_samples: int, the number of samples to generate.
    base: int, the base for the Van der Corput sequence. Must be prime.
    per_dim_shift: boolean, if true then each dim in the sequence is shifted by
      a random float (and then passed through fmod(n, 1.0) to keep in the range
      [0, 1)).
    shuffled_seed_sequence: an optional list of length `base`, used as the input
      sequence to generate samples. Useful for deterministic testing.

  Returns:
    A shuffled Van der Corput sequence of length `num_samples`, and optionally a
    shift added to each dimension.

  Raises:
    ValueError: if `base` is negative or not prime.
  """
  if base < 0 or not _is_prime(base):
    raise ValueError(
        'Each Van der Corput sequence requires a prime `base`, received '
        '{}.'.format(base))

  rng = random.RandomState(base)
  if shuffled_seed_sequence is None:
    shuffled_seed_sequence = list(range(1, base))
    # np.random.RandomState uses MT19937 (see
    # https://numpy.org/devdocs/reference/random/legacy.html#numpy.random.RandomState).
    rng.shuffle(shuffled_seed_sequence)
    shuffled_seed_sequence = [0] + shuffled_seed_sequence

  # Optionally generate a random float in the range [0, 1) to shift this
  # dimension by.
  dim_shift = rng.random_sample() if per_dim_shift else None

  dim_sequence = []
  for i in range(1, num_samples + 1):
    num = 0.
    denominator = base
    while i:
      num += shuffled_seed_sequence[i % base] / denominator
      denominator *= base
      i //= base
    if per_dim_shift:
      num = math.fmod(num + dim_shift, 1.0)
    dim_sequence.append(num)
  return dim_sequence


Matrix = List[List[int]]


def generate_sequence(num_samples: int,
                      num_dims: int,
                      skip: int = 100,
                      per_dim_shift: bool = True,
                      shuffle_sequence: bool = True,
                      primes: Sequence[int] = None,
                      shuffled_seed_sequence: Matrix = None) -> Matrix:
  """Generate `num_samples` from a Halton sequence of dimension `num_dims`.

  Each dimension is generated independently from a shuffled Van der Corput
  sequence with a different base prime, and an optional shift added. The
  generated points are, by default, shuffled before returning.

  Args:
    num_samples: int, the number of samples to generate.
    num_dims: int, the number of dimensions per generated sample.
    skip: non-negative int, if positive then a sequence is generated and the
      first `skip` samples are discarded in order to avoid unwanted
      correlations.
    per_dim_shift: boolean, if true then each dim in the sequence is shifted by
      a random float (and then passed through fmod(n, 1.0) to keep in the range
      [0, 1)).
    shuffle_sequence: boolean, if true then shuffle the sequence before
      returning.
    primes: an optional sequence (of length `num_dims`) of prime numbers to use
      as the base for the Van der Corput sequence for each dimension. Useful for
      deterministic testing.
    shuffled_seed_sequence: an optional list of length `num_dims`, with each
      element being a sequence of length `primes[d]`, used as the input sequence
      to the Van der Corput sequence for each dimension. Useful for
      deterministic testing.

  Returns:
    A shuffled Halton sequence of length `num_samples`, where each sample has
    `num_dims` dimensions, and optionally a shift added to each dimension.

  Raises:
    ValueError: if `skip` is negative.
    ValueError: if `primes` is provided and not of length `num_dims`.
    ValueError: if `shuffled_seed_sequence` is provided and not of length
      `num_dims`.
    ValueError: if `shuffled_seed_sequence[d]` is provided and not of length
      `primes[d]` for any d in range(num_dims).
  """
  if skip < 0:
    raise ValueError('Skip must be non-negative, received: {}.'.format(skip))

  if primes is not None and len(primes) != num_dims:
    raise ValueError(
        'If passing in a sequence of primes it must be the same length as '
        'num_dims={}, received {} (len {})'.format(
            num_dims, primes, len(primes)))

  if shuffled_seed_sequence is not None:
    if len(shuffled_seed_sequence) != num_dims:
      raise ValueError(
          'If passing in `shuffled_seed_sequence` it must be the same length '
          'as num_dims={}, received {} (len {})'.format(
              num_dims, shuffled_seed_sequence, len(shuffled_seed_sequence)))
    for d in range(num_dims):
      if len(shuffled_seed_sequence[d]) != primes[d]:
        raise ValueError(
            'If passing in `shuffled_seed_sequence` it must have element `{d}` '
            'be a sequence of length `primes[{d}]`={expected}, received '
            '{actual} (len {length})'.format(
                d=d, expected=primes[d], actual=shuffled_seed_sequence[d],
                length=shuffled_seed_sequence[d]))

  if primes is None:
    primes = []
    prime_attempts = 1
    while len(primes) < num_dims + 1:
      primes = generate_primes(1000 * prime_attempts)
      prime_attempts += 1
    primes = primes[-num_dims-1:-1]

  # Skip the first `skip` points in the sequence because they can have unwanted
  # correlations.
  num_samples += skip

  halton_sequence = []
  for d in range(num_dims):
    if shuffled_seed_sequence is None:
      dim_shuffled_seed_sequence = None
    else:
      dim_shuffled_seed_sequence = shuffled_seed_sequence[d]
    dim_sequence = _generate_dim(
        num_samples=num_samples,
        base=primes[d],
        shuffled_seed_sequence=dim_shuffled_seed_sequence,
        per_dim_shift=per_dim_shift)
    dim_sequence = dim_sequence[skip:]
    halton_sequence.append(dim_sequence)

  # Transpose the 2-D list to be shape [num_samples, num_dims].
  halton_sequence = list(zip(*halton_sequence))

  # Shuffle the sequence.
  if shuffle_sequence:
    random.shuffle(halton_sequence)
  return halton_sequence


def _generate_double_point(
    name: Text,
    min_val: float,
    max_val: float,
    scaling: Text,
    halton_point: float) -> float:
  """Generate a float hyperparameter value from a Halton sequence point."""
  if scaling not in ['linear', 'log']:
    raise ValueError(
        'Only log or linear scaling is supported for floating point '
        'parameters. Received {}.'.format(scaling))
  if scaling == 'log':
    # To transform from [0, 1] to [min_val, max_val] on a log scale we do:
    # min_val * exp(x * log(max_val / min_val)).
    rescaled_value = (
        min_val * math.exp(halton_point * math.log(max_val / min_val)))
  else:
    rescaled_value = halton_point * (max_val - min_val) + min_val
  return name, rescaled_value


def _generate_discrete_point(
    name: str,
    feasible_points: Sequence[Any],
    halton_point: float) -> Any:
  """Generate a discrete hyperparameter value from a Halton sequence point."""
  index = int(math.floor(halton_point * len(feasible_points)))
  return name, feasible_points[index]


_DiscretePoints = collections.namedtuple('_DiscretePoints', 'feasible_points')


def categorical(feasible_points: Any) -> Sequence[Any]:
  if not isinstance(feasible_points, (list, tuple)):
    feasible_points = [feasible_points]
  return _DiscretePoints(feasible_points)


def discrete(feasible_points: Sequence[Any]) -> Sequence[Any]:
  return _DiscretePoints(feasible_points)


def interval(start: int, end: int) -> Tuple[int, int]:
  return start, end


def loguniform(name: Text, range_endpoints: Tuple[int, int]) -> _GeneratorFn:
  min_val, max_val = range_endpoints
  return functools.partial(
      _generate_double_point, name, min_val, max_val, 'log')


def uniform(
    name: Text,
    search_points: Union[_DiscretePoints, Tuple[int, int]]) -> _GeneratorFn:
  if isinstance(search_points, _DiscretePoints):
    return functools.partial(
        _generate_discrete_point, name, search_points.feasible_points)

  min_val, max_val = search_points
  return functools.partial(
      _generate_double_point, name, min_val, max_val, 'linear')


def product(sweeps: Sequence[_SweepSequence]) -> _SweepSequence:
  """Cartesian product of a list of hyperparameter generators."""
  # A List[Dict] of hyperparameter names to sweep values.
  hyperparameter_sweep = []
  for hyperparameter_index in range(len(sweeps)):
    hyperparameter_sweep.append([])
    # Keep iterating until the iterator in sweep() ends.
    sweep_i = sweeps[hyperparameter_index]
    for point_index in range(len(sweep_i)):
      hyperparameter_name, value = list(sweep_i[point_index].items())[0]
      hyperparameter_sweep[-1].append((hyperparameter_name, value))
  return list(map(dict, itertools.product(*hyperparameter_sweep)))


def sweep(name, feasible_points: _DiscretePoints) -> _SweepSequence:
  return [{name: x} for x in feasible_points.feasible_points]


def zipit(
    generator_fns_or_sweeps: Sequence[Union[_GeneratorFn, _SweepSequence]],
    length: int) -> _SweepSequence:
  """Zip together a list of hyperparameter generators.

  Args:
    generator_fns_or_sweeps: a sequence of either:
      -generator functions that accept a Halton sequence point and return a
      quasi-ranom sample, such as those returned by halton.uniform() or
      halton.loguniform()
      -lists of dicts with one key/value such as those returned by
      halton.sweep()
      We need to support both of these (instead of having halton.sweep() return
      a list of generator functions) so that halton.sweep() can be used directly
      as a list.
    length: the number of hyperparameter points to generate. If any of the
      elements in generator_fns_or_sweeps are sweep lists, and their length is
      less than `length`, the sweep generation will be terminated and will be
      the same length as the shortest sweep sequence.

  Returns:
    A list of dictionaries, one for each trial, with a key for each unique
    hyperparameter name from generator_fns_or_sweeps.
  """
  halton_sequence = generate_sequence(
      num_samples=length,
      num_dims=len(generator_fns_or_sweeps))
  # A List[Dict] of hyperparameter names to sweep values.
  hyperparameter_sweep = []
  for trial_index in range(length):
    hyperparameter_sweep.append({})
    for hyperparameter_index in range(len(generator_fns_or_sweeps)):
      halton_point = halton_sequence[trial_index][hyperparameter_index]
      if callable(generator_fns_or_sweeps[hyperparameter_index]):
        generator_fn = generator_fns_or_sweeps[hyperparameter_index]
        hyperparameter_name, value = generator_fn(halton_point)
      else:
        sweep_list = generator_fns_or_sweeps[hyperparameter_index]
        if trial_index > len(sweep_list):
          break
        hyperparameter_point = sweep_list[trial_index]
        hyperparameter_name, value = list(hyperparameter_point.items())[0]
      hyperparameter_sweep[trial_index][hyperparameter_name] = value
  return hyperparameter_sweep

