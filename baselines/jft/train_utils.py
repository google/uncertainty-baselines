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

"""Training utilities for the ViT experiments.

Several functions in this file were ported from
https://github.com/google-research/vision_transformer.
"""

import multiprocessing
import numbers
import re
import time

from typing import Callable, List, Tuple, Union

from absl import logging
import flax
import jax
import jax.numpy as jnp
import numpy as np

import checkpoint_utils  # local file import from baselines.jft

# TODO(zmariet, dusenberrymw): create separate typing module.
Params = checkpoint_utils.Params


def sigmoid_xent(*, logits, labels, reduction=True):
  """Computes a sigmoid cross-entropy (Bernoulli NLL) loss over examples."""
  log_p = jax.nn.log_sigmoid(logits)
  log_not_p = jax.nn.log_sigmoid(-logits)
  nll = -jnp.sum(labels * log_p + (1. - labels) * log_not_p, axis=-1)
  return jnp.mean(nll) if reduction else nll


def softmax_xent(*, logits, labels, reduction=True, kl=False):
  """Computes a softmax cross-entropy (Categorical NLL) loss over examples."""
  log_p = jax.nn.log_softmax(logits)
  nll = -jnp.sum(labels * log_p, axis=-1)
  if kl:
    nll += jnp.sum(labels * jnp.log(jnp.clip(labels, 1e-8)), axis=-1)
  return jnp.mean(nll) if reduction else nll


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
  """Accumulates gradients over multiple steps to save on memory."""
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}")
    step_size = images.shape[0] // accum_steps
    l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

    def acc_grad_and_loss(i, l_and_g):
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      li, gi = loss_and_grad_fn(params, imgs, lbls)
      l, g = l_and_g
      return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
    return jax.tree_util.tree_map(lambda x: x / accum_steps, (l, g))
  else:
    return loss_and_grad_fn(params, images, labels)


def create_learning_rate_schedule(total_steps,
                                  base=0.,
                                  decay_type="linear",
                                  warmup_steps=0,
                                  linear_end=1e-5):
  """Creates a learning rate schedule.

  Currently only warmup + {linear,cosine} but will be a proper mini-language
  like preprocessing one in the future.

  Args:
    total_steps: The total number of steps to run.
    base: The starting learning-rate (without warmup).
    decay_type: 'linear' or 'cosine'.
    warmup_steps: how many steps to warm up for.
    linear_end: Minimum learning rate.

  Returns:
    A function learning_rate(step): float -> {"learning_rate": float}.
  """

  def step_fn(step):
    """Step to learning rate function."""
    lr = base

    progress = (step - warmup_steps) / float(total_steps - warmup_steps)
    progress = jnp.clip(progress, 0.0, 1.0)
    if decay_type == "linear":
      lr = linear_end + (lr - linear_end) * (1.0 - progress)
    elif decay_type == "cosine":
      lr = lr * 0.5 * (1. + jnp.cos(jnp.pi * progress))
    else:
      raise ValueError(f"Unknown lr type {decay_type}")

    if warmup_steps:
      lr = lr * jnp.minimum(1., step / warmup_steps)

    return jnp.asarray(lr, dtype=jnp.float32)

  return step_fn


def get_weight_decay_fn(
    weight_decay_rules: Union[float, List[Tuple[str, float]]],
    rescale_value: float) -> Callable[[Params, float], Params]:
  """Returns a custom weight-decay function for the learning rate.

  Args:
    weight_decay_rules: either a scalar indicating the strength of the weight
      decay, or a list of tuples of the form (parameter_regex, weight_decay)
      mapping specific variables to the respective weight decay strength. For
      example, `[('.*kernel.*', 0.5), ('.*fast_weight.*', 0.1)]` will decay the
      BatchEnsemble slow and fast weights at different rates.
    rescale_value: scalar indicating by how much the initial learning rate needs
      to be scaled before applying the weight decay.

  Returns:
    A function mapping a pytree of parameters and a learning rate to an updated
      pytree of the same structure with weight decayed values.
  """
  if isinstance(weight_decay_rules, numbers.Number):
    # Append weight decay factor to variable name patterns it applies to.
    weight_decay_rules = [(".*kernel.*", weight_decay_rules)]

  def weight_decay_fn(params, lr):
    return tree_map_with_regex(
        lambda params, wd: (1.0 - lr / rescale_value * wd) * params,
        params, weight_decay_rules)

  return weight_decay_fn


def tree_map_with_regex(f, tree, regex_rules):
  """Performs a JAX-style tree_map with filtering based on regex rules.

  Args:
    f: A function that is being applied to every variable.
    tree: A JAX PyTree of arrays.
    regex_rules: A list of tuples `(pattern, args)`, where `pattern` is a regex
      which used for variable matching and `args` are positional arguments
      passed to `f`. If some variable is not matched, we apply `not_f` transform
      which is id by default.

  Returns:
    A tree, transformed by `f` according to the given rules.
  """

  def _f(path_tuple, v):
    vname = "/".join(path_tuple)
    for pattern, arg in regex_rules:
      if re.match(pattern, vname):
        if jax.process_index() == 0:
          logging.info("Updating %s with %s due to `%s`", vname, arg, pattern)
        return f(v, arg)
    return v

  flat_tree = flax.traverse_util.flatten_dict(tree)
  updated_flat_tree = {k: _f(k, v) for k, v in flat_tree.items()}
  updated_tree = flax.traverse_util.unflatten_dict(updated_flat_tree)
  return updated_tree


def prefetch_scalar(it, nprefetch=1, devices=None):
  """Prefetches a scalar value onto an accelerator device."""
  n_loc_dev = len(devices) if devices else jax.local_device_count()
  repl_iter = (np.ones(n_loc_dev) * i for i in it)
  return flax.jax_utils.prefetch_to_device(repl_iter, nprefetch, devices)


def checkpointing_timeout(writer, timeout):
  """Checks that checkpointing is not a bottleneck."""
  # Make sure checkpoint writing is not a bottleneck
  if writer is not None:
    try:
      writer.get(timeout=timeout)
    except multiprocessing.TimeoutError:
      raise TimeoutError(
          "Checkpoint writing seems to be a bottleneck. Make sure you do "
          "not do something wrong, like writing checkpoints to a distant "
          "cell. In a case you are OK with checkpoint writing being a "
          "bottleneck, you can configure `checkpoint_timeout` parameter")


def itstime(step,
            every_n_steps,
            total_steps,
            process=None,
            last=True,
            first=True):
  """Determines whether or not it is time to trigger an action."""
  is_process = process is None or jax.process_index() == process
  is_step = every_n_steps and (step % every_n_steps == 0)
  is_last = every_n_steps and step == total_steps
  is_first = every_n_steps and step == 1
  return is_process and (is_step or (last and is_last) or (first and is_first))


class Chrono:
  """Measures time and reports progress."""

  def __init__(self, first_step, total_steps, global_bs, accum_train_time=0):
    self.first_step = first_step
    self.total_steps = total_steps
    self.global_bs = global_bs
    self.accum_train_time = accum_train_time
    self.start_time = None
    self.prev_time = None
    self.prev_step = None
    self.pause_start = None
    self.paused_time = 0
    self.warmup = 1  # How many calls to `tick` to skip.
    self.note = f"Steps:{first_step}/{self.total_steps} "
    self.note += f"[{self.first_step/self.total_steps:.1%}]"
    self.note += "\nETA:n/a"
    self.note += "\nTotal time:n/a"

  def tick(self, step):
    """Performs one chronometer tick."""
    now = time.time()

    # We take the start as the second time `tick` is called, so we avoid
    # measuring the overhead of compilation and don't include it in time
    # estimates.
    if self.warmup:
      self.warmup -= 1
      return {}, self.note
    if None in (self.start_time, self.prev_time, self.prev_step):
      self.start_time = self.prev_time = now
      self.prev_step = step
      return {}, self.note

    def hms(s):
      """Formats time in hours/minutes/seconds."""
      if s < 60:
        return f"{s:.0f}s"
      m, s = divmod(s, 60)
      if m < 60:
        return f"{m:.0f}m{s:.0f}s"
      h, m = divmod(m, 60)
      return f"{h:.0f}h{m:.0f}m"  # Seconds intentionally omitted.

    # Progress note with "global" full-program average timings
    dt = now - self.start_time  # Time since process start.
    steps_done = step - self.first_step
    steps_todo = self.total_steps - step
    self.note = f"Steps:{step}/{self.total_steps} [{step/self.total_steps:.1%}]"
    self.note += f"\nETA:{hms(dt / steps_done * steps_todo)}"
    self.note += f"\nTotal time:{hms(dt / steps_done * self.total_steps)}"

    timing_measurements = {}

    # Measurement with micro-timings of current training steps speed.
    dt = now - self.prev_time - self.paused_time  # Time between ticks.
    ds = step - self.prev_step  # Steps between ticks.
    ncores = jax.device_count()  # Global device count.
    timing_measurements["img/sec/core"] = self.global_bs * ds / dt / ncores

    # Accumulate (integrate) training time, good for plots.
    self.accum_train_time += dt
    core_hours = self.accum_train_time * ncores / 60 / 60
    devtype = jax.devices()[0].device_kind
    timing_measurements[f"core_hours_{devtype}"] = core_hours

    self.prev_time = now
    self.prev_step = step
    self.paused_time = 0

    return timing_measurements, self.note

  def pause(self):
    """Pauses the time measurement."""
    assert self.pause_start is None, "Don't pause twice."
    self.pause_start = time.time()

  def resume(self):
    """Resumes the time measurement."""
    self.paused_time += time.time() - self.pause_start
    self.pause_start = None
