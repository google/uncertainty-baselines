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

"""Utils for deterministic ViT."""
import functools
import flax
import jax
import jax.numpy as jnp
import train_utils  # local file import from baselines.jft


def make_update_fn(model, config):
  """Make the update function.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.
  Returns:
    The function that updates the model for one step.
  """
  weight_decay_rules = config.get('weight_decay', []) or []
  rescale_value = config.lr.base if config.get('weight_decay_decouple') else 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)
  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, lr, images, labels, rng):
    """Update step."""

    measurements = {}

    # Split rng and return next_rng for the following step.
    rng, next_rng = jax.random.split(rng, 2)
    rng_local = jax.random.fold_in(rng, jax.lax.axis_index('batch'))

    def loss_fn(params, images, labels):
      logits, _ = model.apply(
          {'params': flax.core.freeze(params)}, images,
          train=True, rngs={'dropout': rng_local})
      label_indices = config.get('label_indices')
      if label_indices:
        logits = logits[:, label_indices]
      loss = getattr(train_utils, config.get('loss', 'sigmoid_xent'))(
          logits=logits, labels=labels)
      return loss, logits

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    (l, logits), g = train_utils.accumulate_gradient(
        jax.value_and_grad(loss_fn, has_aux=True), opt.target, images, labels,
        config.get('grad_accum_steps'))
    l, g = jax.lax.pmean((l, g), axis_name='batch')
    measurements['training_loss'] = l

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get('grad_accum_steps', 1) == 1 or config.get('grad_clip_norm'):
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if config.get('grad_clip_norm'):
      g_factor = jnp.minimum(1.0, config.grad_clip_norm / l2_g)
      g = jax.tree_util.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)

    opt = opt.replace(target=weight_decay_fn(opt.target, lr))

    params, _ = jax.tree_flatten(opt.target)
    measurements['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))

    top1_idx = jnp.argmax(logits, axis=1)
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    prec1 = jax.lax.psum(
        jnp.sum(top1_correct), axis_name='batch') / config.batch_size
    measurements['training_prec@1'] = prec1
    measurements['learning_rate'] = lr
    return opt, next_rng, measurements

  return update_fn
