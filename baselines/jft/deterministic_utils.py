# coding=utf-8
# Copyright 2023 The Uncertainty Baselines Authors.
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
import logging

import flax
import jax
import jax.numpy as jnp
import input_utils  # local file import from baselines.jft
import train_utils  # local file import from baselines.jft


def get_total_steps(config):
  """Get total_steps of training.

  Args:
    config: The config of the experiment.

  Returns:
    Total_steps of training.
  """
  local_batch_size = config.batch_size // jax.process_count()
  ntrain_img = input_utils.get_num_examples(
      config.dataset,
      split=config.train_split,
      process_batch_size=local_batch_size,
      data_dir=config.get('data_dir'))
  steps_per_epoch = ntrain_img // config.batch_size

  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get('total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = config.total_steps

  logging.info('Total train data points: %d', ntrain_img)
  logging.info(
      'Running for %d steps, that means %f epochs and %d steps per epoch',
      total_steps, total_steps * config.batch_size / ntrain_img,
      steps_per_epoch)
  return total_steps


def create_init(model, config, train_ds):
  """Create the initialization function for model parameters.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.
    train_ds: tf.data.Dataset.

  Returns:
    Function that returns initialized model parameters.
  """
  local_batch_size = config.batch_size // jax.process_count()
  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @functools.partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    params = flax.core.unfreeze(model.init(rng, dummy_input,
                                           train=False))['params']

    # Set bias in the head to a low value, such that loss is small initially.
    params['head']['bias'] = jnp.full_like(params['head']['bias'],
                                           config.get('init_head_bias', 0))

    # init head kernel to all zeros for fine-tuning
    if config.get('model_init'):
      params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

    return params

  return init


def create_update_fn(model, config):
  """Create the update function from model and config.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that updates the model for one step.
  """
  weight_decay_rules = config.get('weight_decay', []) or []
  rescale_value = 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)
  logging.info('weight_decay_rules = %s', weight_decay_rules)
  logging.info('rescale_value = %s', rescale_value)

  @functools.partial(jax.pmap, axis_name='batch', donate_argnums=(0))
  def update_fn(opt, lr, images, labels, rng):
    """Update step."""

    measurements = {}

    # Split rng and return next_rng for the following step.
    rng, next_rng = jax.random.split(rng, 2)
    rng_local = jax.random.fold_in(rng, jax.lax.axis_index('batch'))
    logging.info(msg=f'images in loss_fn = {jnp.shape(images)}')
    logging.info(msg=f'labels in loss_fn = {jnp.shape(labels)}')
    def loss_fn(params, images, labels):
      logits, _ = model.apply({'params': flax.core.freeze(params)},
                              images,
                              train=True,
                              rngs={'dropout': rng_local})
      logging.info(msg=f'logits={logits}')
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
    logging.info(msg=f'measurements = {measurements}')

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get('grad_accum_steps', 1) == 1 or config.get('grad_clip_norm'):
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g
    logging.info(msg=f'measurements = {measurements}')

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


def create_evaluation_fn(model, config):
  """Create the evaluation function from model and config.

  Args:
    model: The model to be used in updates.
    config: The config of the experiment.

  Returns:
    The function that evaluates the model for one step.
  """
  @functools.partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels, mask):
    # Ignore the entries with all zero labels for evaluation.
    mask *= labels.max(axis=1)
    logits, out = model.apply({'params': flax.core.freeze(params)},
                              images,
                              train=False)
    label_indices = config.get('label_indices')
    logging.info('!!! mask %s, label_indices %s', mask, label_indices)
    if label_indices:
      logits = logits[:, label_indices]

    # Note that logits and labels are usually of the shape [batch,num_classes].
    # But for OOD data, when num_classes_ood > num_classes_ind, we need to
    # adjust labels to labels[:, :config.num_classes] to match the shape of
    # logits. That is just to avoid shape mismatch. The output losses does not
    # have any meaning for OOD data, because OOD not belong to any IND class.
    losses = getattr(train_utils, config.get('loss', 'sigmoid_xent'))(
        logits=logits,
        labels=labels[:, :(
            len(label_indices) if label_indices else config.num_classes)],
        reduction=False)
    loss = jax.lax.psum(losses * mask, axis_name='batch')

    top1_idx = jnp.argmax(logits, axis=1)
    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]
    ncorrect = jax.lax.psum(top1_correct * mask, axis_name='batch')
    n = jax.lax.psum(mask, axis_name='batch')

    metric_args = jax.lax.all_gather([logits, labels, out['pre_logits'], mask],
                                     axis_name='batch')
    return ncorrect, loss, n, metric_args

  return evaluation_fn
