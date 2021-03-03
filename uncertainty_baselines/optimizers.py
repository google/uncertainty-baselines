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

# Lint as: python3
"""Utilities related to optimizers for Uncertainty Baselines."""

import functools

from typing import Any, Dict, Optional, Union
from absl import logging
import tensorflow as tf
import tensorflow_addons as tfa

LearningRateSchedule = tf.keras.optimizers.schedules.LearningRateSchedule


def _maybe_get_items(d, keys):
  return {k: d[k] for k in keys if k in d}


def get(
    optimizer_name: str,
    learning_rate: float,
    weight_decay: Optional[float] = None,
    learning_rate_schedule: Union[None, str, LearningRateSchedule] = None,
    steps_per_epoch: Optional[int] = None,
    model: tf.keras.Model = None,
    **passed_optimizer_kwargs: Dict[str, Any]) -> tf.keras.optimizers.Optimizer:
  """Builds a tf.keras.optimizers.Optimizer.

  Args:
    optimizer_name: the name of the optimizer to use.
    learning_rate: the base learning rate to use in a possible the learning rate
      schedule.
    weight_decay: an optional weight decay coefficient, applied using decoupled
      weight decay as per
      https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/extend_with_decoupled_weight_decay.
    learning_rate_schedule: if None, then a constant learning rate is used. Else
      a string name can be passed and a schedule created via
      get_learning_rate_schedule(). Additionally, a custom
      tf.keras.optimizers.schedules.LearningRateSchedule can be passed.
    steps_per_epoch: the number of steps per one epoch of training data.
    model: the Keras model being optimized.
    **passed_optimizer_kwargs: additional kwargs passed to the optimizer.

  Returns:
    A Keras optimizer.
  """
  if isinstance(learning_rate_schedule, str):
    learning_rate = get_learning_rate_schedule(
        schedule_name=learning_rate_schedule,
        base_learning_rate=learning_rate,
        steps_per_epoch=steps_per_epoch,
        **passed_optimizer_kwargs)
  optimizer_kwargs = {'learning_rate': learning_rate}

  optimizer_name = optimizer_name.lower()
  if optimizer_name == 'adam':
    optimizer_class = tf.keras.optimizers.Adam
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'beta_1', 'beta_2', 'epsilon', 'amsgrad']))
  elif optimizer_name == 'nadam':
    optimizer_class = tf.keras.optimizers.Nadam
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'beta_1', 'beta_2', 'epsilon']))
  elif optimizer_name == 'rmsprop':
    optimizer_class = tf.keras.optimizers.RMSprop
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs,
            ['learning_rate', 'rho', 'momentum', 'epsilon', 'centered']))
  elif optimizer_name in ['momentum', 'nesterov']:
    optimizer_kwargs.update(
        _maybe_get_items(
            passed_optimizer_kwargs, ['learning_rate', 'momentum', 'nesterov']))
    if optimizer_name == 'nesterov':
      optimizer_kwargs['nesterov'] = True
    optimizer_class = tf.keras.optimizers.SGD
  else:
    raise ValueError('Unrecognized optimizer name: {}'.format(optimizer_name))

  if weight_decay is not None and weight_decay > 0.0:
    optimizer_class = tfa.optimizers.extend_with_decoupled_weight_decay(
        optimizer_class)
    optimizer_kwargs['weight_decay'] = weight_decay

  optimizer = optimizer_class(**optimizer_kwargs)
  if weight_decay is not None and weight_decay > 0.0 and model:
    decay_var_list = []
    skipped_variables = []
    for var in model.trainable_variables:
      if 'kernel' in var.name or 'bias' in var.name:
        decay_var_list.append(var)
      else:
        skipped_variables.append(var)
    logging.info(
        'Not applying weight decay to the following variables:\n%s',
        '\n'.join([var.name for var in skipped_variables]))
    optimizer.apply_gradients = functools.partial(
        optimizer.apply_gradients, decay_var_list=decay_var_list)
  return optimizer


def resnet50_learning_rate_schedule(base_learning_rate, steps_per_epoch):
  """Creates a piecewise LR schedule for the common ResNet-50 ImageNet setup."""
  boundaries = [
      int(40 * steps_per_epoch),
      int(60 * steps_per_epoch),
      int(80 * steps_per_epoch),
      int(90 * steps_per_epoch),
  ]
  values = [
      base_learning_rate,
      base_learning_rate * 1e-1,
      base_learning_rate * 1e-2,
      base_learning_rate * 1e-3,
      base_learning_rate * 5e-4,
  ]
  return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
      boundaries=boundaries,
      values=values)


class LinearWarmupLearningRateSchedule(LearningRateSchedule):
  """Warm up learning rate schedule.

  It starts with a linear warmup to the initial learning rate over
  `warmup_epochs`. This is found to be helpful for large batch size training
  (Goyal et al., 2018). The learning rate's value then uses the initial
  learning rate, and decays by a multiplier at the start of each epoch in
  `decay_epochs`. The stepwise decaying schedule follows He et al. (2015).
  """

  def __init__(self,
               base_learning_rate: float,
               steps_per_epoch: int,
               decay_ratio: float,
               decay_epochs: int,
               warmup_epochs: int):
    super(LinearWarmupLearningRateSchedule, self).__init__()
    self.steps_per_epoch = steps_per_epoch
    self.initial_learning_rate = base_learning_rate
    self.decay_ratio = decay_ratio
    self.decay_epochs = decay_epochs
    self.warmup_epochs = warmup_epochs

  def __call__(self, step: int):
    lr_epoch = tf.cast(step, tf.float32) / self.steps_per_epoch
    learning_rate = self.initial_learning_rate
    if self.warmup_epochs >= 1:
      learning_rate *= lr_epoch / self.warmup_epochs
    decay_epochs = [self.warmup_epochs] + self.decay_epochs
    for index, start_epoch in enumerate(decay_epochs):
      learning_rate = tf.where(
          lr_epoch >= start_epoch,
          self.initial_learning_rate * self.decay_ratio**index,
          learning_rate)
    return learning_rate


def get_learning_rate_schedule(
    schedule_name: str,
    base_learning_rate: float,
    steps_per_epoch: int,
    **schedule_kwargs: Dict[str, Any]) -> Union[float, LearningRateSchedule]:
  """Builds a step-based learning rate schedule."""
  schedule_name = schedule_name.lower()
  if schedule_name == 'constant':
    return base_learning_rate
  elif schedule_name == 'resnet50':
    return resnet50_learning_rate_schedule(base_learning_rate, steps_per_epoch)
  elif schedule_name == 'linear_warmup':
    warmup_hparams = ['decay_ratio', 'decay_epochs', 'warmup_epochs']
    if not all(elem in schedule_kwargs.keys() for elem in warmup_hparams):
      raise ValueError('schedule_kwargs must contain "decay_ratio", '
                       '"decay_epochs" and "warmup_epochs" hyperparameters, '
                       'but only contains ', schedule_kwargs.keys())

    decay_epochs = [int(x) for x in schedule_kwargs['decay_epochs']]
    return LinearWarmupLearningRateSchedule(
        base_learning_rate, steps_per_epoch,
        decay_ratio=schedule_kwargs['decay_ratio'],
        decay_epochs=decay_epochs,
        warmup_epochs=schedule_kwargs['warmup_epochs'])
  else:
    raise ValueError('Unrecognized schedule name: {}'.format(schedule_name))


class MovingAverage(tf.keras.optimizers.Optimizer):
  """Optimizer that computes a moving average of the variables.

  Empirically it has been found that using the moving average of the trained
  parameters of a deep network is better than using its trained parameters
  directly. This optimizer allows you to compute this moving average and swap
  the variables at save time so that any code outside of the training loop
  will use by default the average values instead of the original ones.

  Example of usage for training:
  ```python
  opt = tf.keras.optimizers.SGD(learning_rate)
  opt = MovingAverage(opt)

  opt.shadow_copy(model)
  ```

  At test time, swap the shadow variables to evaluate on the averaged weights:
  ```python
  opt.swap_weights()
  # Test eval the model here
  opt.swap_weights()
  ```
  """

  def __init__(self,
               optimizer,
               average_decay=0.99,
               start_step=0,
               dynamic_decay=True,
               name='moving_average',
               **kwargs):
    """Construct a new MovingAverage optimizer.

    Args:
      optimizer: `tf.keras.optimizers.Optimizer` that will be
        used to compute and apply gradients.
      average_decay: float. Decay to use to maintain the moving averages
        of trained variables.
      start_step: int. What step to start the moving average.
      dynamic_decay: bool. Whether to change the decay based on the number
        of optimizer updates. Decay will start at 0.1 and gradually increase
        up to `average_decay` after each optimizer update. This behavior is
        similar to `tf.train.ExponentialMovingAverage` in TF 1.x.
      name: Optional name for the operations created when applying
        gradients. Defaults to "moving_average".
      **kwargs: keyword arguments. Allowed to be {`clipnorm`,
        `clipvalue`, `lr`, `decay`}.
    """
    super(MovingAverage, self).__init__(name, **kwargs)
    self._optimizer = optimizer
    self._average_decay = average_decay
    self._start_step = tf.constant(start_step, tf.float32)
    self._dynamic_decay = dynamic_decay

  def shadow_copy(self, model):
    """Creates shadow variables for the given model weights."""
    for var in model.weights:
      self.add_slot(var, 'average', initializer='zeros')
    self._average_weights = [
        self.get_slot(var, 'average') for var in model.weights
    ]
    self._model_weights = model.weights

  @property
  def has_shadow_copy(self):
    """Whether this optimizer has created shadow variables."""
    return self._model_weights is not None

  def _create_slots(self, var_list):
    self._optimizer._create_slots(var_list=var_list)  # pylint: disable=protected-access

  def apply_gradients(self, grads_and_vars, name=None):
    result = self._optimizer.apply_gradients(grads_and_vars, name)
    self.update_average(self._optimizer.iterations)
    return result

  @tf.function
  def update_average(self, step):
    step = tf.cast(step, tf.float32)
    if step < self._start_step:
      decay = tf.constant(0., tf.float32)
    elif self._dynamic_decay:
      decay = step - self._start_step
      decay = tf.minimum(self._average_decay, (1. + decay) / (10. + decay))
    else:
      decay = self._average_decay

    def _apply_moving(v_moving, v_normal):
      diff = v_moving - v_normal
      v_moving.assign_sub(tf.cast(1. - decay, v_moving.dtype) * diff)
      return v_moving

    def _update(strategy, v_moving_and_v_normal):
      for v_moving, v_normal in v_moving_and_v_normal:
        strategy.extended.update(v_moving, _apply_moving, args=(v_normal,))

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(_update, args=(zip(self._average_weights,
                                             self._model_weights),))

  def swap_weights(self, strategy):
    """Swap the average and moving weights.

    This is a convenience method to allow one to evaluate the averaged weights
    at test time. Loads the weights stored in `self._average` into the model,
    keeping a copy of the original model weights. Swapping twice will return
    the original weights.

    Args:
      strategy: tf.distribute.Strategy to be used.
    """
    strategy.run(self._swap_weights, args=())

  def _swap_weights(self):
    def fn_0(a, b):
      a.assign_add(b)
      return a
    def fn_1(b, a):
      b.assign(a - b)
      return b
    def fn_2(a, b):
      a.assign_sub(b)
      return a

    def swap(strategy, a_and_b):
      """Swap `a` and `b` and mirror to all devices."""
      for a, b in a_and_b:
        strategy.extended.update(a, fn_0, args=(b,))  # a = a + b
        strategy.extended.update(b, fn_1, args=(a,))  # b = a - b
        strategy.extended.update(a, fn_2, args=(b,))  # a = a - b

    ctx = tf.distribute.get_replica_context()
    return ctx.merge_call(
        swap, args=(zip(self._average_weights, self._model_weights),))

  def assign_average_vars(self, var_list):
    """Assign variables in var_list with their respective averages.

    Args:
      var_list: List of model variables to be assigned to their average.
    Returns:
      assign_op: The op corresponding to the assignment operation of
        variables to their average.
    """
    assign_op = tf.group([
        var.assign(self.get_slot(var, 'average')) for var in var_list
        if var.trainable
    ])
    return assign_op

  def _create_hypers(self):
    self._optimizer._create_hypers()  # pylint: disable=protected-access

  def _prepare(self, var_list):
    return self._optimizer._prepare(var_list=var_list)  # pylint: disable=protected-access

  @property
  def iterations(self):
    return self._optimizer.iterations

  @iterations.setter
  def iterations(self, variable):
    self._optimizer.iterations = variable

  @property
  def weights(self):
    return self._optimizer.weights

  # pylint: disable=protected-access
  @property
  def lr(self):
    return self._optimizer._get_hyper('learning_rate')

  @lr.setter
  def lr(self, lr):
    self._optimizer._set_hyper('learning_rate', lr)

  @property
  def learning_rate(self):
    return self._optimizer._get_hyper('learning_rate')

  @learning_rate.setter
  def learning_rate(self, learning_rate):  # pylint: disable=redefined-outer-name
    self._optimizer._set_hyper('learning_rate', learning_rate)

  def _resource_apply_dense(self, grad, var):
    return self._optimizer._resource_apply_dense(grad, var)

  def _resource_apply_sparse(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse(grad, var, indices)

  def _resource_apply_sparse_duplicate_indices(self, grad, var, indices):
    return self._optimizer._resource_apply_sparse_duplicate_indices(
        grad, var, indices)
  # pylint: enable=protected-access

  def get_config(self):
    config = {
        'optimizer': tf.keras.optimizers.serialize(self._optimizer),
        'average_decay': self._average_decay,
        'start_step': self._start_step,
        'dynamic_decay': self._dynamic_decay,
    }
    base_config = super(MovingAverage, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config, custom_objects=None):
    optimizer = tf.keras.optimizers.deserialize(
        config.pop('optimizer'),
        custom_objects=custom_objects,
    )
    return cls(optimizer, **config)
