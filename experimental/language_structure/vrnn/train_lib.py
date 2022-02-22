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

"""Helper function for training."""

import functools
from typing import Any, Dict, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

BIAS_CONSTANT = 100.0


def create_run_steps_fn(
    step_fn: tf.types.experimental.GenericFunction,
    strategy: tf.distribute.Strategy,
    distributed: bool,
    output_dtypes: Optional[Sequence[tf.dtypes.DType]] = None,
    **step_fn_kwargs):
  """Creates a tf.function running `step_fn` using the distributed strategy.

  Usage example:

    @tf.function
    def inference_step(inputs: Sequence[tf.Tensor]):
      predictions = model(inputs)
      return predictions

    run_inference_steps = create_run_steps_fn(
      inference_step,
      strategy,
      distributed=True,
      output_dtypes=[tf.int32])

    predictions = run_inference_steps(iter(dataset), num_steps=tf.constant(10))

  Args:
      step_fn: the tf.function executing a step of training/evaluation/inference
        operation. It's of interface step_fn(inputs: Sequence[tf.Tensor]).
      strategy: the distrubted strategy.
      distributed: whether the input dataset is a distributed dataset.
      output_dtypes: the output types of step_fn.
      **step_fn_kwargs: the optional arguments of step_fn.

  Returns:
      run_steps_fn: a tf.function running step_fn for the given number of steps
        by feeding the inputs from the dataset iterator, and returning the
        concatenated outputs of it.
  """

  if output_dtypes is None:
    output_dtypes = []

  @tf.function
  def _run_steps(iterator: Any, num_steps: tf.Tensor) -> Sequence[tf.Tensor]:
    outputs = [
        tf.TensorArray(dtype=dtype, size=num_steps) for dtype in output_dtypes
    ]

    for step_id in tf.range(num_steps):
      per_replica_results = strategy.run(
          step_fn, args=(next(iterator),), kwargs=step_fn_kwargs)
      if per_replica_results is None:
        continue

      if isinstance(per_replica_results, tf.distribute.DistributedValues):
        per_replica_results = [per_replica_results]

      if len(per_replica_results) != len(outputs):
        raise ValueError(
            "Expected step_fn results to be of length {}, found {}".format(
                len(outputs), len(per_replica_results)))

      temp_outputs = []
      for replica_id, per_replica_result in enumerate(per_replica_results):
        if distributed:
          reduced_result = strategy.gather(per_replica_result, axis=0)
        else:
          reduced_result = per_replica_result.values[0]
        # TODO(yquan): Investigate why updating outputs[replica_id] doesn't work
        temp_outputs.append(outputs[replica_id].write(step_id, reduced_result))
      outputs = temp_outputs

    outputs = [output.concat() for output in outputs]

    return outputs

  return _run_steps


def build_hidden_state_model(input_size: int, output_size: int,
                             learning_rate: float) -> tf.keras.Model:
  """Builds the simple linear classifer for hidden representation learning."""
  input_layer = tf.keras.layers.Input(input_size)
  output = tf.keras.layers.Dense(output_size, activation="softmax")(input_layer)

  model = tf.keras.Model(input_layer, output)

  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
      loss="sparse_categorical_crossentropy",
      metrics=["sparse_categorical_accuracy"],
      weighted_metrics=["sparse_categorical_accuracy"])

  return model


class FewShotSamplePool(object):
  """Class generating labeled samples for few shots.

  Attributes:
    num_classes: Number of classes of samples in the pool.
  """

  def __init__(self, seed):
    self._seed = seed
    self._rng = np.random.default_rng(seed=self._seed)

    self._features = None
    self._labels = None
    self._class_indices = []

  def refresh(self, features: tf.Tensor, labels: tf.Tensor):
    """Refreshes sample pool."""
    labels = labels.numpy()
    unique_class_ids = np.unique(labels)
    indices = np.arange(labels.size)

    self._class_indices = []
    for class_id in unique_class_ids:
      self._class_indices.append(indices[labels == class_id])

    self._features = features.numpy()
    self._labels = labels

  @property
  def num_classes(self):
    return len(self._class_indices)

  def sample(self, shots: int):
    """Generate samples based of the given number of shots per class."""
    self.batch_sample([shots])

  def batch_sample(self, shots_list: Sequence[int]):
    """Generate samples for each of the number of shots in `shots_list`."""
    # Generate samples with most shots first.
    shots_list = sorted(shots_list, reverse=True)

    all_class_indices = self._class_indices
    for shots in shots_list:
      sample_class_indices = []
      for class_indices in all_class_indices:
        num_samples = min(len(class_indices), shots)
        sample_class_indices.append(
            self._rng.choice(class_indices, num_samples, replace=False))

      # Sample fewer shots from the samples of more shots.
      all_class_indices = sample_class_indices

      indices = np.concatenate(sample_class_indices)
      yield self._features[indices], self._labels[indices], shots




