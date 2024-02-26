# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""JAX (Batch)Ensemble training related functions."""

import jax
import jax.numpy as jnp


# TODO(dusenberrymw,zmariet): Clean up and generalize these log marginal probs.
def log_average_softmax_probs(logits: jnp.ndarray) -> jnp.ndarray:
  # TODO(zmariet): dedicated eval loss function.
  ens_size = logits.shape[0]
  log_p = jax.nn.log_softmax(
      logits, axis=-1)  # (ensemble_size, batch_size, num_classes)
  log_p = jax.nn.logsumexp(log_p, axis=0) - jnp.log(ens_size)
  return log_p
