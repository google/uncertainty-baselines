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

"""TODO(dieterichl): DO NOT SUBMIT without one-line documentation for svgd.

TODO(dieterichl): DO NOT SUBMIT without a detailed description of svgd.
"""
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def pairwise_sos(x):
  """Computes all-pairs sum of squares distance."""
  return tf.reduce_sum((x[:, None] - x[None, :])**2, axis=-1)


def batch_pairwise_sos(x):
  """Computes all-pairs sum of squares distance.

  Args:
    x: A [batch_size, num_members, data_dim] tensor

  Returns:
    pairwise_sos: A [batch_size, num_members, num_members] tensor.
  """
  return tf.reduce_sum((x[:, :, None] - x[:, None, :])**2, axis=-1)


def pairwise_rbf(x, bandwidth=1.):
  """Computes an all-pairs radial basis function kernel."""
  return tf.math.exp((-0.5 / bandwidth) * pairwise_sos(x))


def collapse_vars(variables):
  """Collapses lists of variables into a single tensor.

  Accepts a list containing the variables for each
  member of an ensemble. Those variables are flattened and concatenated
  so that the resulting tensor is [num_members, num_parameters].

  Args:
    variables: The variables of all ensemble members
  Returns:
    A single [num_members, num_parameters] tensor containing all variables.
  """
  flat_vars = []
  num_ensemble_members = len(variables[0])
  for vs in variables:
    flat_v = tf.reshape(tf.stack(vs, axis=0), [num_ensemble_members, -1])
    flat_vars.append(flat_v)

  flat_vars = tf.concat(flat_vars, axis=1)
  assert flat_vars.shape[0] == num_ensemble_members
  return flat_vars


def rbf_weight_kernel(variables, bandwidth=1.):
  return pairwise_rbf(collapse_vars(variables), bandwidth=bandwidth)


def rbf_function_kernel(logits, bandwidth=1.):
  # logits is [num_members, batch_size, num_classes]
  # normalize the logits
  logits = tf.math.log_softmax(logits, axis=2)
  # make logits [batch_size, num_members, num_classes] for the SOS computation.
  logits = tf.transpose(logits, perm=[1, 0, 2])
  # [batch_size, num_members, num_members]
  sos = batch_pairwise_sos(logits)
  return tf.math.exp(-(0.5 / bandwidth) * tf.reduce_mean(sos, axis=0))


def kern_val_and_grad(variables, bandwidth=1.):
  """Compute the values and gradients of an RBF kernel.

  Args:
    variables: A list the variables of each ensemble member.
    bandwidth: A float, the bandwidth of the kernel.

  Returns:
    k_vals: A [num_members, num_members] tensor containing the kernel values
    grads: A list of length num_in_vars. The ith element is a
      [num_members, num_members, ...] tensor representing the gradient of each
      kernel value w.r.t. the ith variable. in_grads[i][j,k] is the
      gradient of k(x_j, x_k) w.r.t. the ith variable of x_j.
  """
  with tf.GradientTape(persistent=True) as g:
    # [num_members, num_members]
    k_vals = rbf_weight_kernel(variables, bandwidth=bandwidth)
    # list of length [num_members] each element is tensor of shape [num_members]
    # [k(w_1, other members), k(w_2, other_members), ...]
    k_v_l = tf.unstack(k_vals, axis=0)

  grads = []
  for v_l in variables:
    jacs = [g.jacobian(k_i, head_i_var) for k_i, head_i_var in zip(k_v_l, v_l)]
    jacs = tf.stack(jacs, axis=0)
    grads.append(jacs)

  return k_vals, grads


def categorical_log_prob(model, inputs, labels, training=True):
  # [num_members, batch_size, num_classes]
  logits = model(inputs, training=training)
  # [num_members, batch_size]
  data_ll = tfd.Categorical(logits=logits).log_prob(labels[tf.newaxis, :])
  return logits, tf.reduce_mean(data_ll, axis=1)


def gaussian_log_prob(model, inputs, labels, variance=1., training=True):
  # [num_members, batch_size, 1]
  pred_means = model(inputs, training=training)
  assert pred_means.shape[2] == 1
  pred_means = tf.squeeze(pred_means, axis=2)
  pred_dist = tfd.Normal(loc=pred_means, scale=tf.sqrt(variance))
  data_ll = pred_dist.log_prob(labels[tf.newaxis, :])
  return tf.reduce_mean(data_ll, axis=1)


def unnormalized_log_gaussian_prior(model, variance):
  """Computes the weight-space isotropic gaussian prior."""
  in_vars, trunk_vars, out_vars = model.all_vars()

  num_members = len(in_vars[0])
  p_vals = [tf.constant(0.) for _ in range(num_members)]
  for member_vs in in_vars + out_vars:
    for i, v in enumerate(member_vs):
      p_vals[i] += tf.reduce_sum(tf.square(v))

  p_vals = tf.stack(p_vals)
  for var in trunk_vars:
    p_vals += tf.reduce_sum(tf.square(var)) * num_members

  return p_vals * - (1. / (2. * variance))


def neg_svgd_grad(
    model, log_joint_fn, images, labels,
    temperature=1., kern_bandwidth=1., num_devices=1.):
  """Compute the SVGD gradient update."""
  in_vars, trunk_vars, out_vars = model.all_vars()
  num_members = model.num_members
  num_in_vars = len(in_vars)
  num_trunk_vars = len(trunk_vars)
  num_out_vars = len(out_vars)

  # Compute kernel vals and grads
  k_vals, k_grads = kern_val_and_grad(
      in_vars + out_vars, bandwidth=kern_bandwidth)

  # Compute grad of log prob
  with tf.GradientTape(persistent=True) as g:
    logits, likelihoods, log_probs = log_joint_fn(
        model, images, labels, training=True)
    log_probs /= num_devices
    log_probs = tf.unstack(log_probs)

  # Compute grad of each log prob w.r.t. each set of variables.
  grads = [g.gradient(log_probs[i], model.member_vars(i))
           for i in range(num_members)]

  # grads is currently [num_members, 3, num_vars, ...]
  # Rearrange so it is [num_vars, num_members, ...]
  in_grads = [tf.stack([g[0][i] for g in grads]) for i in range(num_in_vars)]
  trunk_grads = [
      tf.stack([g[1][i] for g in grads]) for i in range(num_trunk_vars)
  ]
  out_grads = [tf.stack([g[2][i] for g in grads]) for i in range(num_out_vars)]

  all_grads = in_grads + trunk_grads + out_grads

  # Weight by kernel values.
  ll_grads = []
  for g in all_grads:
    weighted_grad = tf.einsum('ij,j...->i...', k_vals, g)
    avg_grad = weighted_grad / (num_members * temperature)
    ll_grads.append(avg_grad)

  in_ll_grads = ll_grads[:num_in_vars]
  in_k_grads = k_grads[:num_in_vars]
  trunk_ll_grads = ll_grads[num_in_vars:num_in_vars + num_trunk_vars]
  out_ll_grads = ll_grads[num_in_vars + num_trunk_vars:]
  out_k_grads = k_grads[num_in_vars:]

  final_in_grads = []
  for ll_grad, k_grad in zip(in_ll_grads, in_k_grads):
    final_in_grads.append(
        tf.unstack(-(ll_grad + tf.reduce_mean(k_grad, axis=0))))

  final_out_grads = []
  for ll_grad, k_grad in zip(out_ll_grads, out_k_grads):
    final_out_grads.append(
        tf.unstack(-(ll_grad + tf.reduce_mean(k_grad, axis=0))))

  final_trunk_grads = [-tf.reduce_mean(g, axis=0) for g in trunk_ll_grads]

  return logits, likelihoods, log_probs, (final_in_grads, final_trunk_grads,
                                          final_out_grads)


def neg_fsvgd_grad(model,
                   log_joint_fn,
                   w_kernel_fn,
                   f_kernel_fn,
                   images,
                   labels,
                   temperature=1.,
                   num_devices=1.):
  """Compute the functional SVGD gradient update."""
  in_vars, trunk_vars, out_vars = model.all_vars()
  num_members = model.num_members
  num_in_vars = len(in_vars)
  num_trunk_vars = len(trunk_vars)
  num_out_vars = len(out_vars)

  # Compute kernel vals and grads
  w_k_vals = w_kernel_fn(in_vars + out_vars)

  # Compute grad of log prob
  with tf.GradientTape(persistent=True) as g:
    # logits is [num_members, batch_size, num_classes]
    logits, likelihoods, log_probs = log_joint_fn(
        model, images, labels, training=True)
    log_probs /= num_devices
    log_probs = tf.unstack(log_probs)
    f_kernel_vals = f_kernel_fn(logits)
    # list of length num_members, each element is tensor of shape [num_members]
    # [k(w_1, other members), k(w_2, other members), ....]
    f_kernel_vals_list = tf.unstack(f_kernel_vals, axis=0)

  # Compute grad of each log prob w.r.t. each set of variables.
  grads = [g.gradient(log_probs[i], model.member_vars(i))
           for i in range(num_members)]

  f_k_grads = []
  # for each variable in the in and out heads
  for var_list in in_vars + out_vars:
    jacs = []
    # compute the gradient of the kernel values for each ensemble member
    for k_vals_i, head_i_var in zip(f_kernel_vals_list, var_list):
      jacs.append(g.jacobian(k_vals_i, head_i_var))
    jacs = tf.stack(jacs, axis=0)
    f_k_grads.append(jacs)

  # grads is currently [num_members, 3, num_vars, ...]
  # Rearrange so it is [num_vars, num_members, ...]
  in_grads = [tf.stack([g[0][i] for g in grads]) for i in range(num_in_vars)]
  trunk_grads = [
      tf.stack([g[1][i] for g in grads]) for i in range(num_trunk_vars)
  ]
  out_grads = [tf.stack([g[2][i] for g in grads]) for i in range(num_out_vars)]

  all_grads = in_grads + trunk_grads + out_grads

  # Weight by kernel values.
  ll_grads = []
  for g in all_grads:
    weighted_grad = tf.einsum('ij,j...->i...', w_k_vals, g)
    avg_grad = weighted_grad / (num_members * temperature)
    ll_grads.append(avg_grad)

  in_ll_grads = ll_grads[:num_in_vars]
  in_f_k_grads = f_k_grads[:num_in_vars]
  trunk_ll_grads = ll_grads[num_in_vars:num_in_vars + num_trunk_vars]
  out_ll_grads = ll_grads[num_in_vars + num_trunk_vars:]
  out_f_k_grads = f_k_grads[num_in_vars:]

  final_in_grads = []
  for ll_grad, k_grad in zip(in_ll_grads, in_f_k_grads):
    final_in_grads.append(
        tf.unstack(-(ll_grad + tf.reduce_mean(k_grad, axis=0))))

  final_out_grads = []
  for ll_grad, k_grad in zip(out_ll_grads, out_f_k_grads):
    final_out_grads.append(
        tf.unstack(-(ll_grad + tf.reduce_mean(k_grad, axis=0))))

  final_trunk_grads = [-tf.reduce_mean(g, axis=0) for g in trunk_ll_grads]

  return logits, likelihoods, log_probs, (final_in_grads, final_trunk_grads,
                                          final_out_grads)
