from functools import partial
from typing import Tuple, Callable, List, Dict, Union

import haiku as hk
import jax
import tree
from jax import eval_shape, numpy as jnp, jacobian, partial
from jax import jit
from tensorflow_probability.substrates import jax as tfp

from baselines.diabetic_retinopathy_detection.fsvi_utils.ntk_utils import explicit_ntk, neural_tangent_ntk, diag_ntk_for_loop
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import partition_params, map_variable_name

tfd = tfp.distributions


def bnn_linearized_predictive_v2(
    apply_fn: Callable,
    params_mean: hk.Params,
    params_log_var: hk.Params,
    params_deterministic: hk.Params,
    state: hk.State,
    inducing_inputs: jnp.ndarray,
    rng_key: jnp.ndarray,
    stochastic_linearization: bool,
    full_ntk: bool,
    for_loop: bool = False,
    identity_cov: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return the mean and covariance of output of linearized BNN

    Currently this function is used for continual learning.

    @return
        mean: array of shape (batch_dim, output_dim)
        cov: array of shape
            if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
            otherwise, (batch_dim, output_dim)
    """
    params = hk.data_structures.merge(params_mean, params_log_var, params_deterministic)
    mean = apply_fn(
        params,
        state,
        None,
        inducing_inputs,
        rng_key,
        stochastic=stochastic_linearization,
        is_training=True,
    )[0]

    params_var = utils.sigma_transform(params_log_var)

    if full_ntk:
        assert not identity_cov, "not implemented"
        predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
            apply_fn,
            inducing_inputs,
            params_log_var,
            params_deterministic,
            state,
            rng_key,
            stochastic_linearization,
        )
        # the following line is equivalent to calculate J*diag(params_var)*J^T
        cov = get_ntk(
            predict_fn_for_empirical_ntk,
            delta_vjp_jvp,
            delta_vjp,
            params_mean,
            params_var,
        )
    else:
        if identity_cov:
            cov = jnp.ones_like(mean)
        else:
            def predict_f(params_mean, x):
                params = hk.data_structures.merge(params_mean, params_log_var, params_deterministic)
                return apply_fn(
                    params,
                    state,
                    None,
                    x,
                    rng_key,
                    stochastic=stochastic_linearization,
                    is_training=True,
                )[0]

            renamed_params_var = map_variable_name(
                params_var, lambda n: f"{n.split('_')[0]}_mu"
            )

            if for_loop:
                # NOTE: if using this option, then do not jit the outside function, otherwise the compilation is slow
                # large sample option, only use this if small_samples solution gives OOM error
                cov = diag_ntk_for_loop(
                    apply_fn=predict_f,
                    x=inducing_inputs,
                    params=params_mean,
                    sigma=renamed_params_var,
                )
            else:
                cov = neural_tangent_ntk(
                    apply_fn=predict_f,
                    x=inducing_inputs,
                    params=params_mean,
                    sigma=renamed_params_var,
                    diag=True
                )
    return mean, cov


# @partial(jit, static_argnums=(0,1,))
def bnn_linearized_predictive(
    apply_fn: Callable,
    params_mean: hk.Params,
    params_log_var: hk.Params,
    params_deterministic: hk.Params,
    state: hk.State,
    inducing_inputs: jnp.ndarray,
    rng_key: jnp.ndarray,
    stochastic_linearization: bool,
    full_ntk: bool,
    direct_ntk: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Return the mean and covariance of output of linearized BNN

    Currently this function is used in the following places
        - in the definition of loss function when model type is "fsvi"
        - in the definition of bnn induced prior function
    Basically, whenever we need to calculate the function distribution of a BNN from its parameter
    distribution, we can use this function.

    @param stochastic_linearization: if True, linearize around sampled parameter; otherwise linearize around mean
        parameters.

    @return
        mean: array of shape (batch_dim, output_dim)
        cov: array of shape
            if full_ntk is True, then (batch_dim, output_dim, batch_dim, output_dim)
            otherwise, (batch_dim, output_dim)
    """
    params = hk.data_structures.merge(params_mean, params_log_var, params_deterministic)
    mean = apply_fn(
        params,
        state,
        None,
        inducing_inputs,
        rng_key,
        stochastic=stochastic_linearization,
        is_training=True,
    )[0]

    params_var = utils.sigma_transform(params_log_var)

    if direct_ntk:
        predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
            apply_fn,
            inducing_inputs,
            params_log_var,
            params_deterministic,
            state,
            rng_key,
            stochastic_linearization,
        )
        renamed_params_var = map_variable_name(
            params_var, lambda n: f"{n.split('_')[0]}_mu"
        )
        # surprisingly, if I jit this function, there will be memory issue
        cov = explicit_ntk(
            fwd_fn=predict_fn_for_empirical_ntk,
            params=params_mean,
            sigma=renamed_params_var,
            diag=not full_ntk,
        )
    else:
        if full_ntk:
            predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
                apply_fn,
                inducing_inputs,
                params_log_var,
                params_deterministic,
                state,
                rng_key,
                stochastic_linearization,
            )
            # the following line is equivalent to calculate J*diag(params_var)*J^T
            cov = get_ntk(
                predict_fn_for_empirical_ntk,
                delta_vjp_jvp,
                delta_vjp,
                params_mean,
                params_var,
            )
        else:
            # TODO: Parallelize this loop
            # FIXME: Make this deterministic
            cov = []
            for i in range(inducing_inputs.shape[0]):
                predict_fn_for_empirical_ntk = convert_predict_f_only_mean(
                    apply_fn,
                    inducing_inputs[i : i + 1],
                    params_log_var,
                    params_deterministic,
                    state,
                    rng_key,
                    stochastic_linearization,
                )
                cov.append(
                    jnp.diag(
                        get_ntk(
                            predict_fn_for_empirical_ntk,
                            delta_vjp_jvp,
                            delta_vjp,
                            params_mean,
                            params_var,
                        )[0, :, 0, :]
                    )
                )
            cov = jnp.array(cov)

    return mean, cov


def induced_prior_fn_refactored(
    apply_fn: Callable,
    params: hk.Params,
    state: hk.State,
    inducing_inputs: Dict[int, jnp.ndarray],
    rng_key: jnp.ndarray,
    task_id: int,
    stochastic_linearization: bool,
    full_ntk: bool = False,
    identity_cov: bool = False,
    for_loop: bool = False
) -> Tuple[Dict[int, jnp.ndarray], Dict[int, jnp.ndarray]]:
    """
    Return mean and covariance matrix on inducing inputs for all tasks whose
    task id is equal or smaller than `task_id`
    """
    params_mean_prior, params_log_var_prior, params_deterministic_prior = partition_params(params)
    prior_means, prior_covs = {}, {}
    task_ids = sorted(inducing_inputs.keys())
    assert max(task_ids) <= task_id
    # TODO: task id here is irrelevant, should be able to get rid of the for loop
    for t_id in task_ids:
        x_inducing = inducing_inputs[t_id]
        prior_mean, prior_cov = bnn_linearized_predictive_v2(
            apply_fn,
            params_mean_prior,
            params_log_var_prior,
            params_deterministic_prior,
            state,
            x_inducing,
            rng_key,
            stochastic_linearization,
            full_ntk,
            identity_cov=identity_cov,
            for_loop=for_loop,
        )
        prior_means[t_id] = prior_mean
        prior_covs[t_id] = prior_cov
    return prior_means, prior_covs


# @partial(jit, static_argnums=(0,1,8,9))
def induced_prior_fn(
    apply_fn: Callable,
    params: hk.Params,
    state: hk.State,
    inducing_inputs: Union[jnp.ndarray, Dict[int, jnp.ndarray]],
    rng_key: jnp.ndarray,
    task_id: int,
    n_inducing_inputs: int,
    architecture,
    stochastic_linearization: bool,
    full_ntk: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Return mean and covariance matrix per task on inducing inputs
    """
    params_mean_prior, params_log_var_prior, params_deterministic_prior = partition_params(params)

    if task_id == 0:
        prior_mean, prior_cov = bnn_linearized_predictive_v2(
            apply_fn,
            params_mean_prior,
            params_log_var_prior,
            params_deterministic_prior,
            state,
            inducing_inputs,
            rng_key,
            stochastic_linearization,
            full_ntk,
            for_loop=False,
        )
        return [prior_mean], [prior_cov]  # arrays
    else:
        # TODO code can be simplified
        nb_tasks = len(inducing_inputs) // n_inducing_inputs
        number_inducing_inputs = nb_tasks * n_inducing_inputs
        inducing_inputs = inducing_inputs[:number_inducing_inputs]
        batched_inducing_inputs = inducing_inputs.reshape(
            (nb_tasks, n_inducing_inputs,) + inducing_inputs.shape[1:]
        )

        # TODO: everytime interface of bnn_linearized_predictive is changed, we need to change the following line
        mapped_fn = jax.vmap(
            bnn_linearized_predictive,
            in_axes=(None, None, None, None, None, 0, None, None, None),
        )

        prior_means, prior_covs = mapped_fn(
            apply_fn,
            params_mean_prior,
            params_log_var_prior,
            params_deterministic_prior,
            state,
            batched_inducing_inputs,
            rng_key,
            stochastic_linearization,
            full_ntk,
        )
        squeeze_fn = lambda array: jnp.squeeze(array, axis=0)
        prior_means = list(map(squeeze_fn, jnp.split(prior_means, nb_tasks, axis=0)))
        prior_covs = list(map(squeeze_fn, jnp.split(prior_covs, nb_tasks, axis=0)))
        return prior_means, prior_covs


# @partial(jit, static_argnums=(0,1,8,9))
def induced_prior_fn_v0(
    apply_fn: Callable,
    params: hk.Params,
    state: hk.State,
    inducing_inputs: Union[jnp.ndarray, Dict[int, jnp.ndarray]],
    rng_key: jnp.ndarray,
    task_id: int,
    n_inducing_inputs: int,
    architecture,
    stochastic_linearization: bool,
    full_ntk: bool = False,
) -> Tuple[List[jnp.ndarray], List[jnp.ndarray]]:
    """
    Return mean and covariance matrix per task on inducing inputs
    """
    params_mean_prior, params_log_var_prior, params_deterministic_prior = partition_params(params)

    inducing_inputs = jnp.expand_dims(inducing_inputs, axis=1)  # ensures that vmap'ed inputs are of shape [1, ...]
    mapped_fn = jax.vmap(bnn_linearized_predictive, in_axes=(None, None, None, None, None, 0, None, None, None), )

    prior_mean, prior_cov = mapped_fn(
        apply_fn,
        params_mean_prior,
        params_log_var_prior,
        params_deterministic_prior,
        state,
        inducing_inputs,
        rng_key,
        stochastic_linearization,
        full_ntk,
    )

    prior_mean, prior_cov = jnp.squeeze(prior_mean, axis=1), jnp.squeeze(prior_cov, axis=1)

    return prior_mean, prior_cov  # arrays


def convert_predict_f_only_mean(
    apply_fn,
    inputs,
    params_log_var,
    params_batchnorm,
    state,
    rng_key,
    stochastic_linearization,
):
    def predict_f_only_mean(params_mean):
        params = hk.data_structures.merge(params_mean, params_log_var, params_batchnorm)
        return apply_fn(
            params,
            state,
            None,
            inputs,
            rng_key,
            stochastic=stochastic_linearization,
            is_training=True,
        )[0]

    return predict_f_only_mean


@partial(jit, static_argnums=(0,))
def delta_vjp(predict_fn, params_mean, params_var, delta):
    vjp_tp = jax.vjp(predict_fn, params_mean)[1](delta)
    renamed_params_var = map_variable_name(
        params_var, lambda n: f"{n.split('_')[0]}_mu"
    )
    return (tree.map_structure(lambda x1, x2: x1 * x2, renamed_params_var, vjp_tp[0]),)


@partial(jit, static_argnums=(0, 1,))
def delta_vjp_jvp(
    predict_fn, delta_vjp: Callable, params_mean, params_var, delta: jnp.ndarray
):
    delta_vjp_ = partial(delta_vjp, predict_fn, params_mean, params_var)
    return jax.jvp(predict_fn, (params_mean,), delta_vjp_(delta))[1]


@partial(jit, static_argnums=(0, 1, 2))
def get_ntk(
    predict_fn, delta_vjp_jvp: Callable, delta_vjp: Callable, params_mean, params_var
) -> jnp.ndarray:
    """
    @param predict_fn: a function that takes in parameter and returns model output
    @param delta_vjp_jvp:
    @param delta_vjp:
    @param params_mean:
    @param params_var:
    @return:
        an array of shape (batch_dim, output_dim, batch_dim, output_dim)
    """
    predict_struct = eval_shape(predict_fn, params_mean)
    fx_dummy = jnp.ones(predict_struct.shape, predict_struct.dtype)
    # fx_dummy = jnp.ones((1,10), jnp.float32)
    delta_vjp_jvp_ = partial(
        delta_vjp_jvp, predict_fn, delta_vjp, params_mean, params_var
    )
    gram_matrix = jacobian(delta_vjp_jvp_)(fx_dummy)
    return gram_matrix
