from functools import partial
from typing import Tuple

import tree
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from jax import jit
from jax.experimental import optimizers
from jax.tree_util import tree_flatten
import haiku as hk

from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils_linearization
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import partition_params

dtype_default = jnp.float32
eps = 1e-6


class Objectives_hk:
    def __init__(
        self,
        architecture,
        apply_fn,
        predict_f,
        predict_f_deterministic,
        predict_y,
        predict_y_multisample,
        predict_y_multisample_jitted,
        output_dim,
        kl_scale: str,
        n_batches,
        predict_f_multisample,
        predict_f_multisample_jitted,
        noise_std,
        regularization,
        n_samples,
        full_cov,
        prior_type,
        stochastic_linearization,
        linear_model,
        full_ntk=False,
    ):
        self.architecture = architecture
        self.apply_fn = apply_fn
        self.predict_f = predict_f
        self.predict_f_deterministic = predict_f_deterministic
        self.predict_y = predict_y
        self.predict_f_multisample = predict_f_multisample
        self.predict_f_multisample_jitted = predict_f_multisample_jitted
        self.predict_y_multisample = predict_y_multisample
        self.predict_y_multisample_jitted = predict_y_multisample_jitted
        self.output_dim = output_dim
        self.kl_scale = kl_scale
        self.regularization = regularization
        self.noise_std = noise_std
        self.n_batches = n_batches
        self.n_samples = n_samples
        self.full_cov = full_cov
        self.prior_type = prior_type
        self.stochastic_linearization = stochastic_linearization
        self.linear_model = linear_model
        self.full_ntk = full_ntk

    @partial(jit, static_argnums=(0,10,))
    def objective_and_state(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        objective_fn,
    ):
        is_training = True
        params = hk.data_structures.merge(trainable_params, non_trainable_params,)

        objective = objective_fn(
            params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            is_training,
        )

        state = self.apply_fn(
            params, state, rng_key, inputs, rng_key, stochastic=True, is_training=is_training
        )[1]

        return objective, state

    @partial(jit, static_argnums=(0,))
    def accuracy(self, preds, targets):
        target_class = jnp.argmax(targets, axis=1)
        predicted_class = jnp.argmax(preds, axis=1)
        return jnp.mean(predicted_class == target_class)

    @partial(jit, static_argnums=(0,))
    def kl_gaussian(self, mean_q, mean_p, log_var_q, log_var_p):
        # sigma_q = jnp.exp(log_var_q / 2)
        # kl = (-log_var_q + 0.5 * (sigma_q ** 2 + mean_q ** 2 - 1)).sum()
        mean_p = jnp.ones_like(mean_q) * mean_p
        cov_p = jnp.ones_like(log_var_q) * jnp.exp(log_var_p)
        cov_q = jnp.exp(log_var_q)
        kl = utils.kl_diag(mean_q, mean_p, cov_q, cov_p)
        return kl

    def _gaussian_log_likelihood(self, preds_f_samples, targets):
        targets = jnp.tile(targets, (preds_f_samples.shape[0], 1, 1))
        likelihood = tfd.Normal(preds_f_samples, self.noise_std)
        log_likelihood = jnp.sum(
            jnp.sum(jnp.mean(likelihood.log_prob(targets), 0), 0), 0
        )
        # preds_f_mean = preds_f_mean.reshape(-1, 1)
        # targets = targets.reshape(-1, 1)
        # likelihood = tfd.Normal(preds_f_mean, self.noise_std)
        # log_likelihood = -jnp.mean(jnp.sum((targets - preds_f_mean) ** 2, 1), 0)
        return log_likelihood

    def _crossentropy_log_likelihood(self, preds_f_samples, targets):
        log_likelihood = jnp.mean(
            jnp.sum(
                jnp.sum(
                    targets * jax.nn.log_softmax(preds_f_samples, axis=-1), axis=-1
                ),
                axis=-1,
            ),
            axis=0,
        )
        return log_likelihood

    def _parameter_kl(
        self,
        params,
        prior_mean,
        prior_cov,
    ):
        params_mean, params_log_var, params_batchnorm = partition_params(params)
        params_mean_tp, _ = tree_flatten(params_mean)
        params_log_var_tp, _ = tree_flatten(params_log_var)
        kl = jnp.array(
            list(map(self.kl_gaussian, params_mean_tp, prior_mean, params_log_var_tp, prior_cov))
        ).sum()
        return kl

    def _function_kl(
        self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ) -> Tuple[jnp.ndarray, float]:
        """
        Evaluate the multi-output KL between the function distribution obtained by linearising BNN around
        params, and the prior function distribution represented by (`prior_mean`, `prior_cov`)

        @param inputs: used for computing scale, only the shape is used
        @param inducing_inputs: used for computing scale and function distribution used in KL

        @return:
            kl: scalar value of function KL
            scale: scale to multiple KL with
        """
        # TODO: Maybe change "params_deterministic" to "params_model"
        params_mean, params_log_var, params_deterministic = partition_params(params)
        scale = compute_scale(self.kl_scale, inputs, inducing_inputs.shape[0])

        # mean, cov = self.linearize_fn(
        #     params_mean=params_mean,
        #     params_log_var=params_log_var,
        #     params_batchnorm=params_batchnorm,
        #     state=state,
        #     inducing_inputs=inducing_inputs,
        #     rng_key=rng_key,
        # )

        mean, cov = utils_linearization.bnn_linearized_predictive(
            self.apply_fn,
            params_mean,
            params_log_var,
            params_deterministic,
            state,
            inducing_inputs,
            rng_key,
            self.stochastic_linearization,
            self.full_ntk,
        )

        kl = utils.kl_divergence(
            mean,
            prior_mean,
            cov,
            prior_cov,
            self.output_dim,
            self.full_cov,
            self.prior_type,
        )

        return kl, scale

    def _nll_loss_regression(self, params, state, inputs, targets, rng_key, is_training):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = (
            self.gaussian_log_likelihood(preds_f_samples, targets) / targets.shape[0]
        )
        loss = -log_likelihood
        return loss

    def _nll_loss_classification(self, params, state, inputs, targets, rng_key, is_training):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = (
            self.crossentropy_log_likelihood(preds_f_samples, targets)
            / targets.shape[0]
        )
        loss = -log_likelihood
        return loss

    def _map_loss_regression(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood = self._nll_loss_regression(
            params, state, inputs, targets, rng_key, is_training,
        )
        reg_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]
        loss = (
            negative_log_likelihood
            + self.regularization * jnp.square(optimizers.l2_norm(reg_params))
        )
        return loss

    def _map_loss_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood = self._nll_loss_classification(
            params, state, inputs, targets, rng_key, is_training,
        )
        reg_params = [p for ((mod_name, _), p) in tree.flatten_with_path(params) if 'batchnorm' not in mod_name]
        loss = (
            negative_log_likelihood
            + self.regularization * jnp.square(optimizers.l2_norm(reg_params))
        )
        return loss

    def _fsmap_loss_regression(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood = self._nll_loss_regression(
            params, state, inputs, targets, rng_key, is_training,
        )
        preds_f_mean = self.apply_fn(
            params, state, rng_key, inducing_inputs, rng_key, stochastic=False, is_training=is_training
        )[0]
        loss = (
            negative_log_likelihood
            + self.regularization * jnp.sum(jnp.square(preds_f_mean))
        )
        return loss

    def _fsmap_loss_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        negative_log_likelihood = self._nll_loss_classification(
            params, state, inputs, targets, rng_key, is_training,
        )
        preds_f_mean = self.apply_fn(
            params, state, rng_key, inducing_inputs, rng_key, stochastic=False, is_training=is_training
        )[0]
        loss = (
            negative_log_likelihood
            + self.regularization * jnp.sum(jnp.square(preds_f_mean))
        )
        return loss

    def _elbo_mfvi_regression(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.gaussian_log_likelihood(preds_f_samples, targets)
        kl = self.parameter_kl(
            params,
            prior_mean,
            prior_cov,
        )
        scale = 1.0 / self.n_batches
        elbo = log_likelihood - dtype_default(self.kl_scale) * scale * kl
        return elbo, log_likelihood, kl, scale

    def _elbo_mfvi_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets)
        kl = self.parameter_kl(
            params,
            prior_mean,
            prior_cov,
        )
        scale = 1.0 / self.n_batches
        elbo = log_likelihood - dtype_default(self.kl_scale) * scale * kl
        return elbo, log_likelihood, kl, scale


    def _elbo_fsvi_regression(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.gaussian_log_likelihood(preds_f_samples, targets)
        kl, scale = self.function_kl(
            params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
        )
        elbo = log_likelihood - scale * kl
        # elbo = log_likelihood / inputs.shape[0] - scale * kl / inducing_inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    def _elbo_fsvi_classification(
        self,
        params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
        is_training,
    ):
        preds_f_samples, _, _ = self.predict_f_multisample_jitted(
            params, state, inputs, rng_key, self.n_samples, is_training,
        )
        log_likelihood = self.crossentropy_log_likelihood(preds_f_samples, targets)
        kl, scale = self.function_kl(
            params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
        )
        elbo = log_likelihood - scale * kl
        # elbo = log_likelihood / inputs.shape[0] - scale * kl / inducing_inputs.shape[0]

        return elbo, log_likelihood, kl, scale

    @partial(jit, static_argnums=(0,))
    def gaussian_log_likelihood(self, preds_f_samples, targets):
        return self._gaussian_log_likelihood(preds_f_samples, targets)

    @partial(jit, static_argnums=(0,))
    def crossentropy_log_likelihood(self, preds_f_samples, targets):
        return self._crossentropy_log_likelihood(preds_f_samples, targets)

    @partial(jit, static_argnums=(0,))
    def function_kl(
        self, params, state, prior_mean, prior_cov, inputs, inducing_inputs, rng_key,
    ):
        return self._function_kl(
            params=params,
            state=state,
            prior_mean=prior_mean,
            prior_cov=prior_cov,
            inputs=inputs,
            inducing_inputs=inducing_inputs,
            rng_key=rng_key,
        )

    @partial(jit, static_argnums=(0,))
    def parameter_kl(
        self,
        params,
        prior_mean,
        prior_cov,
    ):
        return self._parameter_kl(
            params,
            prior_mean,
            prior_cov,
        )

    @partial(jit, static_argnums=(0,))
    def nll_loss_regression(self, trainable_params, non_trainable_params, state, inputs, targets, rng_key):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            inputs,
            targets,
            rng_key,
            self._nll_loss_regression
        )

    @partial(jit, static_argnums=(0,))
    def nll_loss_classification(self, trainable_params, non_trainable_params, state, inputs, targets, rng_key):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            inputs,
            targets,
            rng_key,
            self._nll_loss_classification
        )

    @partial(jit, static_argnums=(0,))
    def map_loss_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._map_loss_regression,
        )

    @partial(jit, static_argnums=(0,))
    def map_loss_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._map_loss_classification,
        )

    @partial(jit, static_argnums=(0,))
    def fsmap_loss_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._fsmap_loss_regression,
        )

    @partial(jit, static_argnums=(0,))
    def fsmap_loss_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        return self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._fsmap_loss_classification,
        )

    @partial(jit, static_argnums=(0,))
    def nelbo_mfvi_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_mfvi_regression,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_mfvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_mfvi_classification,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_fsvi_regression(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_fsvi_regression,
        )

        return -elbo, state

    @partial(jit, static_argnums=(0,))
    def nelbo_fsvi_classification(
        self,
        trainable_params,
        non_trainable_params,
        state,
        prior_mean,
        prior_cov,
        inputs,
        targets,
        inducing_inputs,
        rng_key,
    ):
        (elbo, log_likelihood, kl, scale), state = self.objective_and_state(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            inputs,
            targets,
            inducing_inputs,
            rng_key,
            self._elbo_fsvi_classification,
        )

        return -elbo, state


@partial(jit, static_argnums=(0,))
def compute_scale(kl_scale: str, inputs: jnp.ndarray, n_inducing_inputs: int) -> float:
    if kl_scale == "none":
        scale = 1.0
    elif kl_scale == "equal":
        scale = inputs.shape[0] / n_inducing_inputs
    elif kl_scale == "normalized":
        scale = 1.0 / n_inducing_inputs
    else:
        scale = dtype_default(kl_scale)
    return scale
