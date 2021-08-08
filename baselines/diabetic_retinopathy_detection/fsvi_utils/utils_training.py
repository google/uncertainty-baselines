from functools import partial
from typing import List, Tuple, Callable, Union, Sequence, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import sklearn
import tree

from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import CNN, Model
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils, utils_linearization
from baselines.diabetic_retinopathy_detection.fsvi_utils.haiku_mod import predicate_mean, predicate_var, predicate_batchnorm
from baselines.diabetic_retinopathy_detection.fsvi_utils.objectives import Objectives_hk as Objectives

classification_datasets = [
    "mnist",
    "notmnist",
    "fashionmnist",
    "cifar10",
    "svhn",
    "two_moons",
    "dr",
]
continual_learning_datasets = [
    "pmnist",
    "smnist",
    "smnist_sh",
    "pfashionmnist",
    "sfashionmnist",
    "sfashionmnist_sh",
    "cifar",
    "cifar_small",
    "toy",
    "toy_sh",
    "toy_reprod",
]
classification_datasets.extend(
    [f"continual_learning_{ds}" for ds in continual_learning_datasets]
)
regression_datasets = ["uci", "offline_rl", "snelson", "oat1d", "subspace_inference","ihdp"]

dtype_default = jnp.float32


class Training:
    def __init__(
        self,
        data_training,
        model_type: str,
        optimizer: str,
        inducing_input_type: str,
        prior_type,
        architecture,
        activation: str,
        learning_rate,
        dropout_rate,
        batch_normalization: bool,
        input_shape: List[int],
        output_dim: int,
        full_ntk: bool,
        regularization,
        kl_scale,
        stochastic_linearization: bool,
        linear_model: bool,
        features_fixed: bool,
        full_cov,
        n_samples,
        n_train: int,
        batch_size,
        n_batches,
        inducing_inputs_bound: List[int],
        n_inducing_inputs: int,
        noise_std,
        map_initialization,
        **kwargs,
    ):
        """

        @param task: examples: continual_learning_pmnist, continual_learning_sfashionmnist
        @param n_inducing_inputs: number of inducing points to draw from each task
        @param output_dim: the task-specific number of output dimensions
        """
        self.data_training = data_training
        self.model_type = model_type
        self.optimizer = optimizer
        self.inducing_input_type = inducing_input_type
        self.prior_type = prior_type
        self.architecture = architecture
        self.activation = activation
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.batch_normalization = batch_normalization
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.full_ntk = full_ntk
        self.regularization = regularization
        self.kl_scale = kl_scale
        self.stochastic_linearization = stochastic_linearization
        self.linear_model = linear_model
        self.features_fixed = features_fixed
        self.full_cov = full_cov
        self.n_samples = n_samples
        self.n_batches = n_batches
        self.n_train = n_train
        self.batch_size = batch_size
        self.inducing_inputs_bound = inducing_inputs_bound
        self.n_inducing_inputs = n_inducing_inputs
        self.noise_std = noise_std
        self.data_training_id = 0

        self.map_initialization = map_initialization

        self.dropout = "dropout" in self.model_type
        if not self.dropout and self.dropout_rate > 0:
            raise ValueError("Dropout rate not zero in non-dropout model.")

        if prior_type == 'bnn_induced':
            self.stochastic_linearization_prior = True
        elif prior_type == 'blm_induced':
            self.stochastic_linearization_prior = False
        else:
            self.stochastic_linearization_prior = False

        print(f"\n"
              f"MAP initialization: {self.map_initialization}")
        print(f"Full NTK computation: {self.full_ntk}")
        print(f"Linear model: {self.linear_model}")
        print(f"Stochastic linearization (posterior): {self.stochastic_linearization}")
        print(f"Stochastic linearization (prior): {self.stochastic_linearization_prior}"
              f"\n")

    def initialize_model(
        self,
        rng_key,
    ):
        model = self._compose_model()
        init_fn, apply_fn = model.forward
        # INITIALIZE NETWORK STATE + PARAMETERS
        x_init = jnp.ones(self.input_shape)
        params_init, state = init_fn(
            rng_key, x_init, rng_key, model.stochastic_parameters, is_training=True
        )

        return model, init_fn, apply_fn, state, params_init

    def initialize_optimization(
        self,
        model,
        apply_fn: Callable,
        params_init: hk.Params,
        state: hk.State,
        rng_key: jnp.ndarray,
    ) -> Tuple[
        optax.GradientTransformation,
        Union[optax.OptState, Sequence[optax.OptState]],
        Callable,
        Callable,
        Objectives,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        str,
    ]:
        opt = self._compose_optimizer()
        opt_state = opt.init(params_init)

        get_trainable_params = self.get_trainable_params_fn(params_init)
        get_variational_and_model_params = self.get_params_partition_fn(params_init)

        # # FIXME: doesn't currently seem to work
        # def _pred_fn(apply_fn, params, state, inputs, rng_key, n_samples):
        #     rng_key, subkey = jax.random.split(rng_key)
        #     preds_samples = jnp.expand_dims(apply_fn(params, state, None, inputs, rng_key, stochastic=False, is_training=True)[0], 0)
        #     for i in range(n_samples - 1):
        #         rng_key, subkey = jax.random.split(rng_key)
        #         preds_samples = jnp.concatenate(
        #             (preds_samples, jnp.expand_dims(apply_fn(params, state, None, inputs, rng_key, stochastic=False, is_training=True)[0], 0)), 0)
        #     return preds_samples
        #
        # pred_fn = jax.jit(partial(_pred_fn,
        #     apply_fn=apply_fn,
        #     state=state,
        #     n_samples=self.n_samples,
        # ))

        prediction_type = decide_prediction_type(self.data_training)
        objective = self._compose_objective(
            model=model, apply_fn=apply_fn, state=state, rng_key=rng_key
        )
        # LOSS
        loss, kl_evaluation = self._compose_loss(
            prediction_type=prediction_type, metrics=objective
        )
        # EVALUATION
        (
            log_likelihood_evaluation,
            nll_grad_evaluation,
            task_evaluation,
        ) = self._compose_evaluation_metrics(
            prediction_type=prediction_type, metrics=objective
        )

        return (
            opt,
            opt_state,
            get_trainable_params,
            get_variational_and_model_params,
            objective,
            loss,
            kl_evaluation,
            log_likelihood_evaluation,
            nll_grad_evaluation,
            task_evaluation,
            prediction_type,
        )

    def _compose_model(self) -> Model:
        if "cnn" or "resnet" in self.model_type:
            network_class = CNN
        else:
            raise ValueError("Invalid network type.")

        stochastic_parameters = "mfvi" in self.model_type or "fsvi" in self.model_type

        # DEFINE NETWORK
        model = network_class(
            architecture=self.architecture,
            output_dim=self.output_dim,
            activation_fn=self.activation,
            regularization=self.regularization,
            stochastic_parameters=stochastic_parameters,
            linear_model=self.linear_model,
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
            batch_normalization=self.batch_normalization,
        )
        return model

    def _compose_optimizer(self) -> optax.GradientTransformation:
        if "adam" in self.optimizer:
            # epoch_points = self.optimizer.split("_")[1:]
            # for i in range(len(epoch_points)):
            #     epoch_points[i] = int(epoch_points[i])
            # epoch_points = (jnp.array(epoch_points) * self.n_batches).tolist()
            #
            # def schedule_fn(learning_rate, n_batches):
            #     # return piecewise_constant_schedule(learning_rate, epoch_points, 0.5)
            #     return optax.polynomial_schedule(
            #         init_value=learning_rate,
            #         end_value=1e-5,
            #         power=0.95,
            #         transition_steps=epoch_points[-1]
            #     )
            # schedule_fn_final = schedule_fn(self.learning_rate, self.n_batches)
            #
            # def make_adam_optimizer():
            #     return optax.chain(
            #         optax.scale_by_schedule(schedule_fn_final),
            #         optax.scale_by_adam(),
            #         optax.scale(-self.learning_rate),
            #     )
            #
            # opt = make_adam_optimizer()
            opt = optax.adam(self.learning_rate)
        elif "sgd" in self.optimizer:
            epoch_points = self.optimizer.split("_")[1:]
            for i in range(len(epoch_points)):
                epoch_points[i] = int(epoch_points[i])
            epoch_points = (jnp.array(epoch_points) * self.n_batches).tolist()

            def schedule_fn(learning_rate, n_batches):
                return piecewise_constant_schedule(learning_rate, epoch_points, 0.3)

            schedule_fn_final = schedule_fn(self.learning_rate, self.n_batches)
            momentum = 0.9

            def make_sgd_optimizer():
                return optax.chain(
                    optax.trace(decay=momentum, nesterov=False),
                    optax.scale_by_schedule(schedule_fn_final),
                    optax.scale(-1),
                )

            opt = make_sgd_optimizer()
        else:
            raise ValueError("No optimizer specified.")
        return opt

    def _compose_loss(
        self, prediction_type: str, metrics: Objectives
    ) -> Tuple[Callable, Callable]:
        assert "continual_learning" not in self.data_training, "This method is deprecated for continual learning"
        if "fsvi" in self.model_type:
            if prediction_type == "classification":
                loss = metrics.nelbo_fsvi_classification
            kl_evaluation = metrics.function_kl
        else:
            raise ValueError("No loss specified.")
        return loss, kl_evaluation

    def _compose_objective(self, model, apply_fn, state, rng_key) -> Objectives:
        # `linearize_fn` is a function that takes in params_mean, params_log_var, params_batchnorm,
        # inducing_inputs, rng_key, state, and returns mean and covariance of inducing_inputs

        metrics = Objectives(
            architecture=self.architecture,
            apply_fn=apply_fn,
            predict_f=model.predict_f,
            predict_f_deterministic=model.predict_f_deterministic,
            predict_y=model.predict_y,
            predict_f_multisample=model.predict_f_multisample,
            predict_f_multisample_jitted=model.predict_f_multisample_jitted,
            predict_y_multisample=model.predict_y_multisample,
            predict_y_multisample_jitted=model.predict_y_multisample_jitted,
            regularization=self.regularization,
            kl_scale=self.kl_scale,
            full_cov=self.full_cov,
            n_samples=self.n_samples,
            output_dim=self.output_dim,
            noise_std=self.noise_std,
            prior_type=self.prior_type,
            stochastic_linearization=self.stochastic_linearization,
            linear_model=self.linear_model,
            full_ntk=self.full_ntk,
        )
        return metrics

    def _compose_evaluation_metrics(
        self, prediction_type: str, metrics: Objectives
    ) -> Tuple[Callable, Callable, Callable]:
        if prediction_type == "classification":
            nll_grad_evaluation = metrics.nll_loss_classification
            task_evaluation = metrics.accuracy
            log_likelihood_evaluation = metrics._crossentropy_log_likelihood
        else:
            raise ValueError(f"Unrecognized prediction_type: {prediction_type}")
        return log_likelihood_evaluation, nll_grad_evaluation, task_evaluation

    def get_inducing_input_fn(
            self,
            x_ood=[None],
    ):
        if self.inducing_input_type == "ood_rand" and len(x_ood) > 1:
            raise AssertionError("Inducing point type 'ood_rand' only works if one OOD set is specified.")
        def inducing_input_fn(x_batch, rng_key, n_inducing_inputs=None):
            if n_inducing_inputs is None:
                n_inducing_inputs = self.n_inducing_inputs
            return utils.select_inducing_inputs(
                n_inducing_inputs=n_inducing_inputs,
                inducing_input_type=self.inducing_input_type,
                inducing_inputs_bound=self.inducing_inputs_bound,
                input_shape=self.input_shape,
                x_batch=x_batch,
                x_ood=x_ood,
                n_train=self.n_train,
                rng_key=rng_key,
            )
        return inducing_input_fn

    def get_prior_fn(
        self,
        apply_fn: Callable,
        predict_f_deterministic: Callable,
        state: hk.State,
        params: hk.Params,
        prior_mean: str,
        prior_cov: str,
        rng_key,
        prior_type,
        task_id,
        jit_prior=True,
        identity_cov=False,
    ) -> Tuple[
        Callable[[jnp.ndarray], List[jnp.ndarray]],
    ]:
        assert "continual_learning" not in self.data_training, "This method is deprecated for continual learning"
        if prior_type == "bnn_induced" or prior_type == "blm_induced":
            rng_key0, _ = jax.random.split(rng_key)

            params_prior = get_prior_params(
                params_init=params,
                prior_mean=prior_mean,
                prior_cov=prior_cov,
            )

            # prior_fn is a function of inducing_inputs and params
            if not self.linear_model:
                prior_fn = lambda inducing_inputs, model_params: partial(  # params args are unused
                    utils_linearization.induced_prior_fn_v0,
                    apply_fn=apply_fn,
                    params=params_prior,
                    state=state,
                    rng_key=rng_key0,
                    task_id=task_id,
                    n_inducing_inputs=self.n_inducing_inputs,
                    architecture=self.architecture,
                    stochastic_linearization=self.stochastic_linearization_prior,
                    full_ntk=self.full_ntk,
                )
            else:
                def prior_fn(inducing_inputs, model_params):
                    params_prior_final_layer, _ = self.get_params_partition_fn(params_prior)(params_prior)
                    params_updated = hk.data_structures.merge(params_prior_final_layer, model_params)
                    return partial(
                        utils_linearization.induced_prior_fn_v0,
                        apply_fn=apply_fn,
                        state=state,
                        rng_key=rng_key0,
                        task_id=task_id,
                        n_inducing_inputs=self.n_inducing_inputs,
                        architecture=self.architecture,
                        stochastic_linearization=self.stochastic_linearization_prior,
                        linear_model=self.linear_model,
                        full_ntk=self.full_ntk,
                    )(inducing_inputs=inducing_inputs, params=params_updated)
            if jit_prior and not identity_cov:
                prior_fn = jax.jit(prior_fn)

        elif prior_type == "rbf":
            prior_mean = jnp.ones(self.n_inducing_inputs) * prior_mean
            prior_fn = lambda inducing_inputs, model_params: [
                prior_mean,
                sklearn.metrics.pairwise.rbf_kernel(
                    inducing_inputs.reshape([inducing_inputs.shape[0], -1]), gamma=None
                )
                * prior_cov,
            ]

        elif prior_type == "fixed":
            prior_mean = jnp.ones(self.n_inducing_inputs) * prior_mean
            prior_cov = jnp.ones(self.n_inducing_inputs) * prior_cov
            prior_fn = lambda inducing_inputs, model_params: [prior_mean, prior_cov]

        elif prior_type == "map_mean":
            prior_mean = partial(
                predict_f_deterministic, params=params, state=state, rng_key=rng_key
            )
            prior_cov = jnp.ones(self.n_inducing_inputs) * prior_cov
            prior_fn = lambda inducing_inputs, model_params: [
                prior_mean(inputs=inducing_inputs),
                prior_cov,
            ]

            if jit_prior:
                prior_fn = jax.jit(prior_fn)

        elif prior_type == "map_induced":
            rng_key0, _ = jax.random.split(rng_key)
            params_prior = params

            prior_fn = lambda inducing_inputs, model_params: partial(
                utils_linearization.induced_prior_fn,
                apply_fn=apply_fn,
                params=params_prior,
                state=state,
                rng_key=rng_key0,
                task_id=task_id,
                n_inducing_inputs=self.n_inducing_inputs,
                architecture=self.architecture,
            )
            if jit_prior:
                prior_fn = jax.jit(prior_fn)

        else:
            if "fsvi" in self.model_type or "mfvi" in self.model_type:
                raise ValueError("No prior type specified.")
            else:
                prior_fn = lambda inducing_inputs, model_params: [0, 0]

        return prior_fn

    def get_params_partition_fn(self, params):
        if "fsvi" in self.model_type or "mfvi" in self.model_type:
            if self.linear_model:
                variational_layers = list(params.keys())[-2]  # TODO: set via input parameter
            else:
                variational_layers = list(params.keys())
        else:
            variational_layers = [""]

        def _get_params(params):
            variational_params, model_params = hk.data_structures.partition(lambda m, n, p: m in variational_layers, params)
            return variational_params, model_params

        return _get_params

    def get_trainable_params_fn(self, params):
        if self.linear_model and self.features_fixed:
            trainable_layers = list(params.keys())[-2]  # TODO: set via input parameter
        else:
            trainable_layers = list(params.keys())
        get_trainable_params = lambda params: hk.data_structures.partition(lambda m, n, p: m in trainable_layers, params)
        return get_trainable_params

    def kl_input_functions(
        self,
        apply_fn: Callable,
        predict_f_deterministic: Callable,
        state: hk.State,
        params: hk.Params,
        prior_mean: str,
        prior_cov: str,
        rng_key,
        x_ood=None,
        prior_type=None,
        task_id=None,
        jit_prior=True,
        identity_cov=False,
    ) -> Tuple[
        Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
        Callable[[jnp.ndarray], List[jnp.ndarray]],
    ]:
        """
        @predict_f_deterministic: function to do forward pass
        @param prior_mean: example: "0.0"
        @param prior_cov: example: "0.0"
        @return:
            inducing_input_fn
            prior_fn: a function that takes in an array of inducing input points and return the mean
                and covariance of the outputs at those points
        """
        task_id = self.data_training_id if task_id is None else task_id
        prior_type = self.prior_type if prior_type is None else prior_type
        prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)

        inducing_input_fn = self.get_inducing_input_fn(x_ood=x_ood)
        prior_fn = self.get_prior_fn(
            apply_fn,
            predict_f_deterministic,
            state,
            params,
            prior_mean,
            prior_cov,
            rng_key,
            prior_type,
            task_id,
            jit_prior,
            identity_cov,
        )

        return inducing_input_fn, prior_fn


def get_prior_params(
    params_init: hk.Params,
    prior_mean: str,
    prior_cov: str,
) -> hk.Params:
    prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)

    params_mean = tree.map_structure(
        lambda p: jnp.ones_like(p) * prior_mean,
        hk.data_structures.filter(predicate_mean, params_init),
    )
    params_log_var = tree.map_structure(
        lambda p: jnp.ones_like(p) * jnp.log(prior_cov),
        hk.data_structures.filter(predicate_var, params_init),
    )

    params_prior = hk.data_structures.merge(params_mean, params_log_var)
    return params_prior


def piecewise_constant_schedule(init_value, boundaries, scale):
    """
    Return a function that takes in the update count and returns a step size.

    The step size is equal to init_value * (scale ** <number of boundaries points not greater than count>)
    """
    def schedule(count):
        v = init_value
        for threshold in boundaries:
            indicator = jnp.maximum(0.0, jnp.sign(threshold - count))
            v = v * indicator + (1 - indicator) * scale * v
        return v

    return schedule


def decide_prediction_type(data_training: str) -> str:
    if data_training in classification_datasets:
        prediction_type = "classification"
    elif data_training in regression_datasets:
        prediction_type = "regression"
    else:
        raise ValueError(f"Prediction type not recognized: {data_training}")
    return prediction_type
