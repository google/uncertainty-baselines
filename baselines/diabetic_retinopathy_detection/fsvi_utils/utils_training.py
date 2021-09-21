import pdb
import pickle
from typing import List, Tuple, Callable, Union, Sequence, Dict

import haiku as hk
import jax.numpy as jnp
import optax
import tree
import tensorflow as tf

from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import CNN, Model
from baselines.diabetic_retinopathy_detection.fsvi_utils import utils
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
        optimizer: str,
        inducing_input_type: str,
        activation: str,
        base_learning_rate,
        dropout_rate,
        input_shape: List[int],
        output_dim: int,
        full_ntk: bool,
        regularization,
        kl_scale,
        stochastic_linearization: bool,
        features_fixed: bool,
        full_cov,
        n_samples,
        n_train: int,
        batch_size,
        n_batches,
        inducing_inputs_bound: List[int],
        n_inducing_inputs: int,
        noise_std,
        epochs,
        uniform_init_minval,
        uniform_init_maxval,
        w_init,
        b_init,
        init_strategy,
        one_minus_momentum,
        lr_warmup_epochs,
        lr_decay_ratio,
        lr_decay_epochs,
        final_decay_factor,
        lr_schedule,
        layer_to_linearize=1,
        kl_type=0,
        use_map_loss=False,
        **kwargs,
    ):
        """

        @param task: examples: continual_learning_pmnist, continual_learning_sfashionmnist
        @param n_inducing_inputs: number of inducing points to draw from each task
        @param output_dim: the task-specific number of output dimensions
        """
        self.data_training = data_training
        self.optimizer = optimizer
        self.inducing_input_type = inducing_input_type
        self.activation = activation
        self.base_learning_rate = base_learning_rate
        self.dropout_rate = dropout_rate
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.full_ntk = full_ntk
        self.regularization = regularization
        self.kl_scale = kl_scale
        self.stochastic_linearization = stochastic_linearization
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
        self.epochs = epochs
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        self.w_init = w_init
        self.b_init = b_init
        self.init_strategy = init_strategy
        self.kl_type = kl_type
        self.one_minus_momentum = one_minus_momentum
        self.lr_warmup_epochs = lr_warmup_epochs
        self.lr_decay_ratio = lr_decay_ratio
        self.lr_decay_epochs = lr_decay_epochs
        self.final_decay_factor = final_decay_factor
        self.lr_schedule = lr_schedule
        self.layer_to_linearize = layer_to_linearize
        self.use_map_loss = use_map_loss

        if self.init_strategy == "he_normal_and_zeros":
            self.w_init = "he_normal"
            self.b_init = "zeros"
        elif self.init_strategy == "uniform":
            self.w_init = "uniform"
            self.b_init = "uniform"
        else:
            raise NotImplementedError(self.init_strategy)


        self.dropout = self.dropout_rate > 0

        self.stochastic_linearization_prior = False

        print(f"Full NTK computation: {self.full_ntk}")
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
        params_init: hk.Params,
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
    ]:
        opt = self._compose_optimizer()
        opt_state = opt.init(params_init)

        get_trainable_params = self.get_trainable_params_fn(params_init)
        get_variational_and_model_params = self.get_params_partition_fn(params_init)

        objective = self.initialize_objective(model=model)
        # LOSS
        loss, kl_evaluation = self._compose_loss(
            metrics=objective
        )
        # EVALUATION
        (
            log_likelihood_evaluation,
            nll_grad_evaluation,
            task_evaluation,
        ) = self._compose_evaluation_metrics(
            metrics=objective
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
        )

    def _compose_model(self) -> Model:
        model = CNN(
            output_dim=self.output_dim,
            activation_fn=self.activation,
            stochastic_parameters=True,
            linear_model=True,
            dropout=self.dropout,
            dropout_rate=self.dropout_rate,
            uniform_init_minval=self.uniform_init_minval,
            uniform_init_maxval=self.uniform_init_maxval,
            w_init=self.w_init,
            b_init=self.b_init,
        )
        return model

    def _compose_optimizer(self) -> optax.GradientTransformation:
        if "adam" in self.optimizer:
            opt = optax.adam(self.base_learning_rate)
        elif "sgd" == self.optimizer and self.lr_schedule == "linear":
            print("*" * 100)
            print("The linear learning schedule to reproducing deterministic is used")
            lr_schedule = warm_up_polynomial_schedule(
                base_learning_rate=self.base_learning_rate,
                end_learning_rate=self.final_decay_factor * self.base_learning_rate,
                decay_steps=(self.n_batches * (self.epochs - self.lr_warmup_epochs)),
                warmup_steps=self.n_batches * self.lr_warmup_epochs,
                decay_power=1.0
            )
            momentum = 1 - self.one_minus_momentum
            opt = optax.chain(
                optax.trace(decay=momentum, nesterov=True),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1),
            )
        elif "sgd" in self.optimizer and self.lr_schedule == "step":
            print("*" * 100)
            print("The step learning schedule to reproducing deterministic is used")
            DEFAULT_NUM_EPOCHS = 90
            lr_decay_epochs = [
                (int(start_epoch_str) * self.epochs) // DEFAULT_NUM_EPOCHS
                for start_epoch_str in self.lr_decay_epochs
            ]
            lr_schedule = warm_up_piecewise_constant_schedule(
                steps_per_epoch=self.n_batches,
                base_learning_rate=self.base_learning_rate,
                decay_ratio=self.lr_decay_ratio,
                decay_epochs=lr_decay_epochs,
                warmup_epochs=self.lr_warmup_epochs)

            momentum = 1 - self.one_minus_momentum
            opt = optax.chain(
                optax.trace(decay=momentum, nesterov=True),
                optax.scale_by_schedule(lr_schedule),
                optax.scale(-1),
            )
        else:
            raise ValueError("No optimizer specified.")
        return opt

    def _compose_loss(
        self, metrics: Objectives
    ) -> Tuple[Callable, Callable]:
        if self.use_map_loss:
            loss = metrics.map_loss_classification
            kl_evaluation = None
        else:
            loss = metrics.nelbo_fsvi_classification
            kl_evaluation = metrics.function_kl
        return loss, kl_evaluation

    def initialize_objective(self, model) -> Objectives:
        metrics = Objectives(
            model=model,
            regularization=self.regularization,
            kl_scale=self.kl_scale,
            full_cov=self.full_cov,
            n_samples=self.n_samples,
            output_dim=self.output_dim,
            stochastic_linearization=self.stochastic_linearization,
            full_ntk=self.full_ntk,
            kl_type=self.kl_type,
        )
        return metrics

    def _compose_evaluation_metrics(
        self, metrics: Objectives
    ) -> Tuple[Callable, Callable, Callable]:
        nll_grad_evaluation = metrics.nll_loss_classification
        task_evaluation = metrics.accuracy
        log_likelihood_evaluation = metrics._crossentropy_log_likelihood
        return log_likelihood_evaluation, nll_grad_evaluation, task_evaluation

    def get_inducing_input_fn(
            self,
            x_ood=[None],
    ):
        if self.inducing_input_type == "ood_rand" and len(x_ood) > 1:
            raise AssertionError("Inducing point type 'ood_rand' only works if one OOD set is specified.")
        def inducing_input_fn(x_batch, rng_key, n_inducing_inputs):
            # if n_inducing_inputs is None:
            #     n_inducing_inputs = self.n_inducing_inputs
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
        prior_mean: str,
        prior_cov: str,
    ) -> Tuple[
        Callable[[jnp.ndarray], List[jnp.ndarray]],
    ]:
        prior_mean = jnp.ones(self.n_inducing_inputs) * prior_mean
        prior_cov = jnp.ones(self.n_inducing_inputs) * prior_cov
        prior_fn = lambda inducing_inputs, model_params: [prior_mean, prior_cov]

        return prior_fn

    def get_params_partition_fn(self, params):
        variational_layers = list(params.keys())[-self.layer_to_linearize]  # TODO: set via input parameter

        def _get_params(params):
            variational_params, model_params = hk.data_structures.partition(lambda m, n, p: m in variational_layers, params)
            return variational_params, model_params

        return _get_params

    def get_trainable_params_fn(self, params):
        if self.features_fixed:
            trainable_layers = list(params.keys())[-self.layer_to_linearize]  # TODO: set via input parameter
        else:
            trainable_layers = list(params.keys())
        get_trainable_params = lambda params: hk.data_structures.partition(lambda m, n, p: m in trainable_layers, params)
        return get_trainable_params

    def kl_input_functions(
        self,
        prior_mean: str,
        prior_cov: str,
        x_ood=None,
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
        prior_mean, prior_cov = dtype_default(prior_mean), dtype_default(prior_cov)

        inducing_input_fn = self.get_inducing_input_fn(x_ood=x_ood)
        prior_fn = self.get_prior_fn(
            prior_mean=prior_mean,
            prior_cov=prior_cov,
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


def warm_up_piecewise_constant_schedule(
        steps_per_epoch,
        base_learning_rate,
        warmup_epochs,
        decay_epochs,
        decay_ratio,
    ):
    def schedule(count):
        lr_epoch = jnp.array(count, jnp.float32) / steps_per_epoch
        learning_rate = base_learning_rate
        if warmup_epochs >= 1:
            learning_rate *= lr_epoch / warmup_epochs
        _decay_epochs = [warmup_epochs] + decay_epochs
        for index, start_epoch in enumerate(_decay_epochs):
            learning_rate = jnp.where(
                lr_epoch >= start_epoch,
                base_learning_rate * decay_ratio ** index,
                learning_rate)
        return learning_rate
    return schedule


def warm_up_polynomial_schedule(
    base_learning_rate,
    end_learning_rate,
    decay_steps,
    warmup_steps,
    decay_power,
):
    poly_schedule = optax.polynomial_schedule(
        init_value=base_learning_rate,
        end_value=end_learning_rate,
        power=decay_power,
        transition_steps=decay_steps,
    )

    def schedule(step):
        lr = poly_schedule(step)
        indicator = jnp.maximum(0.0, jnp.sign(warmup_steps - step))
        warmup_lr = base_learning_rate * step / warmup_steps
        lr = warmup_lr * indicator + (1 - indicator) * lr
        return lr

    return schedule
