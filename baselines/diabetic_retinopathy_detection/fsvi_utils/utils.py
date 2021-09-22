import os
import getpass
from copy import copy
from typing import Tuple, List
import random as random_py

import jax
import tree
from jax import jit, random
from jax import numpy as jnp
import tensorflow as tf
import torch
import numpy as np
import matplotlib.pyplot as plt

from baselines.diabetic_retinopathy_detection.fsvi_utils.jax_utils import KeyHelper


dtype_default = jnp.float32
TWO_TUPLE = Tuple[int, int]


def initialize_random_keys(seed: int) -> KeyHelper:
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng_key = jax.random.PRNGKey(seed)
    kh = KeyHelper(key=rng_key)
    random_py.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.random.manual_seed(seed)
    return kh


def to_one_hot(x, k, dtype=dtype_default):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


@jit
def sigma_transform(params_log_var):
    return tree.map_structure(lambda p: jnp.exp(p), params_log_var)


@jit
def kl_divergence(
    mean_q, mean_p, cov_q, cov_p,
):
    """
    Return KL(q || p)

    All inputs are either of shape (batch_dim, output_dim).
    """
    function_kl = 0
    output_dim = mean_q.shape[1]
    for i in range(output_dim):
        mean_q_tp = mean_q[:, i]
        cov_q_tp = cov_q[:, i]
        mean_p_tp = mean_p[:, i]
        cov_p_tp = cov_p[:, i]
        function_kl += kl_diag(
            mean_q_tp,
            mean_p_tp,
            cov_q_tp,
            cov_p_tp,
        )
    return function_kl


@jit
def kl_diag(mean_q, mean_p, cov_q, cov_p) -> jnp.ndarray:
    """
    Return KL(q || p)
    NOte: all inputs are 1D arrays.

    @param cov_q: the diagonal of covariance
    @return:
        a scalar
    """
    kl_1 = jnp.log(cov_p ** 0.5) - jnp.log(cov_q ** 0.5)
    kl_2 = (cov_q + (mean_q - mean_p) ** 2) / (2 * cov_p)
    kl_3 = -1 / 2
    kl = jnp.sum(kl_1 + kl_2 + kl_3)
    return kl


def select_inducing_inputs(
    n_inducing_inputs: int,
    inducing_input_type: str,
    inducing_inputs_bound: List[int],
    input_shape: List[int],
    x_batch: np.ndarray,
    x_ood,
    n_train: int,
    rng_key: jnp.ndarray,
    plot_samples: bool = False,
) -> jnp.ndarray:
    """
    Select inducing points

    @param task:
    @param model_type:
    @param n_inducing_inputs: integer, number of inducing points to select
    @param inducing_input_type: strategy to select inducing points
    @param inducing_inputs_bound: a list of two floats, usually [0.0, 1.0]
    @param input_shape: expected shape of inducing points, including batch dimension
    @param x_batch: input data of the current task
    @param x_ood:
    @param n_train: number of training points
    @param rng_key:
    @param plot_samples:
    @return:
    """
    permutation = jax.random.permutation(key=rng_key, x=x_batch.shape[0])
    x_batch_permuted = x_batch[permutation, :]
    # avoid modifying input variables
    input_shape = copy(input_shape)

    if inducing_input_type == "uniform_rand":
        input_shape[0] = n_inducing_inputs
        inducing_inputs = jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
    elif inducing_input_type == "gaussian_rand":
        input_shape[0] = n_inducing_inputs
        inducing_inputs = random.normal(rng_key, input_shape, dtype_default)
    elif "uniform_fix" in inducing_input_type:
        inducing_set_size = int(
            dtype_default(
                inducing_input_type.split("uniform_fix", 1)[1].split("_", 1)[1]
            )
        )
        input_shape[0] = inducing_set_size
        inducing_inputs = jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
    elif inducing_input_type == "training":
        inducing_inputs = x_batch_permuted[:n_inducing_inputs]
    elif "uniform_train_fix_" in inducing_input_type:
        scale, _, inducing_set_size = inducing_input_type.split(
            "uniform_train_fix_", 1
        )[1].split("_")
        scale = dtype_default(scale)
        inducing_set_size = int(inducing_set_size)
        input_shape[0] = n_train
        uniform_samples = jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
        print(
            f"Inducing input selection: Using {inducing_input_type}, but scale {scale} is not used explicitly. Default is 0.5."
        )
        # inducing_inputs = jnp.zeros([2*n_train, image_dim, image_dim, 3], dtype=dtype_default)
        inducing_inputs = []
        for i in range(n_train):
            inducing_inputs.append(x_batch_permuted[i])
            inducing_inputs.append(uniform_samples[i])
        inducing_inputs = np.array(inducing_inputs)[:inducing_set_size]
    elif "uniform_train_rand_" in inducing_input_type:
        # mix uniform samples with train data with a certain ratio
        scale = dtype_default(
            inducing_input_type.split("uniform_train_rand_", 1)[1].split("_", 1)[0]
        )
        n_inducing_inputs_sample = int(
            scale * n_inducing_inputs
        )  # TODO: fix: only allows for even n_inducing_inputs
        n_inducing_inputs_train = n_inducing_inputs - n_inducing_inputs_sample
        input_shape[0] = n_inducing_inputs_sample
        uniform_samples = jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
        training_samples = x_batch_permuted[:n_inducing_inputs_train]
        inducing_inputs = np.concatenate([uniform_samples, training_samples], 0)
    elif "train_pixel_rand" in inducing_input_type:
        scale = float(inducing_input_type.split("train_pixel_rand_", 1)[1].split("_", 1)[0])
        n_inducing_inputs_sample = int(
            (1-scale) * n_inducing_inputs
        )  # TODO: fix: only allows for even n_inducing_inputs
        n_inducing_inputs_train = n_inducing_inputs - n_inducing_inputs_sample
        training_samples = x_batch_permuted[:n_inducing_inputs_train]
        if len(input_shape) == 4 and input_shape[-1] == 1:
            # Select random pixel values
            random_pixels = jax.random.choice(
                a=x_batch_permuted.flatten(),
                shape=(n_inducing_inputs_sample,),
                replace=False,
                key=rng_key,
            )
            pixel_samples = jnp.transpose(
                random_pixels * jnp.ones(input_shape), (3, 1, 2, 0)
            )
        elif len(input_shape) == 4 and input_shape[-1] > 1:
            image_dim = input_shape[1]
            num_channels = input_shape[-1]
            pixel_samples_list = []
            for channel in range(num_channels):
                # Select random pixel values for given channel
                random_pixels = jax.random.choice(
                    a=x_batch_permuted[:, :, :, channel].flatten(),
                    shape=(n_inducing_inputs_sample,),
                    replace=False,
                    key=rng_key,
                )
                _pixel_samples = jnp.transpose(random_pixels * jnp.ones([1, image_dim, image_dim, 1]), (3, 1, 2, 0))
                pixel_samples_list.append(_pixel_samples)
            pixel_samples = jnp.concatenate(pixel_samples_list, axis=3)
        else:
            # Select random pixel values
            random_pixels = jax.random.choice(
                a=x_batch_permuted.flatten(),
                shape=(n_inducing_inputs_sample,),
                replace=False,
                key=rng_key,
            )[:, None]
            pixel_samples = random_pixels * jnp.ones(input_shape[-1])
        inducing_inputs = jnp.concatenate([pixel_samples, training_samples], 0)
    elif "gauss_uniform_train" in inducing_input_type:
        scale = dtype_default(
            inducing_input_type.split("gauss_uniform_train_", 1)[1].split("_", 1)[0]
        )
        n_inducing_inputs_sample = int(
            scale * n_inducing_inputs
        )  # TODO: fix: only allows for even n_inducing_inputs
        n_inducing_inputs_train = n_inducing_inputs - n_inducing_inputs_sample
        input_shape[0] = n_inducing_inputs_sample
        variant = jax.random.bernoulli(rng_key, p=0.5)
        if variant == 0:
            samples = jax.random.uniform(
                rng_key,
                input_shape,
                dtype_default,
                inducing_inputs_bound[0],
                inducing_inputs_bound[1],
            )
        else:
            normal_scale = jnp.absolute(
                jax.random.uniform(
                    rng_key,
                    [1],
                    dtype_default,
                    inducing_inputs_bound[0],
                    inducing_inputs_bound[1],
                )
            )
            samples = x_batch_permuted[:n_inducing_inputs_sample]
            samples = samples + normal_scale * random.normal(
                rng_key, samples.shape, dtype_default
            )
        training_samples = x_batch_permuted[:n_inducing_inputs_train]
        inducing_inputs = np.concatenate([samples, training_samples], 0)
    elif "train_gaussian_noise" in inducing_input_type:
        scale = dtype_default(
            inducing_input_type.split("train_gauss_", 1)[1].split("_", 1)[0]
        )
        inducing_inputs = x_batch_permuted[:n_inducing_inputs]
        inducing_inputs = inducing_inputs + scale * random.normal(
            rng_key, inducing_inputs.shape, dtype_default
        )
    elif "train_uniform_noise" in inducing_input_type:  # TODO: Test
        input_shape[0] = input_shape
        inducing_inputs = x_batch_permuted + jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
    elif inducing_input_type == "ood_rand":
        assert x_ood is not None
        permutation = np.random.permutation(x_ood.shape[0])
        x_ood_permuted = x_ood[permutation, :]
        inducing_inputs = x_ood_permuted[:n_inducing_inputs]
    elif "ood_rand_fixed" in inducing_input_type:
        assert x_ood is not None
        set_size = int(inducing_input_type.split("_")[-1])
        permutation = np.random.permutation(np.arange(set_size))
        x_ood_permuted = x_ood[:set_size, :][permutation, :]
        inducing_inputs = x_ood_permuted[:n_inducing_inputs]
    elif "uniform_ood_rand" in inducing_input_type:
        assert x_ood is not None
        scale = dtype_default(
            inducing_input_type.split("uniform_ood_rand", 1)[1].split("_", 1)[0]
        )
        n_inducing_inputs_sample = int(
            scale * n_inducing_inputs
        )  # TODO: fix: only allows for even n_inducing_inputs
        n_inducing_inputs_ood = n_inducing_inputs - n_inducing_inputs_sample
        input_shape[0] = n_inducing_inputs_sample
        uniform_samples = jax.random.uniform(
            rng_key,
            input_shape,
            dtype_default,
            inducing_inputs_bound[0],
            inducing_inputs_bound[1],
        )
        permutation = np.random.permutation(x_ood.shape[0])
        x_ood_permuted = x_ood[permutation, :]
        ood_samples = x_ood_permuted[:n_inducing_inputs_ood]
        inducing_inputs = np.concatenate([uniform_samples, ood_samples], 0)
    elif "not_specified" in inducing_input_type:
        inducing_inputs = None
    else:
        raise ValueError(
            f"Inducing point select method specified ({inducing_input_type}) is not a valid setting."
        )

    if plot_samples:
        for i in range(9):
            plt.subplot(330 + 1 + i)
            plt.imshow(inducing_inputs[i])
        if getpass.getuser() == "timner":
            plt.show()
        else:
            plt.close()

    return inducing_inputs
