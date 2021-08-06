import os
import getpass
from copy import copy
from typing import Tuple, List
import random as random_py

import jax
import tree
from jax import jit, random
from jax import numpy as jnp, partial

import tensorflow as tf
from tensorflow_probability.substrates import jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

tfd = tfp.distributions

import torch

from sklearn.metrics import roc_auc_score, roc_curve
import numpy as np

# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from baselines.diabetic_retinopathy_detection.fsvi_utils.jax_utils import KeyHelper

sns.set()

dtype_default = jnp.float32
eps = 1e-10
jitter = 1e-3


TWO_TUPLE = Tuple[int, int]
TUPLE_OF_TWO_TUPLES = Tuple[TWO_TUPLE, ...]


def initialize_random_keys(seed: int) -> KeyHelper:
    os.environ["PYTHONHASHSEED"] = str(seed)
    rng_key = jax.random.PRNGKey(seed)
    kh = KeyHelper(key=rng_key)
    random_py.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.random.manual_seed(seed)
    return kh


def _one_hot(x, k, dtype=dtype_default):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def get_minibatch(
    data: Tuple, output_dim, input_shape, prediction_type
) -> Tuple[np.ndarray, np.ndarray]:
    """
    @return:
        x: input data
        y: 2D array of shape (batch_dim, output_dim) if classification
    """
    x, y = data
    if prediction_type == "regression":
        x_batch = np.array(x)
        y_batch = np.array(y)
    elif prediction_type == "classification":
        if len(input_shape) <= 2:
            x_batch = np.reshape(x, [x.shape[0], -1])
        elif len(input_shape) == 4 and len(x.shape) != 4:  # handles flattened image inputs
            x_batch = np.array(x).reshape(x.shape[0], input_shape[1], input_shape[2], input_shape[3])
        else:
            if x.shape[1] != x.shape[2]:
                x_batch = np.array(x).transpose([0, 2, 3, 1])
            else:
                x_batch = np.array(x)

        assert len(y.shape) == 1, "the label is supposed to have only one dimension"
        y_batch = _one_hot(np.array(y), output_dim)
    else:
        ValueError("Unknown prediction type")

    return x_batch, y_batch


@jit
def sigma_transform(params_log_var):
    return tree.map_structure(lambda p: jnp.exp(p), params_log_var)


@partial(jit, static_argnums=(4,))
def kl_multioutput(mean_q, mean_p, cov_q, cov_p, full_cov: bool):
    """
    Return KL(q || p)

    @param mean_q: array of shape (batch_dim, output_dim)
    @param mean_p: array of shape (batch_dim, output_dim)
    @param cov_q: array of shape (batch_dim, output_dim, batch_dim, output_dim)
        or (batch_dim, batch_dim, output_dim)
    @param cov_p: array of shape (batch_dim, output_dim, batch_dim, output_dim)
        or (batch_dim, batch_dim, output_dim)
    @param full_cov: if True, use full covariance, otherwise, use diagonal of covariance
    @return:
    """
    function_KL = 0
    ndim = cov_q.ndim
    for i in range(cov_q.shape[-1]):
        cov_q_tp = cov_q[:, i, :, i] if ndim == 4 else cov_q[:, :, i]
        mean_q_tp = mean_q[:, i]
        cov_p_tp = cov_p[:, i, :, i] if ndim == 4 else cov_p[:, :, i]
        mean_p_tp = mean_p[:, i]
        kl = kl_general(mean_q_tp, mean_p_tp, cov_q_tp, cov_p_tp, full_cov)
        function_KL += kl
    return function_KL


@partial(
    jit,
    static_argnums=(
        4,
        5,
        6,
    ),
)
def kl_divergence(
    mean_q, mean_p, cov_q, cov_p, output_dim: int, full_cov: bool, prior_type: str
):
    """
    Return KL(q || p)
    # TODO: make the shape of cov be independent of prior_type, so that the code can be simplified
    If prior_type is bnn_induced, then this function is equivalent to kl_multioutput

    @param mean_q: array of shape (batch_dim, output_dim)
    @param cov_q: array of shape (batch_dim, output_dim, batch_dim, output_dim)
    """
    function_kl = 0
    if prior_type == "bnn_induced" or prior_type == "rbf":
        full_cov_prior = True
    else:
        full_cov_prior = False

    if output_dim == 1:
        mean_q_tp = jnp.squeeze(mean_q)
        cov_q_tp = jnp.squeeze(cov_q)

        mean_p_tp = jnp.squeeze(mean_p)
        if prior_type == "bnn_induced" or prior_type == "map_induced":
            cov_p_tp = _slice_cov_diag(cov=cov_p, index=0)
        else:
            # prior_type can be "empirical_mean", "map_mean" or other
            cov_p_tp = jnp.squeeze(cov_p)

        if cov_q_tp.shape[0] != cov_p_tp.shape[0]:
            mean_p_tp =mean_p_tp[:mean_q_tp.shape[0]]
            cov_p_tp = cov_p_tp[:cov_q_tp.shape[0]]

        function_kl = kl_general(
            mean_q_tp,
            mean_p_tp,
            cov_q_tp,
            cov_p_tp,
            full_cov and full_cov_prior,
        )
    else:
        for i in range(output_dim):
            mean_q_tp = jnp.squeeze(mean_q[:, i])
            cov_q_tp = _slice_cov_diag(cov=cov_q, index=i)
            if prior_type == "bnn_induced" or prior_type == "map_induced":
                mean_p_tp = jnp.squeeze(mean_p[:, i])
                cov_p_tp = _slice_cov_diag(cov=cov_p, index=i)
                cov_p_tp = jnp.diag(cov_p_tp)  # TODO: check this
            elif prior_type == "empirical_mean" or prior_type == "map_mean":
                mean_p_tp = jnp.squeeze(mean_p[:, i])
                cov_p_tp = jnp.squeeze(cov_p)
            else:
                mean_p_tp = jnp.squeeze(mean_p)
                cov_p_tp = jnp.squeeze(cov_p)

            function_kl += kl_general(
                mean_q_tp,
                mean_p_tp,
                cov_q_tp,
                cov_p_tp,
                full_cov and full_cov_prior,
            )

    return function_kl


@jit
def kl_diag(mean_q, mean_p, cov_q, cov_p) -> jnp.ndarray:
    """
    All inputs are 1D arrays.

    @param cov_q: the diagonal of covariance
    @return:
        a scalar
    """
    # assert (
    #     cov_q.ndim == cov_p.ndim <= 1
    # ), f"cov_q.shape={cov_q.shape}, cov_p.shape={cov_p.shape}"
    kl_1 = jnp.log(cov_p ** 0.5) - jnp.log(cov_q ** 0.5)
    kl_2 = (cov_q + (mean_q - mean_p) ** 2) / (2 * cov_p)
    kl_3 = -1 / 2
    kl = jnp.sum(kl_1 + kl_2 + kl_3)
    # Experimental: logsumexp as supremum approximation
    # kl = jnp.log(jnp.sum(jnp.exp(kl_1 + kl_2 + kl_3)))
    # Experimental: KL between two univariate Laplace distributions:
    # kl_1 = cov_q * jnp.exp(-(jnp.abs(mean_q - mean_p) / cov_q)) / cov_p
    # kl_2 = jnp.abs(mean_q - mean_p) / cov_p
    # kl_3 = jnp.log(cov_p) - jnp.log(cov_q) - 1
    # kl = jnp.sum(kl_1 + kl_2 + kl_3)
    return kl


@partial(jit, static_argnums=(4,))
def kl_general(
    mean_q: jnp.ndarray,
    mean_p: jnp.ndarray,
    cov_q: jnp.ndarray,
    cov_p: jnp.ndarray,
    full_cov: bool,
):
    """
    Return KL(q || p)

    @param mean_q: 1D array
    @param mean_p: 1D array
    @param cov_q: 1D or 2D array
    @param cov_p: 1D or 2D array
    @param full_cov: if True, use full covariance, otherwise, use diagonal of covariance
    """
    # assert cov_p.ndim in {1, 2} and cov_q.ndim in {
    #     1,
    #     2,
    # }, f"cov_q.shape={cov_q.shape}, cov_p.shape={cov_p.shape}"
    if not full_cov:
        to_1D = lambda x: jnp.diag(x) if x.ndim == 2 else x
        cov_p, cov_q = list(map(to_1D, [cov_p, cov_q]))
        return kl_diag(mean_q, mean_p, cov_q, cov_p)
    else:
        return kl_full_cov(mean_q, mean_p, cov_q, cov_p)


def kl_full_cov(
    mean_q: jnp.ndarray,
    mean_p: jnp.ndarray,
    cov_q: jnp.ndarray,
    cov_p: jnp.ndarray,
):
    q = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_q,
        covariance_matrix=cov_q,
        validate_args=False,
        allow_nan_stats=False,
    )
    p = tfp.distributions.MultivariateNormalFullCovariance(
        loc=mean_p,
        covariance_matrix=cov_p,
        validate_args=False,
        allow_nan_stats=False,
    )
    kl = tfd.kl_divergence(q, p, allow_nan_stats=False)
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
        scale = dtype_default(
            inducing_input_type.split("train_pixel_rand_", 1)[1].split("_", 1)[0]
        )
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
        inducing_inputs = np.concatenate([pixel_samples, training_samples], 0)
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


def to_image(fig):
    """Create image from plot."""
    fig.tight_layout(pad=1)
    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(
        fig.canvas.get_width_height()[::-1] + (3,)
    )
    return image_from_plot


def predictive_entropy(predicted_labels):
    entropy = -((predicted_labels + eps) * jnp.log(predicted_labels + eps)).sum(-1)
    return entropy


def predictive_entropy_logits(predicted_logits):
    entropy = -(
        jax.nn.softmax(predicted_logits, axis=-1)
        * jax.nn.log_softmax(predicted_logits, axis=-1)
    ).sum(-1)
    return entropy


def auroc(predicted_labels_test, predicted_labels_ood, score):
    ood_size = predicted_labels_ood.shape[1]
    test_size = predicted_labels_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = predictive_entropy(predicted_labels_test.mean(0))
        entropy_ood = predictive_entropy(predicted_labels_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = predictive_entropy(predicted_labels_test).mean(0)
        entropy_ood = predictive_entropy(predicted_labels_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = np.mean(
            np.mean(
                np.square(
                    predicted_labels_test - predicted_labels_test.mean(0),
                    dtype=dtype_default,
                ),
                0,
                dtype=dtype_default,
            ),
            -1,
            dtype=dtype_default,
        )
        mutual_information_ood = np.mean(
            np.mean(
                np.square(
                    predicted_labels_ood - predicted_labels_ood.mean(0),
                    dtype=dtype_default,
                ),
                0,
                dtype=dtype_default,
            ),
            -1,
            dtype=dtype_default,
        )
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    # elif score=='mutual information':
    #     predictive_variance_test = -(predicted_labels_test * jnp.log(predicted_labels_test + eps)).sum(-1).mean(0)
    #     predictive_variance_ood = -(predicted_labels_ood * jnp.log(predicted_labels_ood + eps)).sum(-1).mean(0)
    #     scores = jnp.concatenate((predictive_variance_test, predictive_variance_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


def auroc_logits(predicted_logits_test, predicted_logits_ood, score):
    predicted_labels_test = jax.nn.softmax(predicted_logits_test, axis=-1)
    predicted_labels_ood = jax.nn.softmax(predicted_logits_ood, axis=-1)

    ood_size = predicted_labels_ood.shape[1]
    test_size = predicted_labels_test.shape[1]
    anomaly_targets = jnp.concatenate((np.zeros(test_size), np.ones(ood_size)))
    if score == "entropy":
        entropy_test = predictive_entropy(predicted_labels_test.mean(0))
        entropy_ood = predictive_entropy(predicted_labels_ood.mean(0))
        scores = jnp.concatenate((entropy_test, entropy_ood))
    if score == "expected entropy":
        entropy_test = predictive_entropy(predicted_labels_test).mean(0)
        entropy_ood = predictive_entropy(predicted_labels_ood).mean(0)
        scores = jnp.concatenate((entropy_test, entropy_ood))
    elif score == "mutual information":
        mutual_information_test = np.mean(
            np.mean(
                np.square(
                    predicted_labels_test - predicted_labels_test.mean(0),
                    dtype=dtype_default,
                ),
                0,
                dtype=dtype_default,
            ),
            -1,
            dtype=dtype_default,
        )
        mutual_information_ood = np.mean(
            np.mean(
                np.square(
                    predicted_labels_ood - predicted_labels_ood.mean(0),
                    dtype=dtype_default,
                ),
                0,
                dtype=dtype_default,
            ),
            -1,
            dtype=dtype_default,
        )
        scores = jnp.concatenate((mutual_information_test, mutual_information_ood))
    # elif score=='mutual information':
    #     predictive_variance_test = -(predicted_labels_test * jnp.log(predicted_labels_test + eps)).sum(-1).mean(0)
    #     predictive_variance_ood = -(predicted_labels_ood * jnp.log(predicted_labels_ood + eps)).sum(-1).mean(0)
    #     scores = jnp.concatenate((predictive_variance_test, predictive_variance_ood))
    else:
        NotImplementedError
    fpr, tpr, _ = roc_curve(anomaly_targets, scores)
    auroc_score = roc_auc_score(anomaly_targets, scores)
    return auroc_score


def _slice_cov_diag(cov: jnp.ndarray, index: int) -> jnp.ndarray:
    """
    This function slices and takes diagonal

    index is for the output dimension
    """
    ndims = len(cov.shape)
    if ndims == 2:
        cov_i = cov[:, index]
    elif ndims == 3:
        cov_i = cov[:, :, index]
    elif ndims == 4:
        cov_i = cov[:, index, :, index]
    else:
        raise ValueError("Posterior covariance shape not recognized.")
    return cov_i


@jit
def kl_diag_tfd(mean_q, mean_p, cov_q, cov_p) -> jnp.ndarray:
    """
    Return KL(q || p)
    All inputs are 1D array
    """
    q = tfd.MultivariateNormalDiag(loc=mean_q, scale_diag=(cov_q ** 0.5))
    p = tfd.MultivariateNormalDiag(loc=mean_p, scale_diag=(cov_p ** 0.5))
    return tfd.kl_divergence(q, p)
    # Experimental: KL between two univariate Laplace distributions:
    # kl_1 = cov_q * jnp.exp(-(jnp.abs(mean_q - mean_p) / cov_q)) / cov_p
    # kl_2 = jnp.abs(mean_q - mean_p) / cov_p
    # kl_3 = jnp.log(cov_p) - jnp.log(cov_q) - 1
    # kl = jnp.sum(kl_1 + kl_2 + kl_3)
    # return kl


def to_float_if_possible(x):
    try:
        return float(x)
    except ValueError:
        return x
