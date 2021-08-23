import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
print('WARNING: TensorFlow is set to only use CPU.')

import pathlib
from datetime import datetime
import logging
import pickle
from functools import partial
import sys

import jax
from absl import app, flags
from jax import random
from tqdm import tqdm
import jax.numpy as jnp
import wandb
from jax.lib import xla_bridge

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils import (
    initialize_random_keys,
)
from baselines.diabetic_retinopathy_detection.fsvi_utils.networks import CNN
from baselines.diabetic_retinopathy_detection import utils

# original flags
flags.DEFINE_string(
    "model_type",
    "fsvi_cnn",
    "Model used (default: not_specified). Example: 'fsvi_mlp', 'mfvi_cnn'",
)
flags.DEFINE_string(
    "architecture", "resnet", "Architecture of NN (default: not_specified)",
)

flags.DEFINE_string(
    "activation",
    "relu",
    "Activation function used in NN (default: not_specified)",
)

flags.DEFINE_float("dropout_rate", default=0.0, help="Dropout rate (default: 0.0)")


flags.DEFINE_integer("seed", default=0, help="Random seed (default: 0)")

flags.DEFINE_bool("linear_model", default=True, help="Linear model")

flags.DEFINE_integer('per_core_batch_size', 64,
                     'The per-core batch size for both training '
                     'and evaluation.')

flags.DEFINE_string(
    "output_dir",
    "/tmp/diabetic_retinopathy_detection/deterministic",
    "The directory where the model weights and training/evaluation summaries "
    "are stored. If you aim to use these as trained models for ensemble.py, "
    "you should specify an output_dir name that includes the random seed to "
    "avoid overwriting.",
)
flags.DEFINE_string("data_dir", None, "Path to training and testing data.")

flags.DEFINE_bool("use_validation", True, "Whether to use a validation split.")
flags.DEFINE_bool('use_test', False, 'Whether to use a test split.')
flags.DEFINE_string(
    'dr_decision_threshold', 'moderate',
    ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
     "binary classification task. Only affects the APTOS dataset partitioning. "
     "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
     "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))

# Metric flags.
flags.DEFINE_integer("num_bins", 15, "Number of bins for ECE.")

# Accelerator flags.
flags.DEFINE_integer("num_cores", 4, "Number of TPU cores or number of GPUs.")
flags.DEFINE_integer(
    "n_samples_test", 5, "Number of MC samples used for validation and testing",
)

flags.DEFINE_float('l2', 0.0, 'L2 regularization coefficient.')

# OOD flags.
flags.DEFINE_string(
    'distribution_shift', None,
    ("Specifies distribution shift to use, if any."
     "aptos: loads APTOS (India) OOD validation and test datasets. "
     "  Kaggle/EyePACS in-domain datasets are unchanged."
     "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
     "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
     "  as OOD."))
flags.DEFINE_bool(
    'load_train_split', True,
    "Should always be enabled - required to load train split of the dataset.")
flags.DEFINE_bool('cache_eval_datasets', False, 'Caches eval datasets.')

# Logging and hyperparameter tuning.
flags.DEFINE_bool('use_wandb', True, 'Use wandb for logging.')
flags.DEFINE_string('wandb_dir', 'wandb', 'Directory where wandb logs go.')
flags.DEFINE_string('project', 'ub-debug', 'Wandb project name.')
flags.DEFINE_string('exp_name', None, 'Give experiment a name.')
flags.DEFINE_string('exp_group', None, 'Give experiment a group name.')

FLAGS = flags.FLAGS


def main(argv):
    del argv

    # Wandb Setup
    if FLAGS.use_wandb:
        pathlib.Path(FLAGS.wandb_dir).mkdir(parents=True, exist_ok=True)
        wandb_args = dict(
            project=FLAGS.project,
            entity="uncertainty-baselines",
            dir=FLAGS.wandb_dir,
            reinit=True,
            name=FLAGS.exp_name,
            group=FLAGS.exp_group)
        wandb_run = wandb.init(**wandb_args)
        wandb.config.update(FLAGS, allow_val_change=True)
        output_dir = os.path.join(
            FLAGS.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        wandb_run = None
        output_dir = FLAGS.output_dir

    print("*" * 100)
    print("Platform that is used by JAX:", xla_bridge.get_backend().platform)
    print("*" * 100)

    kh = initialize_random_keys(seed=FLAGS.seed)
    rng_key, rng_key_train, rng_key_test = random.split(kh.next_key(), 3)

    # LOAD DATA
    num_cores = FLAGS.num_cores

    per_core_batch_size = FLAGS.per_core_batch_size * num_cores

    datasets, steps = utils.load_dataset(
        train_batch_size=per_core_batch_size, eval_batch_size=per_core_batch_size,
        flags=FLAGS, strategy=None)
    available_splits = list(datasets.keys())
    test_splits = [split for split in available_splits if 'test' in split]
    eval_splits = [split for split in available_splits
                   if 'validation' in split or 'test' in split]
    eval_datasets = {split: iter(datasets[split]) for split in eval_splits}

    # Get the wrapper function which will produce uncertainty estimates for
    # our choice of method and Y/N ensembling.
    uncertainty_estimator_fn = utils.get_uncertainty_estimator(
        "fsvi", use_ensemble=False, use_tf=False)


    # INITIALIZE MODEL
    model = CNN(
        architecture=FLAGS.architecture,
        output_dim=2,
        activation_fn=FLAGS.activation,
        stochastic_parameters=True,
        linear_model=FLAGS.linear_model,
        dropout="dropout" in FLAGS.model_type,
        dropout_rate=FLAGS.dropout_rate,
    )

    latest_checkpoint = get_latest_fsvi_checkpoint(FLAGS.output_dir)
    assert latest_checkpoint is not None
    with tf.io.gfile.GFile(latest_checkpoint, mode="rb") as f:
        chkpt = pickle.load(f)
    state, params, opt_state = chkpt["state"], chkpt["params"], chkpt["opt_state"]
    logging.info("Loaded checkpoint %s", latest_checkpoint)

    use_tpu = any(["tpu" in str(d).lower() for d in jax.devices()])
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=use_tpu,
        num_bins=FLAGS.num_bins,
        use_validation=FLAGS.use_validation,
        available_splits=available_splits)
    # Define metrics outside the accelerator scope for CPU eval.
    # This will cause an error on TPU.
    if not use_tpu:
        metrics.update(
            utils.get_diabetic_retinopathy_cpu_metrics(
                available_splits=available_splits,
                use_validation=FLAGS.use_validation))

    for test_split in test_splits:
        metrics.update({f'{test_split}/ms_per_example': tf.keras.metrics.Mean()})

    ########################## specify functions ##########################
    def parallelizable_evaluate_per_batch_computation(x_batch, labels, rng_key,
                                                      params, state, is_deterministic, ):
        """
        Captured variables:
            model,
        """
        # Compute prediction, total, aleatoric, and epistemic
        # uncertainty estimates
        pred_and_uncert = uncertainty_estimator_fn(
            x_batch, model, training_setting=False, params=params, state=state,
            num_samples=FLAGS.n_samples_test, rng_key=rng_key, )

        result = {
            "y_true": labels,
            "y_pred": pred_and_uncert['prediction'],
            "y_pred_entropy": pred_and_uncert['predictive_entropy'],
            "y_pred_variance": pred_and_uncert['predictive_variance'],
        }

        if not is_deterministic:
            result["y_aleatoric_uncert"] = pred_and_uncert["aleatoric_uncertainty"]
            result["y_epistemic_uncert"] = pred_and_uncert["epistemic_uncertainty"]

        return result

    if num_cores > 1:
        parallelizable_evaluate_per_batch_computation = jax.pmap(
            parallelizable_evaluate_per_batch_computation,
            in_axes=(0, 0, 0, None, None, None),
            static_broadcasted_argnums=(5,))

    def eval_step_jax(
            dataset_iterator, dataset_steps, is_deterministic, params, state, rng_key,
    ):
        """
        Captured variables:
            parallelizable_evaluate_per_batch_computation, FLAGS
        """
        list_of_results = []
        for _ in tqdm(range(dataset_steps), "evaluation loop"):
            data = next(dataset_iterator)
            images = data['features']._numpy()
            labels = data['labels']._numpy()
            _, rng_key = random.split(rng_key)
            if num_cores > 1:
                images, labels = reshape_to_multiple_cores(images, labels, num_cores)
                keys = random.split(rng_key, num_cores)
            else:
                keys = rng_key
            # Compute prediction, total, aleatoric, and epistemic
            # uncertainty estimates
            arrays_dict = parallelizable_evaluate_per_batch_computation(
                images, labels, keys, params, state, is_deterministic, )
            reshaped_arrays_dict = {name: reshape_to_one_core(array) for name, array in arrays_dict.items()}
            list_of_results.append(reshaped_arrays_dict)

        results_arrs = merge_list_of_dicts(list_of_results)
        return results_arrs

    # it is important to define this variable `estimator_args` here instead
    # of declaring it in the loop. This is to avoid JAX memory leak.

    ########################## eval loop ##########################

    _, rng_key_test = random.split(rng_key_test)

    estimator_args = {
        "rng_key": rng_key_test,
        "params": params,
        "state": state,
    }

    per_pred_results, total_results = utils.evaluate_model_and_compute_metrics(
        None, eval_datasets, steps, metrics, None,
        None, per_core_batch_size, available_splits,
        estimator_args=estimator_args, is_deterministic=False, num_bins=FLAGS.num_bins,
        use_tpu=use_tpu, backend="jax", eval_step_jax=eval_step_jax, return_per_pred_results=True,
        call_dataset_iter=False)

    if wandb_run is not None:
        wandb_run.finish()


@jax.jit
def reshape_to_one_core(x):
    return jnp.reshape(x, [-1] + list(x.shape[2:]))


@partial(jax.jit, static_argnums=(2,))
def reshape_to_multiple_cores(x_batch, y_batch, num_cores):
    func = lambda x: x.reshape([num_cores, -1] + list(x.shape)[1:])
    x_batch, y_batch = list(map(func, [x_batch, y_batch], ))
    return x_batch, y_batch


def get_latest_fsvi_checkpoint(path):
    files = tf.io.gfile.listdir(path)
    if "final_checkpoint" in files:
        return os.path.join(path, "final_checkpoint")
    else:
        chkpts = [f for f in files if "chkpt_" in f]
        if chkpts:
            latest = max(chkpts, key=lambda x: int(x.split("_")[1]))
            return os.path.join(path, latest)


def merge_list_of_dicts(list_of_dicts):
    if not list_of_dicts:
        return list_of_dicts
    keys = list_of_dicts[0].keys()
    merged_dict = {k: jnp.stack([d[k] for d in list_of_dicts]) for k in keys}
    return merged_dict


if __name__ == "__main__":
    app.run(main)
