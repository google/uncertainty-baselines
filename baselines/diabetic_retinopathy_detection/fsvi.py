import os
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.85"
import pathlib
from datetime import datetime
from pprint import pformat

import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")
print('WARNING: TensorFlow is set to only use CPU.')
import json
import logging
import pickle
import time
import types
from functools import partial
import sys

import haiku as hk
import jax
import optax
import tree
import numpy as np
from absl import app, flags
from jax import jit
from jax import random
from tqdm import tqdm
import jax.numpy as jnp
from tensorboard.plugins.hparams import api as hp
import wandb

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils import (
    initialize_random_keys,
    to_one_hot,
)
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils_training import Training
from baselines.diabetic_retinopathy_detection import utils

# original flags
flags.DEFINE_string(
    "data_training",
    "dr",
    "Training and in-distribution dataset used (default: not_specified)\n"
    "Examples: 'continual_learning_pmnist', 'continual_learning_smnist', "
    "'continual_learning_sfashionmnist'",
)


flags.DEFINE_bool(
    "use_map_loss",
    False,
    "If True, this script reproduces deterministic training",
)

flags.DEFINE_string("optimizer", "sgd", "Optimizer used (default: adam)")

flags.DEFINE_string(
    "activation",
    "relu",
    "Activation function used in NN (default: not_specified)",
)

flags.DEFINE_string("prior_mean", "0", "Prior mean function (default: 0)")

flags.DEFINE_string("prior_cov", "15.359475558718179",
                    help="Prior cov function (default: 0)")

flags.DEFINE_integer(
    "epochs", default=90, help="Number of epochs for each task (default: 100)",
)

flags.DEFINE_float(
    "base_learning_rate", default=0.02145079969396404,
    help="Learning rate (default: top performing config on APTOS, tuned ID.)",
)

flags.DEFINE_float("dropout_rate", default=0.0, help="Dropout rate (default: 0.0)")

flags.DEFINE_integer(
    "n_inducing_inputs", default=10, help="Number of BNN inducing points (default: 0)",
)

flags.DEFINE_string(
    "inducing_input_type",
    default="train_pixel_rand_1.0",
    help="Inducing input selection method (default: not_specified)",
)

flags.DEFINE_string("kl_scale", default="normalized", help="KL scaling factor (default: 1)")

flags.DEFINE_boolean("full_cov", default=False, help="Use full covariance")

flags.DEFINE_integer(
    "n_samples", default=5, help="Number of exp log lik samples (default: 1)",
)

flags.DEFINE_float("noise_std", default=1.0, help="Likelihood variance (default: 1)")

flags.DEFINE_list(
    "inducing_inputs_bound",
    default="0,0",
    help="Inducing point range (default: [-1, 1])",
)

flags.DEFINE_integer("seed", default=0, help="Random seed (default: 0)")

flags.DEFINE_bool("map_initialization", default=False, help="MAP initialization")

flags.DEFINE_bool(
    "stochastic_linearization", default=True, help="Stochastic linearization"
)

flags.DEFINE_bool("features_fixed", default=False, help="Fixed feature maps")


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
flags.DEFINE_bool(
  'load_from_checkpoint', False, "Attempt to load from checkpoint")
flags.DEFINE_string(
    "class_reweight_mode",
    None,
    "Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,"
    "test respectively). `None` (default) will not perform any loss reweighting. "
    "`constant` will use the train proportions to reweight the binary cross "
    "entropy loss. `minibatch` will use the proportions of each minibatch to "
    "reweight the loss.",
)
flags.DEFINE_integer(
    "loss_type",
    5,
    "type of loss, see objectives.py for details",
)

# General model flags.
flags.DEFINE_integer(
    "checkpoint_interval",
    1,
    "Number of epochs between saving checkpoints. " "Use -1 to never save checkpoints.",
)

# Metric flags.
flags.DEFINE_integer("num_bins", 15, "Number of bins for ECE.")

# Learning rate / SGD flags.
flags.DEFINE_float("final_decay_factor", 1e-3, "How much to decay the LR by.")
flags.DEFINE_float("one_minus_momentum", 0.01921801498056592,
                   "Optimizer momentum.")
flags.DEFINE_string("lr_schedule", "step", "Type of LR schedule.")
flags.DEFINE_integer(
    "lr_warmup_epochs",
    1,
    "Number of epochs for a linear warmup to the initial "
    "learning rate. Use 0 to do no warmup.",
)
flags.DEFINE_float("lr_decay_ratio", 0.2, "Amount to decay learning rate.")
flags.DEFINE_list("lr_decay_epochs", ["30", "60"], "Epochs to decay learning rate by.")

# Accelerator flags.
flags.DEFINE_integer("num_cores", 4, "Number of TPU cores or number of GPUs.")
flags.DEFINE_integer(
    "n_samples_test", 5, "Number of MC samples used for validation and testing",
)

# Parameter Initialization flags.
flags.DEFINE_float(
    "uniform_init_minval",
    -20.0,
    "lower bound of uniform distribution for variational log variance",
)
flags.DEFINE_float(
    "uniform_init_maxval",
    -18.0,
    "lower bound of uniform distribution for variational log variance",
)
flags.DEFINE_string(
    "w_init",
    "uniform",
    "initializer for weights (he_normal or uniform)",
)
flags.DEFINE_string(
    "b_init",
    "uniform",
    "initializer for bias (zeros or uniform)",
)
flags.DEFINE_string(
    "init_strategy",
    "uniform",
    "if init_strategy==he_normal_and_zeros, then w_init=he_normal, b_init=zeros,"
    "if init_strategy==uniform, then w_init=uniform, b_init=uniform",
)
flags.DEFINE_float('l2', 0.00015793214082680183,
                   'L2 regularization coefficient.')

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


flags.DEFINE_integer(
    "layer_to_linearize",
    1,
    "The layer number to use",
)
flags.DEFINE_float(
    "regularization", default=0, help="Regularization parameter (default: 0)",
)
FLAGS = flags.FLAGS


def process_args():
    """
    This is the only place where it is allowed to modify kwargs

    This function should not have side-effect.

    @param flags: input arguments
    @return:
    """
    # FLAGS doesn't accept renaming!
    FLAGS.inducing_inputs_bound = [float(v) for v in FLAGS.inducing_inputs_bound]


def get_dict_of_flags():
    return {k: getattr(FLAGS, k) for k in dir(FLAGS)}


def write_flags(path):
    d = get_dict_of_flags()
    string = json.dumps(d, indent=4, separators=(",", ":"))
    tf.io.gfile.makedirs(os.path.dirname(path))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(string)


def main(argv):
    del argv
    process_args()

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

    write_flags(os.path.join(FLAGS.output_dir, "flags.txt"))

    # Log Run Hypers
    hypers_dict = {
        'per_core_batch_size': FLAGS.per_core_batch_size,
        'base_learning_rate': FLAGS.base_learning_rate,
        'final_decay_factor': FLAGS.final_decay_factor,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
        'loss_type': FLAGS.loss_type,
        'stochastic_linearization': FLAGS.stochastic_linearization
    }
    logging.info('Hypers:')
    logging.info(pformat(hypers_dict))

    from jax.lib import xla_bridge
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
    input_shape = [1] + utils.load_input_shape(dataset_train=datasets["train"])
    # TODO: remove this hardcoded value
    output_dim = 2
    # TODO: do we really need to keep all the inducing inputs selection methods?
    n_train = steps["train"] * per_core_batch_size
    dataset_train = datasets['train']
    train_steps_per_epoch = steps["train"]

    # Get the wrapper function which will produce uncertainty estimates for
    # our choice of method and Y/N ensembling.
    uncertainty_estimator_fn = utils.get_uncertainty_estimator(
        "fsvi", use_ensemble=False, use_tf=False)

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        n_batches=train_steps_per_epoch,
        full_ntk=False,
        batch_size=per_core_batch_size,
        # TODO: is there a better way than this?
        **get_dict_of_flags(),
    )

    # INITIALIZE MODEL
    (model, _, apply_fn, state, params) = training.initialize_model(rng_key=rng_key)

    # INITIALIZE OPTIMIZATION
    (
        opt,
        opt_state,
        get_trainable_params,
        get_variational_and_model_params,
        objectives,
        loss,
        kl_evaluation,
        log_likelihood_evaluation,
        nll_grad_evaluation,
        task_evaluation,
        prediction_type,
    ) = training.initialize_optimization(
        model=model,
        params_init=params,
    )

    summary_writer = tf.summary.create_file_writer(
        os.path.join(output_dir, "summaries")
    )

    latest_checkpoint = None  # get_latest_fsvi_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint and FLAGS.load_from_checkpoint:
        with tf.io.gfile.GFile(latest_checkpoint, mode="rb") as f:
            chkpt = pickle.load(f)
        # TODO: need to validate the chkpt has compatible hyperparameters, such as
        # the type of the model, optimizer
        state, params, opt_state = chkpt["state"], chkpt["params"], chkpt["opt_state"]

        logging.info("Loaded checkpoint %s", latest_checkpoint)
        initial_epoch = chkpt["epoch"] + 1

    # INITIALIZE KL INPUT FUNCTIONS
    inducing_input_fn, prior_fn = training.kl_input_functions(
        prior_mean=FLAGS.prior_mean,
        prior_cov=FLAGS.prior_cov,
    )

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
    @jit
    def update_fn(
        params, state, opt_state, x_batch, y_batch, inducing_inputs, rng_key,
    ):
        """
        Captured variables:
            loss, FLAGS, num_cores
        """
        trainable_params, non_trainable_params = get_trainable_params(params)
        variational_params, model_params = get_variational_and_model_params(params)
        prior_mean, prior_cov = prior_fn(
            inducing_inputs=inducing_inputs, model_params=model_params,
        )

        grads, additional_info = jax.grad(loss, argnums=0, has_aux=True)(
            trainable_params,
            non_trainable_params,
            state,
            prior_mean,
            prior_cov,
            x_batch,
            y_batch,
            inducing_inputs,
            rng_key,
            FLAGS.class_reweight_mode == "constant",
            FLAGS.loss_type,
            FLAGS.l2,
        )

        zero_grads = jax.tree_map(lambda x: x * 0.0, non_trainable_params)
        grads = jax.tree_map(lambda x: x * 1.0, grads)
        grads_full = hk.data_structures.merge(grads, zero_grads)

        if num_cores > 1:
            grads_full = tree.map_structure(
                partial(jax.lax.pmean, axis_name="i"), grads_full
            )
            additional_info = tree.map_structure(
                partial(jax.lax.pmean, axis_name="i"), additional_info
            )
        updates, opt_state = opt.update(grads_full, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_state = additional_info["state"]
        return new_params, new_state, opt_state, additional_info

    def parallelisable_train_per_batch_computation(
        params, state, opt_state, rng_key_train, x_batch, labels,
    ):
        """
        Captured variables:
            output_dim, FLAGS, update_fn, inducing_input_fn, model
        """
        # features has shape (batch_dim, 128, 128, 3), labels has shape (batch_dim,)
        y_batch = to_one_hot(labels, output_dim)
        inducing_key, rng_key_train, rng_key_eval = jax.random.split(rng_key_train, 3)
        inducing_inputs = inducing_input_fn(
            x_batch, inducing_key, FLAGS.n_inducing_inputs
        )
        params, state, opt_state, additional_info = update_fn(
            params, state, opt_state, x_batch, y_batch, inducing_inputs, rng_key_train,
        )
        # compute metrics
        _, probs, _ = model.predict_y_multisample(
            params=params,
            state=state,
            inputs=x_batch,
            rng_key=rng_key_eval,
            n_samples=1,
            is_training=False,
        )
        return params, state, opt_state, additional_info, probs

    if num_cores > 1:
        parallelisable_train_per_batch_computation = jax.pmap(
            parallelisable_train_per_batch_computation,
            axis_name="i",
            in_axes=(None, None, None, 0, 0, 0),
            out_axes=(None, None, None, None, 0),
        )

    def parallelizable_evaluate_per_batch_computation(x_batch, labels, rng_key,
                                                      params, state, is_deterministic,):
        """
        Captured variables:
            model,
        """
        # Compute prediction, total, aleatoric, and epistemic
        # uncertainty estimates
        pred_and_uncert = uncertainty_estimator_fn(
            x_batch, model, training_setting=False, params=params, state=state,
        num_samples=FLAGS.n_samples_test, rng_key=rng_key,)

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
                images, labels, keys, params, state, is_deterministic,)
            reshaped_arrays_dict = {name: reshape_to_one_core(array) for name, array in arrays_dict.items()}
            list_of_results.append(reshaped_arrays_dict)

        results_arrs = merge_list_of_dicts(list_of_results)
        return results_arrs

    # it is important to define this variable `estimator_args` here instead
    # of declaring it in the loop. This is to avoid JAX memory leak.
    estimator_args = {
        "rng_key": rng_key_test,
        "params": params,
        "state": state,
    }

    ########################## train-eval loop ##########################

    verbose = False

    print(f"\n--- Training for {FLAGS.epochs} epochs ---\n")
    train_iterator = iter(dataset_train)
    for epoch in range(initial_epoch, FLAGS.epochs):
        t0 = time.time()
        for _ in tqdm(range(train_steps_per_epoch), desc="gradient steps..."):
            if verbose:
                start = time.time()
            data = next(train_iterator)
            x_batch, labels = data["features"]._numpy(), data["labels"]._numpy()
            orig_labels = labels
            if verbose:
                print(f"loading data used {time.time() - start:.2f} seconds")
                start = time.time()
            _, rng_key_train = random.split(rng_key_train)
            if num_cores > 1:
                keys = random.split(rng_key_train, num_cores)
                x_batch, labels = reshape_to_multiple_cores(x_batch, labels, num_cores)
            else:
                keys = rng_key_train
            (
                params,
                state,
                opt_state,
                additional_info,
                probs,
            ) = parallelisable_train_per_batch_computation(
                params, state, opt_state, keys, x_batch, labels,
            )

            if verbose:
                print(f"per-batch computation used {time.time() - start:.2f} seconds")
                start = time.time()

            if num_cores > 1:
                probs = reshape_to_one_core(probs)

            if verbose:
                print(f"reshape again used {time.time() - start:.2f} seconds")
                start = time.time()

            metrics["train/loss"].update_state(additional_info["loss"].item())
            log_likelihood_per_input = (
                additional_info["log_likelihood"].item() / orig_labels.shape[0]
            )
            metrics["train/negative_log_likelihood"].update_state(
                -log_likelihood_per_input
            )
            probs_of_labels = probs[:, 1]
            metrics["train/accuracy"].update_state(orig_labels, probs_of_labels)
            metrics["train/auprc"].update_state(orig_labels, probs_of_labels)
            metrics["train/auroc"].update_state(orig_labels, probs_of_labels)

            if not use_tpu:
                metrics["train/ece"].add_batch(probs_of_labels, label=orig_labels)
            if verbose:
                print(f"compute metric used {time.time() - start:.2f} seconds")

        _, rng_key_test = random.split(rng_key_test)

        estimator_args["rng_key"] = rng_key_test
        estimator_args["params"] = params
        estimator_args["state"] = state

        per_pred_results, total_results = utils.evaluate_model_and_compute_metrics(
            None, eval_datasets, steps, metrics, None,
            None, per_core_batch_size, available_splits,
            estimator_args=estimator_args, is_deterministic=False, num_bins=FLAGS.num_bins,
            use_tpu=use_tpu, backend="jax", eval_step_jax=eval_step_jax, return_per_pred_results=True,
            call_dataset_iter=False)

        # Optionally log to wandb
        if FLAGS.use_wandb:
            wandb.log(total_results, step=epoch)

        with summary_writer.as_default():
            for name, result in total_results.items():
                if result is not None:
                    tf.summary.scalar(name, result, step=epoch + 1)

        for metric in metrics.values():
            metric.reset_states()

        if (
            FLAGS.checkpoint_interval > 0
            and (epoch + 1) % FLAGS.checkpoint_interval == 0
        ):
            hparams = {
                k: v
                for k, v in get_dict_of_flags().items()
                if not isinstance(v, types.GeneratorType)
            }
            to_save = {
                "params": params,
                "state": state,
                # TODO: improve this way of saving hyperparameters
                # TODO: figure out why the figsize has type generator
                "hparams": hparams,
                "opt_state": opt_state,
                "epoch": epoch,
            }

            chkpt_name = os.path.join(output_dir, f"chkpt_{epoch + 1}")
            with tf.io.gfile.GFile(chkpt_name, mode="wb") as f:
                pickle.dump(to_save, f)
            logging.info("Saved checkpoint to %s", chkpt_name)

            # Save per-prediction metrics
            utils.save_per_prediction_results(
                output_dir, epoch + 1, per_pred_results, verbose=False)

        T0 = time.time()
        print(f"Epoch {epoch} used {T0 - t0:.2f} seconds")

    hparams = {
        k: v
        for k, v in get_dict_of_flags().items()
        if not isinstance(v, types.GeneratorType)
    }
    to_save = {
        "params": params,
        "state": state,
        # TODO: improve this way of saving hyperparameters
        # TODO: figure out why the figsize has type generator
        "hparams": hparams,
        "opt_state": opt_state,
        "epoch": FLAGS.epochs,
    }
    final_checkpoint_name = os.path.join(output_dir, "final_checkpoint")
    with tf.io.gfile.GFile(final_checkpoint_name, mode="wb") as f:
        pickle.dump(to_save, f)
    logging.info("Saved last checkpoint to %s", final_checkpoint_name)

    # Save per-prediction metrics
    utils.save_per_prediction_results(
        output_dir, FLAGS.epochs, per_pred_results, verbose=False)

    with summary_writer.as_default():
        hp.hparams(
            {
                "base_learning_rate": FLAGS.base_learning_rate,
                'per_core_batch_size': FLAGS.per_core_batch_size,
                'final_decay_factor': FLAGS.final_decay_factor,
                'one_minus_momentum': FLAGS.one_minus_momentum,
                'optimizer': FLAGS.optimizer,
                'loss_type': FLAGS.loss_type,
                'l2': FLAGS.l2
            }
        )
    if wandb_run is not None:
        wandb_run.finish()


@jax.jit
def reshape_to_one_core(x):
    return jnp.reshape(x, [-1] + list(x.shape[2:]))


@partial(jax.jit, static_argnums=(2,))
def reshape_to_multiple_cores(x_batch, y_batch, num_cores):
    func = lambda x: x.reshape([num_cores, -1] + list(x.shape)[1:])
    x_batch, y_batch = list(map(func, [x_batch, y_batch],))
    return x_batch, y_batch


def get_latest_fsvi_checkpoint(path):
    chkpts = [f for f in tf.io.gfile.listdir(path) if "chkpt_" in f]
    if chkpts:
        latest = max(chkpts, key=lambda x: int(x.split("_")[1]))
        return os.path.join(path, latest)


def merge_list_of_dicts(list_of_dicts):
    if not list_of_dicts:
        return list_of_dicts
    keys = list_of_dicts[0].keys()
    merged_dict = {k: jnp.stack([d[k] for d in list_of_dicts]) for k in keys}
    return merged_dict


def find_diff(x, y):
    difference = tree.map_structure(lambda a, b: jnp.abs(a - b).max(),
                                    x,
                                    y)
    difference = np.max([x for x in tree.flatten(difference)])
    return difference


def find_diff_dict(x, y):
    for k in x.keys():
        print(k, find_diff(x[k], y[k]))


def deepcopy_np(d):
    return {k: tree.map_structure(lambda a: np.array(a), v) for k, v in d.items()}


def jax_sum(x):
    return np.sum([jnp.sum(a) for a in tree.flatten(x)])


if __name__ == "__main__":
    app.run(main)
