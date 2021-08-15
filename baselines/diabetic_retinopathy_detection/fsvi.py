import json
import logging
import os
import pdb
import pickle
import time
import types
from functools import partial

import haiku as hk
import jax
import optax
import tree
import numpy as np
from absl import app, flags
from jax import jit
from jax import random
from tensorflow_probability.substrates import jax as tfp
import tensorflow as tf
from tqdm import tqdm
import jax.numpy as jnp
from tensorboard.plugins.hparams import api as hp

from baselines.diabetic_retinopathy_detection.fsvi_utils.objectives import Objectives_hk
from baselines.diabetic_retinopathy_detection.utils import log_epoch_metrics, get_latest_fsvi_checkpoint

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
path = dname + "/../.."
# print(f'Setting working directory to {path}\n')
os.chdir(path)

from baselines.diabetic_retinopathy_detection.fsvi_utils import datasets
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils import (
    get_minibatch,
    initialize_random_keys,
)
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils_training import Training
from baselines.diabetic_retinopathy_detection import utils


tfd = tfp.distributions


# original flags
flags.DEFINE_string(
    "data_training",
    "not_specified",
    "Training and in-distribution dataset used (default: not_specified)\n"
    "Examples: 'continual_learning_pmnist', 'continual_learning_smnist', "
    "'continual_learning_sfashionmnist'",
)


flags.DEFINE_string(
    "model_type",
    "not_specified",
    "Model used (default: not_specified). Example: 'fsvi_mlp', 'mfvi_cnn'",
)

flags.DEFINE_string("optimizer", "adam", "Optimizer used (default: adam)")

flags.DEFINE_string(
    "architecture", "not_specified", "Architecture of NN (default: not_specified)",
)

flags.DEFINE_string(
    "activation",
    "not_specified",
    "Activation function used in NN (default: not_specified)",
)

flags.DEFINE_string("prior_mean", "0", "Prior mean function (default: 0)")

flags.DEFINE_string("prior_cov", "0", help="Prior cov function (default: 0)")

flags.DEFINE_string(
    "prior_type",
    default="not_specified",
    help="Type of prior (default: not_specified)",
)

flags.DEFINE_integer(
    "epochs", default=100, help="Number of epochs for each task (default: 100)",
)

flags.DEFINE_integer(
    "train_batch_size",
    default=100,
    help="Per-core batch size to use for training (default: 100)",
)

flags.DEFINE_float(
    "learning_rate", default=1e-3, help="Learning rate (default: 1e-3)",
)

flags.DEFINE_float("dropout_rate", default=0.0, help="Dropout rate (default: 0.0)")

flags.DEFINE_float(
    "regularization", default=0, help="Regularization parameter (default: 0)",
)

flags.DEFINE_integer(
    "n_inducing_inputs", default=0, help="Number of BNN inducing points (default: 0)",
)

flags.DEFINE_string(
    "inducing_input_type",
    default="not_specified",
    help="Inducing input selection method (default: not_specified)",
)

flags.DEFINE_string("kl_scale", default="1", help="KL scaling factor (default: 1)")

flags.DEFINE_boolean("full_cov", default=False, help="Use full covariance")

flags.DEFINE_integer(
    "n_samples", default=1, help="Number of exp log lik samples (default: 1)",
)

flags.DEFINE_float("tau", default=1.0, help="Likelihood precision (default: 1)")

flags.DEFINE_float("noise_std", default=1.0, help="Likelihood variance (default: 1)")

flags.DEFINE_list(
    "inducing_inputs_bound",
    default="-1.,1.",
    help="Inducing point range (default: [-1, 1])",
)

flags.DEFINE_integer(
    "logging_frequency",
    default=10,
    help="Logging frequency in number of epochs (default: 10)",
)

flags.DEFINE_list(
    "figsize", default="10,4", help="Size of figures (default: (10, 4))",
)

flags.DEFINE_integer("seed", default=0, help="Random seed (default: 0)")

flags.DEFINE_string(
    "save_path", default="debug", help="Path to save results (default: debug)",
)

flags.DEFINE_bool("save", default=False, help="Save output to file")

flags.DEFINE_bool("resume_training", default=False, help="Resume training")

flags.DEFINE_bool("map_initialization", default=False, help="MAP initialization")

flags.DEFINE_bool(
    "stochastic_linearization", default=False, help="Stochastic linearization"
)

# TODO: remove this option
flags.DEFINE_bool("batch_normalization", default=False, help="Batch normalization")

flags.DEFINE_bool("linear_model", default=False, help="Linear model")

flags.DEFINE_bool("features_fixed", default=False, help="Fixed feature maps")

flags.DEFINE_bool("debug", default=False, help="Debug model")

flags.DEFINE_string(
    "logroot",
    default=None,
    help="The root result folder that store runs for this type of experiment",
)

flags.DEFINE_string(
    "subdir",
    default=None,
    help="The subdirectory in logroot/runs/ corresponding to this run",
)
flags.DEFINE_integer(
    "loader_n_batches", None, "Number of batches to use",
)
flags.DEFINE_integer(
    "eval_batch_size", 32, "Number of batches for evaluation",
)

# new flags copied from deterministic.py
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
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, val,'
    'test respectively). `None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. `minibatch` will use the proportions of each minibatch to '
    'reweight the loss.')

# General model flags.
# TODO: decide if we keep this
flags.DEFINE_integer(
    "checkpoint_interval",
    25,
    "Number of epochs between saving checkpoints. " "Use -1 to never save checkpoints.",
)

# Metric flags.
flags.DEFINE_integer("num_bins", 15, "Number of bins for ECE.")

# Accelerator flags.
flags.DEFINE_bool("force_use_cpu", False, "If True, force usage of CPU")
flags.DEFINE_bool("use_gpu", True, "Whether to run on GPU or otherwise TPU.")
flags.DEFINE_bool("use_bfloat16", False, "Whether to use mixed precision.")
flags.DEFINE_integer("num_cores", 1, "Number of TPU cores or number of GPUs.")
flags.DEFINE_string(
    "tpu",
    None,
    "Name of the TPU. Only used if force_use_cpu and use_gpu are both False.",
)
flags.DEFINE_integer(
    "n_samples_test",
    1,
    "Number of MC samples used for validation and testing",
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
    FLAGS.figsize = (int(v) for v in FLAGS.figsize)
    FLAGS.inducing_inputs_bound = [float(v) for v in FLAGS.inducing_inputs_bound]


def get_dict_of_flags():
    return {k: getattr(FLAGS, k) for k in dir(FLAGS)}


def write_flags(path):
    d = get_dict_of_flags()
    string = json.dumps(d, indent=4, separators=(",", ":"))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(string)


def main(argv):
    del argv

    write_flags(os.path.join(FLAGS.output_dir, "flags.txt"))

    from jax.lib import xla_bridge

    print("*" * 100)
    print("Platform that is used by JAX:", xla_bridge.get_backend().platform)
    print("*" * 100)

    process_args()
    kh = initialize_random_keys(seed=FLAGS.seed)
    rng_key, rng_key_train, rng_key_test = random.split(kh.next_key(), 3)

    # LOAD DATA
    train_batch_size = FLAGS.train_batch_size * FLAGS.num_cores
    eval_batch_size = FLAGS.eval_batch_size * FLAGS.num_cores

    (
        dataset_train,
        dataset_validation,
        dataset_test,
        input_shape,
        output_dim,
        n_train,
        n_valid,
        n_test,
    ) = datasets.load_data(
        train_batch_size=train_batch_size,
        data_dir=FLAGS.data_dir,
        eval_batch_size=eval_batch_size,
        use_validation=FLAGS.use_validation,
    )
    steps_per_epoch = n_train // train_batch_size
    steps_per_validation_eval = n_valid // eval_batch_size
    steps_per_test_eval = n_test // eval_batch_size

    # INITIALIZE TRAINING CLASS
    training = Training(
        input_shape=input_shape,
        output_dim=output_dim,
        n_train=n_train,
        n_batches=n_train // train_batch_size,
        full_ntk=False,
        batch_size=train_batch_size,
        # TODO: is there a better way than this?
        **get_dict_of_flags(),
    )

    # INITIALIZE MODEL
    (model, _, apply_fn, state, params) = training.initialize_model(
        rng_key=rng_key
    )

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
        apply_fn=apply_fn,
        params_init=params,
        state=state,
        rng_key=rng_key,
    )

    summary_writer = tf.summary.create_file_writer(
        os.path.join(FLAGS.output_dir, "summaries")
    )

    latest_checkpoint = get_latest_fsvi_checkpoint(FLAGS.output_dir)
    initial_epoch = 0
    if latest_checkpoint:
        with tf.io.gfile.GFile(latest_checkpoint, mode="rb") as f:
            chkpt = pickle.load(f)
        # TODO: need to validate the chkpt has compatible hyperparameters, such as
        # the type of the model, optimizer
        state, params, opt_state = chkpt["state"], chkpt["params"], chkpt["opt_state"]

        logging.info('Loaded checkpoint %s', latest_checkpoint)
        initial_epoch = chkpt["epoch"] + 1

    # INITIALIZE KL INPUT FUNCTIONS
    inducing_input_fn, prior_fn = training.kl_input_functions(
        apply_fn=apply_fn,
        predict_f_deterministic=model.predict_f_deterministic,
        state=state,
        params=params,
        prior_mean=FLAGS.prior_mean,
        prior_cov=FLAGS.prior_cov,
        rng_key=rng_key,
    )

    use_tpu = not (FLAGS.force_use_cpu or FLAGS.use_gpu)
    metrics = utils.get_diabetic_retinopathy_base_metrics(
        use_tpu=use_tpu, num_bins=FLAGS.num_bins, use_validation=FLAGS.use_validation
    )
    # Define metrics outside the accelerator scope for CPU eval.
    # This will cause an error on TPU.
    if not use_tpu:
        metrics.update(
            utils.get_diabetic_retinopathy_cpu_metrics(
                use_validation=FLAGS.use_validation
            )
        )
    metrics.update({"test/ms_per_example": tf.keras.metrics.Mean()})

    @jit
    def update(
        params, state, x_batch, y_batch, inducing_inputs, rng_key,
    ):
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
        )

        zero_grads = jax.tree_map(lambda x: x * 0.0, non_trainable_params)
        grads = jax.tree_map(lambda x: x * 1.0, grads)
        grads_full = hk.data_structures.merge(grads, zero_grads)

        if FLAGS.num_cores > 1:
            grads_full = tree.map_structure(
                partial(jax.lax.pmean, axis_name="i"), grads_full
            )
            additional_info = tree.map_structure(
                partial(jax.lax.pmean, axis_name="i"), additional_info
            )
        return grads_full, additional_info

    @jit
    def update_apply_grad(grads_full, opt_state, params, additional_info):
        updates, opt_state = opt.update(grads_full, opt_state)
        new_params = optax.apply_updates(params, updates)
        new_state = additional_info["state"]
        return new_params, opt_state, new_state, additional_info

    @jit
    def reshape(x_batch, y_batch, inducing_inputs, rng_key_train):
        x_batch, y_batch, inducing_inputs = list(
            map(
                lambda x: x.reshape([FLAGS.num_cores, -1] + list(x.shape)[1:]),
                [x_batch, y_batch, inducing_inputs],
            )
        )
        keys = jax.random.split(rng_key_train, num=FLAGS.num_cores)
        return x_batch, y_batch, inducing_inputs, keys

    update_pmap = jax.pmap(
        update, axis_name="i", in_axes=(None, None, 0, 0, 0, 0)
    )


    print(f"\n--- Training for {FLAGS.epochs} epochs ---\n")
    train_iterator = iter(dataset_train)
    for epoch in range(FLAGS.epochs):
        t0 = time.time()

        for _ in tqdm(range(initial_epoch, steps_per_epoch), desc="gradient steps..."):
            data = next(train_iterator)
            rng_key_train, _ = random.split(rng_key_train)
            # features has shape (batch_dim, 128, 128, 3), labels has shape (batch_dim,)
            features, labels = data["features"]._numpy(), data["labels"]._numpy()
            x_batch, y_batch = get_minibatch(
                (features, labels), output_dim, input_shape, prediction_type
            )
            inducing_inputs = inducing_input_fn(
                x_batch, rng_key_train, FLAGS.num_cores * FLAGS.n_inducing_inputs
            )

            if FLAGS.num_cores > 1:
                x_batch, y_batch, inducing_inputs, keys = reshape(
                    x_batch, y_batch, inducing_inputs, rng_key_train
                )
                update_to_use = update_pmap
            else:
                update_to_use = update
                keys = rng_key_train

            grads_full, additional_info = update_to_use(
                params, state, x_batch, y_batch, inducing_inputs, keys,
            )
            if FLAGS.num_cores > 1:
                # pdb.set_trace()
                grads_full = take_first_copy(
                    structure=grads_full,
                    num_cores=FLAGS.num_cores,
                )
                additional_info = take_first_copy(
                    structure=additional_info,
                    num_cores=FLAGS.num_cores,
                )
            params, opt_state, state, additional_info = update_apply_grad(
                grads_full, opt_state, params, additional_info
            )

            # TODO: in case num_cores>1, I need to deal with multiple copies of returned data

            # compute metrics
            metrics["train/loss"].update_state(-additional_info["elbo"].item())
            log_likelihood_per_input = additional_info["log_likelihood"].item() / y_batch.shape[0]
            metrics["train/negative_log_likelihood"].update_state(
                -log_likelihood_per_input
            )
            _, rng_key_eval = jax.random.split(rng_key_train)
            _, probs, _ = model.predict_y_multisample(
                params=params,
                state=state,
                inputs=features,
                rng_key=rng_key_eval,
                n_samples=1,
                is_training=False,
            )
            probs_of_labels = probs[:, 1]
            metrics["train/accuracy"].update_state(labels, probs_of_labels)
            metrics["train/auprc"].update_state(labels, probs_of_labels)
            metrics["train/auroc"].update_state(labels, probs_of_labels)

            if not use_tpu:
                metrics["train/ece"].add_batch(probs_of_labels, label=labels)

        # evaluation on validation set
        if FLAGS.use_validation:
            _, rng_key_test = jax.random.split(rng_key_test)
            evaluate_on_valid_or_test(
                dataset_split="validation",
                metrics=metrics,
                model=model,
                params=params,
                state=state,
                data_iterator=iter(dataset_validation),
                num_steps=steps_per_validation_eval,
                rng_key=rng_key_test,
                use_tpu=use_tpu,
                objectives=objectives,
                output_dim=output_dim,
                input_shape=input_shape,
                prediction_type=prediction_type,
                n_samples=FLAGS.n_samples_test,
            )
        # evaluation on test set
        _, rng_key_test = jax.random.split(rng_key_test)
        evaluate_on_valid_or_test(
            dataset_split="test",
            metrics=metrics,
            model=model,
            params=params,
            state=state,
            data_iterator=iter(dataset_test),
            num_steps=steps_per_test_eval,
            rng_key=rng_key_test,
            use_tpu=use_tpu,
            objectives=objectives,
            output_dim=output_dim,
            input_shape=input_shape,
            prediction_type=prediction_type,
            n_samples=FLAGS.n_samples_test,
        )

        log_epoch_metrics(metrics=metrics, use_tpu=use_tpu)

        total_results = {name: metric.result() for name, metric in metrics.items()}
        # Metrics from Robustness Metrics (like ECE) will return a dict with a
        # single key/value, instead of a scalar.
        total_results = {
            k: (list(v.values())[0] if isinstance(v, dict) else v)
            for k, v in total_results.items()
        }
        with summary_writer.as_default():
            for name, result in total_results.items():
                tf.summary.scalar(name, result, step=epoch + 1)

        for metric in metrics.values():
            metric.reset_states()

        T0 = time.time()
        print(f"Epoch {epoch} used {T0 - t0:.2f} seconds")

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
            # Also save Keras model, due to checkpoint.save issue
            chkpt_name = os.path.join(FLAGS.output_dir, f"chkpt_{epoch + 1}")
            with tf.io.gfile.GFile(chkpt_name, mode="wb") as f:
                pickle.dump(to_save, f)
            logging.info("Saved checkpoint to %s", chkpt_name)


    to_save = {
        "params": params,
        "state": state,
        # TODO: improve this way of saving hyperparameters
        # TODO: figure out why the figsize has type generator
        "hparams": hparams,
        "opt_state": opt_state,
        "epoch": epoch,
    }
    final_checkpoint_name = os.path.join(FLAGS.output_dir, 'final_checkpoint')
    with tf.io.gfile.GFile(final_checkpoint_name, mode="wb") as f:
        pickle.dump(to_save, f)
    logging.info('Saved last checkpoint to %s', final_checkpoint_name)

    with summary_writer.as_default():
        hp.hparams({
            'learning_rate': FLAGS.learning_rate,
        })


def take_first_copy(structure, num_cores):
    def _func(x):
        assert x.shape[0] == num_cores, f"x.shape={x.shape}, num_cores={num_cores}"
        assert jnp.abs(x[0] - x[1]).max().item() < 1e-8
        return x[0]
    return tree.map_structure(_func, structure)


def evaluate_on_valid_or_test(
    dataset_split: str,
    metrics,
    model,
    params,
    state,
    data_iterator,
    num_steps,
    rng_key,
    use_tpu: bool,
    objectives: Objectives_hk,
    output_dim,
    input_shape,
    prediction_type,
    n_samples,
):
    for _ in tqdm(range(num_steps), desc=f"evaluation on {dataset_split}"):
        data = next(data_iterator)
        features, labels = data["features"]._numpy(), data["labels"]._numpy()
        x_batch, y_batch = get_minibatch(
            (features, labels), output_dim, input_shape, prediction_type
        )
        _, rng_key = jax.random.split(rng_key)
        preds_f_samples, preds_f_mean, preds_f_var = model.predict_f_multisample_jitted(
            params=params,
            state=state,
            inputs=x_batch,
            rng_key=rng_key,
            n_samples=n_samples,
            is_training=False,
        )
        log_likelihood = objectives.crossentropy_log_likelihood(
            preds_f_samples=preds_f_samples, targets=y_batch,
            class_weight=False,
        )
        # to make it comparable to log likelihood reported in other scripts, e.g. deterministic.py
        log_likelihood_per_input = log_likelihood / y_batch.shape[0]
        probs = jax.nn.softmax(preds_f_mean, axis=-1)
        probs_of_labels = probs[:, 1]

        metrics[dataset_split + "/negative_log_likelihood"].update_state(
            -log_likelihood_per_input
        )
        metrics[dataset_split + "/accuracy"].update_state(labels, probs_of_labels)
        metrics["test/accuracy"].update_state(labels, probs_of_labels)
        metrics[dataset_split + "/auprc"].update_state(labels, probs_of_labels)
        metrics[dataset_split + "/auroc"].update_state(labels, probs_of_labels)

        if not use_tpu:
            metrics[dataset_split + "/ece"].add_batch(probs_of_labels, label=labels)


if __name__ == "__main__":
    app.run(main)
