# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""FSVI."""
# pylint: disable=g-bare-generic
# pylint: disable=g-doc-args
# pylint: disable=g-doc-return-or-yield
# pylint: disable=g-importing-member
# pylint: disable=g-no-space-after-docstring-summary
# pylint: disable=g-short-docstring-punctuation
# pylint: disable=logging-format-interpolation
# pylint: disable=logging-fstring-interpolation
# pylint: disable=missing-function-docstring
import datetime
import functools
import logging
import os
import pathlib
import pickle
import pprint
import sys
import time
from typing import Any, Dict, List, Tuple
from absl import app
from absl import flags
from baselines.diabetic_retinopathy_detection import utils
from baselines.diabetic_retinopathy_detection.fsvi_utils.initializers import Initializer
from baselines.diabetic_retinopathy_detection.fsvi_utils.initializers import OptimizerInitializer
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils import initialize_random_keys
from baselines.diabetic_retinopathy_detection.fsvi_utils.utils import to_one_hot
import jax
from jax import jit
from jax import random
from jax.lib import xla_bridge
import jax.numpy as jnp
import optax
import tensorflow as tf
from tqdm import tqdm
import tree
import wandb
from tensorboard.plugins.hparams import api as hp

tf.config.experimental.set_visible_devices([], "GPU")
print("WARNING: TensorFlow is set to only use CPU.")

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, root_path)

# original flags
flags.DEFINE_string("optimizer", "sgd", "Optimizer used (default: adam)")

flags.DEFINE_string(
    "activation",
    "relu",
    "Activation function used in NN (default: not_specified)",
)

flags.DEFINE_string("prior_mean", "0", "Prior mean function (default: 0)")

flags.DEFINE_string(
    "prior_cov", "15.359475558718179", help="Prior cov function (default: 0)")

flags.DEFINE_integer(
    "epochs",
    default=90,
    help="Number of epochs for each task (default: 100)",
)

flags.DEFINE_float(
    "base_learning_rate",
    default=0.02145079969396404,
    help="Learning rate (default: top performing config on APTOS, tuned ID.)",
)

flags.DEFINE_float(
    "dropout_rate", default=0.0, help="Dropout rate (default: 0.0)")

flags.DEFINE_integer(
    "n_inducing_inputs",
    default=10,
    help="Number of BNN inducing points (default: 0)",
)

flags.DEFINE_string(
    "kl_scale", default="normalized", help="KL scaling factor (default: 1)")

flags.DEFINE_integer(
    "n_samples",
    default=5,
    help="Number of Monte-Carlo log lik samples (default: 1)",
)

flags.DEFINE_integer("seed", default=0, help="Random seed (default: 0)")

flags.DEFINE_bool(
    "stochastic_linearization", default=True, help="Stochastic linearization")

flags.DEFINE_integer(
    "per_core_batch_size",
    64,
    "The per-core batch size for both training "
    "and evaluation.",
)

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
flags.DEFINE_bool("use_test", False, "Whether to use a test split.")
flags.DEFINE_string("preproc_builder_config", "btgraham-300", (
    "Determines the preprocessing procedure for the images. Supported options: "
    "{btgraham-300, blur-3-btgraham-300, blur-5-btgraham-300, "
    "blur-10-btgraham-300, blur-20-btgraham-300}."))
flags.DEFINE_string(
    "dr_decision_threshold",
    "moderate",
    ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
     "binary classification task. Only affects the APTOS dataset partitioning. "
     "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
     "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"),
)
flags.DEFINE_bool("load_from_checkpoint", False,
                  "Attempt to load from checkpoint")
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
    "Number of epochs between saving checkpoints. "
    "Use -1 to never save checkpoints.",
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
flags.DEFINE_list("lr_decay_epochs", ["30", "60"],
                  "Epochs to decay learning rate by.")

# Accelerator flags.
flags.DEFINE_integer("num_cores", 4, "Number of TPU cores or number of GPUs.")
flags.DEFINE_integer(
    "n_samples_test",
    5,
    "Number of MC samples used for validation and testing",
)

# Parameter Initialization flags.
flags.DEFINE_float(
    "uniform_init_minval",
    -20.0,
    "lower bound of uniform distribution for log variational variance",
)
flags.DEFINE_float(
    "uniform_init_maxval",
    -18.0,
    "upper bound of uniform distribution for log variational variance",
)
flags.DEFINE_string(
    "init_strategy",
    "uniform",
    "if init_strategy==he_normal_and_zeros, then w_init=he_normal, b_init=zeros,"
    "if init_strategy==uniform, then w_init=uniform, b_init=uniform",
)
flags.DEFINE_float("l2", 0.00015793214082680183,
                   "L2 regularization coefficient.")

# OOD flags.
flags.DEFINE_string(
    "distribution_shift",
    None,
    ("Specifies distribution shift to use, if any."
     "aptos: loads APTOS (India) OOD validation and test datasets. "
     "  Kaggle/EyePACS in-domain datasets are unchanged."
     "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
     "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
     "  as OOD."),
)
flags.DEFINE_bool(
    "load_train_split",
    True,
    "Should always be enabled - required to load train split of the dataset.",
)
flags.DEFINE_bool("cache_eval_datasets", False, "Caches eval datasets.")

# Logging and hyperparameter tuning.
flags.DEFINE_bool("use_wandb", True, "Use wandb for logging.")
flags.DEFINE_string("wandb_dir", "wandb", "Directory where wandb logs go.")
flags.DEFINE_string("project", "ub-debug", "Wandb project name.")
flags.DEFINE_string("exp_name", None, "Give experiment a name.")
flags.DEFINE_string("exp_group", None, "Give experiment a group name.")

flags.DEFINE_string("checkpoint_path", None, "path to the checkpoint file")

FLAGS = flags.FLAGS

OUTPUT_DIM = 2


def main(argv):
  del argv

  # Wandb setup
  if FLAGS.use_wandb:
    pathlib.Path(FLAGS.wandb_dir).mkdir(parents=True, exist_ok=True)
    wandb_args = dict(
        project=FLAGS.project,
        entity="uncertainty-baselines",
        dir=FLAGS.wandb_dir,
        reinit=True,
        name=FLAGS.exp_name,
        group=FLAGS.exp_group,
    )
    wandb_run = wandb.init(**wandb_args)
    wandb.config.update(FLAGS, allow_val_change=True)
    output_dir = os.path.join(
        FLAGS.output_dir, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
  else:
    wandb_run = None
    output_dir = FLAGS.output_dir

  # Log run hypers
  hypers_dict = {
      "per_core_batch_size": FLAGS.per_core_batch_size,
      "base_learning_rate": FLAGS.base_learning_rate,
      "final_decay_factor": FLAGS.final_decay_factor,
      "one_minus_momentum": FLAGS.one_minus_momentum,
      "l2": FLAGS.l2,
      "loss_type": FLAGS.loss_type,
      "stochastic_linearization": FLAGS.stochastic_linearization,
  }
  logging.info("Hypers:")
  logging.info(pprint.pformat(hypers_dict))

  logging.info("*" * 100)
  logging.info("Platform that is used by JAX: %s",
               xla_bridge.get_backend().platform)
  logging.info("*" * 100)

  kh = initialize_random_keys(seed=FLAGS.seed)
  rng_key, rng_key_train, rng_key_test = random.split(kh.next_key(), 3)

  num_cores = FLAGS.num_cores
  per_core_batch_size = FLAGS.per_core_batch_size * num_cores

  datasets, steps = utils.load_dataset(
      train_batch_size=per_core_batch_size,
      eval_batch_size=per_core_batch_size,
      flags=FLAGS,
      strategy=None,
  )
  available_splits = list(datasets.keys())
  test_splits = [split for split in available_splits if "test" in split]
  eval_splits = [
      split for split in available_splits
      if "validation" in split or "test" in split
  ]
  eval_datasets = {split: iter(datasets[split]) for split in eval_splits}
  input_shape = [1] + utils.load_input_shape(dataset_train=datasets["train"])
  dataset_train = datasets["train"]
  train_steps_per_epoch = steps["train"]

  uncertainty_estimator_fn = utils.get_uncertainty_estimator(
      "fsvi", use_ensemble=False, use_tf=False)

  # Initialization of model, ptimizer, prior and loss
  initializer = Initializer(
      activation=FLAGS.activation,
      dropout_rate=FLAGS.dropout_rate,
      input_shape=input_shape,
      output_dim=OUTPUT_DIM,
      kl_scale=FLAGS.kl_scale,
      stochastic_linearization=FLAGS.stochastic_linearization,
      n_samples=FLAGS.n_samples,
      uniform_init_minval=FLAGS.uniform_init_minval,
      uniform_init_maxval=FLAGS.uniform_init_maxval,
      init_strategy=FLAGS.init_strategy,
      prior_mean=FLAGS.prior_mean,
      prior_cov=FLAGS.prior_cov,
  )
  opt = OptimizerInitializer(
      optimizer=FLAGS.optimizer,
      base_learning_rate=FLAGS.base_learning_rate,
      n_batches=train_steps_per_epoch,
      epochs=FLAGS.epochs,
      one_minus_momentum=FLAGS.one_minus_momentum,
      lr_warmup_epochs=FLAGS.lr_warmup_epochs,
      lr_decay_ratio=FLAGS.lr_decay_ratio,
      lr_decay_epochs=FLAGS.lr_decay_epochs,
      final_decay_factor=FLAGS.final_decay_factor,
      lr_schedule=FLAGS.lr_schedule,
  ).get()
  model, _, state, params = initializer.initialize_model(rng_key=rng_key)
  prior_fn = initializer.initialize_prior()
  opt_state = opt.init(params)
  loss = initializer.initialize_loss(model=model)

  summary_writer = tf.summary.create_file_writer(
      os.path.join(output_dir, "summaries"))

  # Loading from checkpoint
  initial_epoch = 0
  if FLAGS.checkpoint_path and FLAGS.load_from_checkpoint:
    with tf.io.gfile.GFile(FLAGS.checkpoint_path, mode="rb") as f:
      chkpt = pickle.load(f)
    assert chkpt["hparams"]["optimizer"] == FLAGS.optimizer
    assert chkpt["hparams"]["lr_schedule"] == FLAGS.lr_schedule
    state, params, opt_state = chkpt["state"], chkpt["params"], chkpt[
        "opt_state"]
    logging.info("Loaded checkpoint %s", FLAGS.checkpoint_path)
    initial_epoch = chkpt["epoch"] + 1

  # Update metrics
  use_tpu = any(["tpu" in str(d).lower() for d in jax.devices()])
  metrics = utils.get_diabetic_retinopathy_base_metrics(
      use_tpu=use_tpu,
      num_bins=FLAGS.num_bins,
      use_validation=FLAGS.use_validation,
      available_splits=available_splits,
  )
  # Define metrics outside the accelerator scope for CPU eval.
  if not use_tpu:
    metrics.update(
        utils.get_diabetic_retinopathy_cpu_metrics(
            available_splits=available_splits,
            use_validation=FLAGS.use_validation))

  for test_split in test_splits:
    metrics.update({f"{test_split}/ms_per_example": tf.keras.metrics.Mean()})

  ########################## specify functions ##########################
  @jit
  def update_fn(
      params,
      state,
      opt_state,
      x_batch,
      y_batch,
      inducing_inputs,
      rng_key,
  ):
    """Captured variables:

        loss, FLAGS, num_cores
    """
    prior_mean, prior_cov = prior_fn((inducing_inputs.shape[0], OUTPUT_DIM))

    grads, additional_info = jax.grad(
        loss, argnums=0, has_aux=True)(
            params,
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

    if num_cores > 1:
      grads = tree.map_structure(
          functools.partial(jax.lax.pmean, axis_name="i"), grads)
      additional_info = tree.map_structure(
          functools.partial(jax.lax.pmean, axis_name="i"), additional_info)
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    new_state = additional_info["state"]
    return new_params, new_state, opt_state, additional_info

  def parallelizable_train_per_batch_computation(
      params,
      state,
      opt_state,
      rng_key_train,
      x_batch,
      labels,
  ):
    """Variables captured by closure:

        output_dim, FLAGS, update_fn, inducing_input_fn, model
    """
    # features have shape (batch_dim, 128, 128, 3), labels shape (batch_dim,)
    y_batch = to_one_hot(labels, OUTPUT_DIM)
    inducing_key, rng_key_train, rng_key_eval = jax.random.split(
        rng_key_train, 3)
    inducing_inputs = inducing_input_fn(x_batch, inducing_key,
                                        FLAGS.n_inducing_inputs)
    params, state, opt_state, additional_info = update_fn(
        params,
        state,
        opt_state,
        x_batch,
        y_batch,
        inducing_inputs,
        rng_key_train,
    )
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
    parallelizable_train_per_batch_computation = jax.pmap(
        parallelizable_train_per_batch_computation,
        axis_name="i",
        in_axes=(None, None, None, 0, 0, 0),
        out_axes=(None, None, None, None, 0),
    )

  def parallelizable_evaluate_per_batch_computation(
      x_batch,
      labels,
      rng_key,
      params,
      state,
      is_deterministic,
  ):
    """Variables captured by closure:

        model
    """
    # Compute prediction, total, aleatoric, and epistemic uncertainties
    pred_and_uncert = uncertainty_estimator_fn(
        x_batch,
        model,
        training_setting=False,
        params=params,
        state=state,
        num_samples=FLAGS.n_samples_test,
        rng_key=rng_key,
    )

    result = {
        "y_true": labels,
        "y_pred": pred_and_uncert["prediction"],
        "y_pred_entropy": pred_and_uncert["predictive_entropy"],
        "y_pred_variance": pred_and_uncert["predictive_variance"],
    }

    if not is_deterministic:
      result["y_aleatoric_uncert"] = pred_and_uncert["aleatoric_uncertainty"]
      result["y_epistemic_uncert"] = pred_and_uncert["epistemic_uncertainty"]

    return result

  if num_cores > 1:
    parallelizable_evaluate_per_batch_computation = jax.pmap(
        parallelizable_evaluate_per_batch_computation,
        in_axes=(0, 0, 0, None, None, None),
        static_broadcasted_argnums=(5,),
    )

  def eval_step_jax(
      dataset_iterator,
      dataset_steps,
      is_deterministic,
      params,
      state,
      rng_key,
  ):
    """Variables captured by closure:

        parallelizable_evaluate_per_batch_computation, FLAGS
    """
    list_of_results = []
    for _ in tqdm(range(dataset_steps), "evaluation loop"):
      data = next(dataset_iterator)
      images = data["features"]._numpy()  # pylint: disable=protected-access
      labels = data["labels"]._numpy()  # pylint: disable=protected-access
      _, rng_key = random.split(rng_key)
      if num_cores > 1:
        images, labels = reshape_to_multiple_cores(images, labels, num_cores)
        keys = random.split(rng_key, num_cores)
      else:
        keys = rng_key
      # Compute prediction, total, aleatoric, and epistemic
      # uncertainty estimates
      arrays_dict = parallelizable_evaluate_per_batch_computation(
          images,
          labels,
          keys,
          params,
          state,
          is_deterministic,
      )
      reshaped_arrays_dict = {
          name: reshape_to_one_core(array)
          for name, array in arrays_dict.items()
      }
      list_of_results.append(reshaped_arrays_dict)

    results_arrs = merge_list_of_dicts(list_of_results)
    return results_arrs

  # It is important to define this variable `estimator_args` here instead
  # of declaring it in the loop. This is to avoid JAX memory leak.
  estimator_args = {
      "rng_key": rng_key_test,
      "params": params,
      "state": state,
  }

  ########################## train-eval loop ##########################

  logging.info(f"\n--- Training for {FLAGS.epochs} epochs ---\n")
  train_iterator = iter(dataset_train)
  for epoch in range(initial_epoch, FLAGS.epochs):
    t0 = time.time()
    for _ in tqdm(range(train_steps_per_epoch), desc="gradient steps..."):
      start = time.time()
      data = next(train_iterator)
      x_batch = data["features"]._numpy()  # pylint: disable=protected-access
      labels = data["labels"]._numpy()  # pylint: disable=protected-access
      orig_labels = labels
      logging.debug(f"loading data used {time.time() - start:.2f} seconds")
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
      ) = parallelizable_train_per_batch_computation(
          params,
          state,
          opt_state,
          keys,
          x_batch,
          labels,
      )

      logging.debug(
          f"per-batch computation used {time.time() - start:.2f} seconds")
      start = time.time()

      if num_cores > 1:
        probs = reshape_to_one_core(probs)

      logging.debug(f"reshape again used {time.time() - start:.2f} seconds")
      start = time.time()

      metrics["train/loss"].update_state(additional_info["loss"].item())
      log_likelihood_per_input = (
          additional_info["log_likelihood"].item() / orig_labels.shape[0])
      metrics["train/negative_log_likelihood"].update_state(
          -log_likelihood_per_input)
      probs_of_labels = probs[:, 1]
      metrics["train/accuracy"].update_state(orig_labels, probs_of_labels)
      metrics["train/auprc"].update_state(orig_labels, probs_of_labels)
      metrics["train/auroc"].update_state(orig_labels, probs_of_labels)

      if not use_tpu:
        metrics["train/ece"].add_batch(probs_of_labels, label=orig_labels)
      logging.debug(f"compute metric used {time.time() - start:.2f} seconds")

    _, rng_key_test = random.split(rng_key_test)

    estimator_args["rng_key"] = rng_key_test
    estimator_args["params"] = params
    estimator_args["state"] = state

    per_pred_results, total_results = utils.evaluate_model_and_compute_metrics(
        strategy=None,
        eval_datasets=eval_datasets,
        steps=steps,
        metrics=metrics,
        eval_estimator=None,
        uncertainty_estimator_fn=None,
        eval_batch_size=per_core_batch_size,
        available_splits=available_splits,
        estimator_args=estimator_args,
        call_dataset_iter=False,
        is_deterministic=False,
        num_bins=FLAGS.num_bins,
        use_tpu=use_tpu,
        return_per_pred_results=True,
        backend="jax",
        eval_step_jax=eval_step_jax,
    )

    # Optionally log to wandb
    if FLAGS.use_wandb:
      wandb.log(total_results, step=epoch)

    with summary_writer.as_default():
      for name, result in total_results.items():
        if result is not None:
          tf.summary.scalar(name, result, step=epoch + 1)

    for metric in metrics.values():
      metric.reset_states()

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      to_save = {
          "params": params,
          "state": state,
          "hparams": flags_2_dict(),
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

    now_t0 = time.time()
    logging.info(f"Epoch {epoch} used {now_t0 - t0:.2f} seconds")

  to_save = {
      "params": params,
      "state": state,
      "hparams": flags_2_dict(),
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
    hp.hparams({
        "base_learning_rate": FLAGS.base_learning_rate,
        "per_core_batch_size": FLAGS.per_core_batch_size,
        "final_decay_factor": FLAGS.final_decay_factor,
        "one_minus_momentum": FLAGS.one_minus_momentum,
        "optimizer": FLAGS.optimizer,
        "loss_type": FLAGS.loss_type,
        "l2": FLAGS.l2,
    })
  if wandb_run is not None:
    wandb_run.finish()


@jax.jit
def reshape_to_one_core(x: jnp.ndarray) -> jnp.ndarray:
  return jnp.reshape(x, [-1] + list(x.shape[2:]))


@functools.partial(jax.jit, static_argnums=(2,))
def reshape_to_multiple_cores(
    x_batch: jnp.ndarray,
    y_batch: jnp.ndarray,
    num_cores: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
  func = lambda x: x.reshape([num_cores, -1] + list(x.shape)[1:])
  x_batch, y_batch = list(map(
      func,
      [x_batch, y_batch],
  ))
  return x_batch, y_batch


def merge_list_of_dicts(list_of_dicts: List[Dict]) -> Dict:
  if not list_of_dicts:
    return {}
  keys = list_of_dicts[0].keys()
  merged_dict = {k: jnp.stack([d[k] for d in list_of_dicts]) for k in keys}
  return merged_dict


def inducing_input_fn(x_batch: jnp.ndarray, rng_key: jnp.ndarray,
                      n_inducing_inputs: int) -> jnp.ndarray:
  """Select inducing inputs from training input data

  Args:
    x_batch: training data inputs
    rng_key: jax random key
    n_inducing_inputs: number of inducing inputs to select

  Returns:
    a jax ndarray, inducing inputs
  """
  permutation = jax.random.permutation(key=rng_key, x=x_batch.shape[0])
  x_batch_permuted = x_batch[permutation, :]
  inducing_inputs = x_batch_permuted[:n_inducing_inputs]
  return inducing_inputs


def flags_2_dict() -> Dict[str, Any]:
  current_module = sys.modules[__name__]
  flags_list = FLAGS.get_flags_for_module(current_module)
  return {f.name: f.value for f in flags_list}


if __name__ == "__main__":
  app.run(main)
