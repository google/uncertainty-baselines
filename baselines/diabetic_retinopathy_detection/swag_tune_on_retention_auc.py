import os
import time

import numpy as np
import tensorflow as tf
import torch
from absl import app
from absl import flags
from absl import logging
from tensorboard.plugins.hparams import api as hp

import uncertainty_baselines as ub
import utils  # local file import
from swag_utils import utils as swag_utils
from swag_utils.posterior import SWAG
from torch_utils import count_parameters

"""
Stochastic Weight Averaging -- Gaussian (SWAG)
Introduced in "A Simple Baseline for Bayesian Deep Learning",
Maddox et al., _NeurIPS 2019_.
Based on Wesley Maddox's implementation: 
  https://github.com/wjmaddox/swa_gaussian
"""

DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
    'output_dir', '/tmp/diabetic_retinopathy_detection/swag',
    'The directory where the model weights and training/evaluation summaries '
    'are stored. If you aim to use these as trained models for an ensemble, '
    'you should specify an output_dir name that includes the random seed to '
    'avoid overwriting.')
flags.DEFINE_string('data_dir', None, 'Path to training and testing data.')
flags.DEFINE_bool('use_validation', True, 'Whether to use a validation split.')
flags.DEFINE_bool('use_test', True, 'Whether to use a test split.')
flags.DEFINE_string(
  'dr_decision_threshold', 'moderate',
  ("specifies where to binarize the labels {0, 1, 2, 3, 4} to create the "
   "binary classification task. Only affects the APTOS dataset partitioning. "
   "'mild': classify {0} vs {1, 2, 3, 4}, i.e., mild DR or worse?"
   "'moderate': classify {0, 1} vs {2, 3, 4}, i.e., moderate DR or worse?"))

# Stochastic Weight Averaging -- Gaussian (SWAG) flags.
flags.DEFINE_float("swa_start", 70, "SWA start epoch number (default: 70)")
flags.DEFINE_float("swa_lr", 0.02, "SWA LR (default: 0.02)")
flags.DEFINE_integer(
  "swa_c_epochs", 1,
  "SWA model collection frequency/cycle length in epochs (default: 1)")
flags.DEFINE_bool("cov_mat", True, "Save sample covariance")
flags.DEFINE_integer(
  "max_num_models", 20, "Maximum number of SWAG models to save")
flags.DEFINE_bool("store_schedule", True, "Store schedule")
flags.DEFINE_float("scale", 1.0, "Scale para")

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

# Learning Rate / SGD flags.
flags.DEFINE_float('base_learning_rate', 0.01, 'Base learning rate.')
flags.DEFINE_float('one_minus_momentum', 0.1, 'Optimizer momentum.')

# General model flags.
flags.DEFINE_integer('seed', 42, 'Random seed.')
flags.DEFINE_string(
    'class_reweight_mode', None,
    'Dataset is imbalanced. '
    '`None` (default) will not perform any loss reweighting. '
    '`constant` will use the train proportions to reweight the binary cross '
    'entropy loss. '
    '`minibatch` will use the proportions of each minibatch to reweight '
    'the loss.')
flags.DEFINE_float('l2', 5e-5, 'L2 regularization coefficient.')
flags.DEFINE_integer('train_epochs', DEFAULT_NUM_EPOCHS,
                     'Number of training epochs.')
flags.DEFINE_integer('train_batch_size', 16,
                     'The per-core training batch size.')
flags.DEFINE_integer('eval_batch_size', 32,
                     'The per-core validation/test batch size.')
flags.DEFINE_integer(
    'checkpoint_interval', 25, 'Number of epochs between saving checkpoints. '
    'Use -1 to never save checkpoints.')

flags.DEFINE_integer(
  'num_mc_samples_eval', 10,
  'Number of samples from approximate posterior to use for prediction.')

# Metric flags.
flags.DEFINE_integer('num_bins', 15, 'Number of bins for ECE.')

# Accelerator flags.
flags.DEFINE_bool('use_gpu', True,
                  'Whether to run on (a single) GPU or otherwise CPU.')
FLAGS = flags.FLAGS


def main(argv):
  del argv  # unused arg
  tf.io.gfile.makedirs(FLAGS.output_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)

  # Set seeds
  tf.random.set_seed(FLAGS.seed)
  np.random.seed(FLAGS.seed)
  torch.manual_seed(FLAGS.seed)

  # Resolve CUDA device(s)
  if FLAGS.use_gpu and torch.cuda.is_available():
    print('Running model with CUDA.')
    device = 'cuda:0'
    # torch.backends.cudnn.benchmark = True
  else:
    print('Running model on CPU.')
    device = 'cpu'

  train_batch_size = FLAGS.train_batch_size

  # We don't do MC sampling at eval time in parallel, so no need to scale down
  # (as we do with MC Dropout, for example.)
  eval_batch_size = FLAGS.eval_batch_size

  # Reweighting loss for class imbalance
  class_reweight_mode = FLAGS.class_reweight_mode
  if class_reweight_mode == 'constant':
    class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  else:
    class_weights = None

  # Load in datasets.
  datasets, steps = utils.load_dataset(
    train_batch_size, eval_batch_size, flags=FLAGS, strategy=None)
  available_splits = list(datasets.keys())
  test_splits = [split for split in available_splits if 'test' in split]
  eval_splits = [split for split in available_splits
                 if 'validation' in split or 'test' in split]
  eval_datasets = {split: datasets[split] for split in eval_splits}
  dataset_train = datasets['train']
  train_steps_per_epoch = steps['train']

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  # * Build Model *
  logging.info('Building PyTorch ResNet-50 SWAG model.')
  model = ub.models.resnet50_torch(num_classes=1)
  model = model.to(device)
  logging.info('Model input shape: %s', utils.load_input_shape(dataset_train))
  logging.info('(Single) model number of weights: %s', count_parameters(model))

  # * Init Model and Optimizer *
  no_cov_mat = not FLAGS.cov_mat
  swag_model = SWAG(
    ub.models.resnet50_torch, no_cov_mat=no_cov_mat,
    max_num_models=FLAGS.max_num_models, num_classes=1)
  swag_model.to(device)
  optimizer = torch.optim.SGD(
    model.parameters(), lr=FLAGS.base_learning_rate,
    momentum=(1 - FLAGS.one_minus_momentum))
  # TODO: add load from checkpoint

  # * Init Metrics *
  metrics = utils.get_diabetic_retinopathy_base_metrics(
    use_tpu=False,
    num_bins=FLAGS.num_bins,
    use_validation=FLAGS.use_validation,
    available_splits=available_splits)
  metrics.update(
    utils.get_diabetic_retinopathy_cpu_metrics(
      available_splits=available_splits,
      use_validation=FLAGS.use_validation))

  for test_split in test_splits:
    metrics.update({f'{test_split}/ms_per_example': tf.keras.metrics.Mean()})

  # * Init Loss *
  # Initialize loss function based on class reweighting setting
  loss_fn = utils.get_diabetic_retinopathy_loss_fn(
      class_reweight_mode=class_reweight_mode, class_weights=class_weights)

  # Set constants for use with TF dataloader
  sigmoid = torch.nn.Sigmoid()
  max_steps = steps['train'] * FLAGS.train_epochs
  image_h = 512
  image_w = 512

  def train_step(iterator):
    def step_fn(inputs):
      images = inputs['features']
      labels = inputs['labels']

      # For minibatch class reweighting, initialize per-batch loss function
      if class_reweight_mode == 'minibatch':
        batch_loss_fn = utils.get_minibatch_reweighted_loss_fn(
          labels=labels, loss_fn_type='torch')
      else:
        batch_loss_fn = loss_fn

      images = torch.from_numpy(images._numpy()).view(train_batch_size, 3,  # pylint: disable=protected-access
                                                      image_h,
                                                      image_w).to(device)
      labels = torch.from_numpy(labels._numpy()).to(device).float()  # pylint: disable=protected-access

      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward
      logits = model(images)
      probs = torch.squeeze(sigmoid(logits))

      # Add L2 regularization loss to NLL
      negative_log_likelihood = batch_loss_fn(
        y_true=labels, y_pred=probs, from_logits=False)
      l2_loss = sum(p.pow(2.0).sum() for p in model.parameters())
      loss = negative_log_likelihood + (FLAGS.l2 * l2_loss)

      # Backward/optimizer
      loss.backward()
      optimizer.step()

      # Convert to NumPy for metrics updates
      loss = loss.detach()
      negative_log_likelihood = negative_log_likelihood.detach()
      labels = labels.detach()
      probs = probs.detach()

      if device != 'cpu':
        loss = loss.cpu()
        negative_log_likelihood = negative_log_likelihood.cpu()
        labels = labels.cpu()
        probs = probs.cpu()

      loss = loss.numpy()
      negative_log_likelihood = negative_log_likelihood.numpy()
      labels = labels.numpy()
      probs = probs.numpy()

      metrics['train/loss'].update_state(loss)
      metrics['train/negative_log_likelihood'].update_state(
        negative_log_likelihood)
      metrics['train/accuracy'].update_state(labels, probs)
      metrics['train/auprc'].update_state(labels, probs)
      metrics['train/auroc'].update_state(labels, probs)
      metrics['train/ece'].add_batch(probs, label=labels)

    model.train()
    for step in range(steps['train']):
      step_fn(next(iterator))

  def test_step_swag(model_to_eval, iterator, num_steps):
    """Need to BatchNorm over the training set for every stochastic sample.
    Therefore, we:
      1. Sample from the posterior,
      2. (here) Compute a full epoch, and return predictions and labels,
      3. Update metrics altogether at the end.
  """
    def step_fn(inputs):
      images = inputs['features']
      labels = inputs['labels']
      images = torch.from_numpy(images._numpy()).view(eval_batch_size, 3,  # pylint: disable=protected-access
                                                      image_h,
                                                      image_w).to(device)
      labels = torch.from_numpy(
          labels._numpy()).to(device).float().unsqueeze(-1)  # pylint: disable=protected-access
      with torch.no_grad():
        logits = model_to_eval(images)

      probs = sigmoid(logits)
      return labels, probs

    labels_list = []
    probs_list = []
    model.eval()
    for _ in range(num_steps):
      labels, probs = step_fn(next(iterator))
      labels_list.append(labels)
      probs_list.append(probs)

    return {
      'labels': torch.cat(labels_list, dim=0),
      'probs': torch.cat(probs_list, dim=0)
    }

  def update_probs_labels_dicts(dataset_key, dataset_results):
    probs = dataset_results['probs']
    labels = dataset_results['labels']
    if dataset_key not in dataset_split_to_probs:
      dataset_split_to_probs[dataset_key] = probs
    else:
      dataset_split_to_probs[dataset_key] += probs

    dataset_split_to_labels[dataset_key] = labels

  def compute_and_update_swag_eval_metrics(
      key_to_probs, key_to_labels
  ):
    key_to_probs = {
      dataset_split: arr / num_samples
      for dataset_split, arr in key_to_probs.items()}
    for dataset_split in key_to_probs.keys():
      probs = key_to_probs[dataset_split]
      labels = key_to_labels[dataset_split]

      negative_log_likelihood = torch.nn.BCELoss()(input=probs, target=labels)

      # Convert to NumPy for metrics updates
      negative_log_likelihood = negative_log_likelihood.detach()
      labels = labels.detach()
      probs = probs.detach()

      if device != 'cpu':
        negative_log_likelihood = negative_log_likelihood.cpu()
        labels = labels.cpu()
        probs = probs.cpu()

      negative_log_likelihood = negative_log_likelihood.numpy()
      labels = labels.numpy()
      probs = probs.numpy()

      metrics[dataset_split +
              '/negative_log_likelihood'].update_state(negative_log_likelihood)
      metrics[dataset_split + '/accuracy'].update_state(labels, probs)
      metrics[dataset_split + '/auprc'].update_state(labels, probs)
      metrics[dataset_split + '/auroc'].update_state(labels, probs)
      metrics[dataset_split + '/ece'].add_batch(probs, label=labels)

    return metrics

  # * Pre-Training Loop Housekeeping *
  initial_epoch = 0
  swag_is_active = False
  sample_with_cov = FLAGS.cov_mat

  # * Start of Training Loop *
  start_time = time.time()
  train_iterator = iter(dataset_train)
  for epoch in range(initial_epoch, FLAGS.train_epochs):
    # * SWAG Setup *
    # Weight averaging should start after model has essentially converged
    if (epoch + 1) > FLAGS.swa_start and not swag_is_active:
      swag_is_active = True
      print(f'Reached epoch {epoch + 1}. Activating SWAG model averaging.')

    # * Update SWAG LR/Schedule *
    lr = swag_utils.schedule(
      swa_start=FLAGS.swa_start, swa_lr=FLAGS.swa_lr,
      base_learning_rate=FLAGS.base_learning_rate, epoch=epoch)
    swag_utils.adjust_learning_rate(optimizer, lr)

    logging.info('Starting to run epoch: %s', epoch + 1)

    # * Train Epoch *
    train_step(train_iterator)
    current_step = (epoch + 1) * steps['train']
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('{:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   current_step / max_steps, epoch + 1, FLAGS.train_epochs,
                   steps_per_sec, eta_seconds / 60, time_elapsed / 60))
    logging.info(message)

    if swag_is_active:
      # Collect the model parameters
      swag_model.collect_model(model)
      # We will do MC integration with our SWAG approximate posterior
      eval_model = swag_model
      num_samples = FLAGS.num_mc_samples_eval
    else:
      eval_model = model
      num_samples = 1

    eval_datasets = {
      dataset_key: dataset for dataset_key, dataset in datasets.items()
      if 'validation' in dataset_key or 'test' in dataset_key}
    dataset_split_to_probs = {}
    dataset_split_to_labels = {}

    for sample in range(num_samples):
      # Sample from approx posterior
      if swag_is_active:
        swag_model.sample(scale=FLAGS.scale, cov=sample_with_cov)
        # BN Update requires iteration over full train loop, hence we
        # put the MC sampling outside of the evaluation loops.
        swag_utils.bn_update(
          train_iterator, swag_model, num_train_steps=steps['train'],
          train_batch_size=train_batch_size, image_h=512, image_w=512,
          device=device)

      if FLAGS.use_validation:
        for validation_split in validation_datasets:
          dataset_validation = datasets[validation_split]
          validation_iterator = iter(dataset_validation)
          logging.info(
            f'Starting to run {validation_split} eval, sample {sample} '
            f'at epoch: %s', epoch + 1)
          val_results = test_step_swag(
            eval_model, validation_iterator, steps[validation_split])
          update_probs_labels_dicts(
              dataset_key=validation_split, dataset_results=val_results)

      for test_split in test_datasets:
        dataset_test = datasets[test_split]
        test_iterator = iter(dataset_test)
        logging.info(
          f'Starting to run {test_split} eval, sample {sample} '
          f'at epoch: %s', epoch + 1)
        test_start_time = time.time()
        test_results = test_step_swag(
          eval_model, test_iterator, steps[test_split])
        update_probs_labels_dicts(
          dataset_key=test_split, dataset_results=test_results)

        # Just record on the last sample, and multiply by number of samples.
        # (Approximation, for convenience.)
        if sample == (num_samples - 1):
          ms_per_example = num_samples * (
            time.time() - test_start_time) * 1e6 / eval_batch_size
          metrics[f'{test_split}/ms_per_example'].update_state(ms_per_example)

    # Updates metrics dict with the
    # dataset_split_to_probs and dataset_split_to_labels
    compute_and_update_swag_eval_metrics(
      key_to_probs=dataset_split_to_probs,
      key_to_labels=dataset_split_to_labels)

    # Log and write to summary the epoch metrics
    utils.log_epoch_metrics(metrics=metrics, use_tpu=False,
                            dataset_splits=list(datasets.keys()))
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

    if ((FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0) or
        (epoch + 1) == FLAGS.train_epochs):
      log_prefix = 'Saved Torch base model'
      swag_utils.save_checkpoint(
        FLAGS.output_dir, epoch + 1,
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict())
      if swag_is_active:
        swag_utils.save_checkpoint(
          FLAGS.output_dir, epoch + 1,
          name="swag",
          state_dict=swag_model.state_dict())
        log_prefix += ' and SWAG model'

      logging.info(f'{log_prefix} to {FLAGS.output_dir}.')

  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
