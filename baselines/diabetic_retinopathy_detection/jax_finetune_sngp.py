import tensorflow as tf

tf.config.experimental.set_visible_devices([], 'GPU')
tf.config.experimental.set_visible_devices([], 'TPU_SYSTEM')
tf.config.experimental.set_visible_devices([], 'TPU')

print(tf.config.experimental.get_visible_devices())

import itertools
import multiprocessing
import numbers
import os
import pathlib
from datetime import datetime
from functools import partial  # pylint: disable=g-importing-member so standard
from time import time
import ml_collections

import flax
import flax.jax_utils as flax_utils
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from absl import app
from absl import flags
from absl import logging
from clu import metric_writers
from clu import parameter_overview
from clu import periodic_actions
from clu import preprocess_spec
from scipy.stats import entropy
from tensorflow.io import gfile

import checkpoint_utils  # local file import
import input_utils  # local file import
import preprocess_utils  # local file import
import train_utils  # local file import
import uncertainty_baselines as ub
from baselines.diabetic_retinopathy_detection.utils import (
  save_per_prediction_results, evaluate_vit_predictions)
# local file import
from experiments.config.imagenet21k_vit_base16_sngp_finetune import (
  get_config as get_i21k_sngp_config)

DEFAULT_NUM_EPOCHS = 90

# Data load / output flags.
flags.DEFINE_string(
  'output_dir', '/tmp/diabetic_retinopathy_detection/vit-16-i21k',
  'The directory where the model weights and training/evaluation summaries '
  'are stored. If you aim to use these as trained models for ensemble.py, '
  'you should specify an output_dir name that includes the random seed to '
  'avoid overwriting.')
flags.DEFINE_string(
  'distribution_shift', None,
  ("Specifies distribution shift to use, if any."
   "aptos: loads APTOS (India) OOD validation and test datasets. "
   "  Kaggle/EyePACS in-domain datasets are unchanged."
   "severity: uses DiabeticRetinopathySeverityShift dataset, a subdivision "
   "  of the Kaggle/EyePACS dataset to hold out clinical severity labels "
   "  as OOD."))
flags.mark_flag_as_required('distribution_shift')
flags.DEFINE_string(
  'pretrain_dataset', 'imagenet21k',
  ("Dataset for model pretraining. Specifies the config to use."))

# Logging and hyperparameter tuning.
flags.DEFINE_bool('use_wandb', False, 'Use wandb for logging.')
flags.DEFINE_string('wandb_dir', 'wandb', 'Directory where wandb logs go.')
flags.DEFINE_string('project', 'ub-debug', 'Wandb project name.')
flags.DEFINE_string('exp_name', None, 'Give experiment a name.')
flags.DEFINE_string('exp_group', None, 'Give experiment a group name.')

# Learning rate / SGD flags.
flags.DEFINE_float('grad_clip_norm', 20.0, 'Gradient clipping threshold.')
flags.DEFINE_float('weight_decay', None, 'Gradient clipping threshold.')
flags.DEFINE_float('lr_base', 0.1, 'Base learning rate.')
flags.DEFINE_integer('lr_warmup_steps', 500, 'Number of LR warmup steps.')
flags.DEFINE_string('lr_decay_type', 'cosine',
                    'Type of LR decay / schedule. Options: cosine, linear.')

# General model flags.
flags.DEFINE_integer('total_steps', 10000, 'Total steps.')
flags.DEFINE_integer('batch_size', 128, 'Batch size.')
flags.DEFINE_integer('seed', 0, 'Random seed.')
flags.DEFINE_string(
  'class_reweight_mode', None,
  'Dataset is imbalanced (19.6%, 18.8%, 19.2% positive examples in train, '
  'val, test respectively). `None` (default) will not perform any loss '
  'reweighting. `constant` will use the train proportions to reweight the '
  'binary cross entropy loss. `minibatch` will use the proportions of each '
  'minibatch to reweight the loss.')

# SNGP flags.
flags.DEFINE_float('sngp_ridge_penalty', 1., 'SNGP ridge penalty.')
flags.DEFINE_float('sngp_covmat_momentum', -1., 'SNGP covmat momentum.')
flags.DEFINE_float('sngp_mean_field_factor', 20., 'SNGP Mean Field factor.')

FLAGS = flags.FLAGS

# Utility functions.
def accumulate_gradient_with_states(
    loss_and_grad_fn,
    params,
    states,  # Allows for states.
    images,
    labels,
    accum_steps):
  """Improved version of `train_utils.accumulate_gradient()` that allows for states."""
  # This function handles the `loss_and_grad_fn` function which takes a state
  # argument and returns ((losses, states), grads).
  if accum_steps and accum_steps > 1:
    assert images.shape[0] % accum_steps == 0, (
        f'Bad accum_steps {accum_steps} for batch size {images.shape[0]}')
    step_size = images.shape[0] // accum_steps

    # Run the first step.
    (l, s), g = loss_and_grad_fn(params, states, images[:step_size],
                                 labels[:step_size])

    # Run the rest of the steps.
    def acc_grad_and_loss(i, l_s_g):
      # Extract data for current step.
      imgs = jax.lax.dynamic_slice(images, (i * step_size, 0, 0, 0),
                                   (step_size,) + images.shape[1:])
      lbls = jax.lax.dynamic_slice(labels, (i * step_size, 0),
                                   (step_size, labels.shape[1]))
      # Update state and accumulate gradient.
      l, s, g = l_s_g
      (li, si), gi = loss_and_grad_fn(params, s, imgs, lbls)
      return (l + li, si, jax.tree_multimap(lambda x, y: x + y, g, gi))

    l, s, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, s, g))
    l, g = jax.tree_map(lambda x: x / accum_steps, (l, g))
    return (l, s), g
  else:
    return loss_and_grad_fn(params, states, images, labels)


def get_gp_kwargs(gp_config):
  """Extract keyword argument parameters for the Gaussian process layer."""
  covmat_momentum = gp_config.get('covmat_momentum', 0.999)

  # Extracts model parameter.
  logging.info('gp_config.covmat_momentum = %s', covmat_momentum)
  covmat_momentum = None if covmat_momentum < 0. else covmat_momentum
  covmat_kwargs = dict(momentum=covmat_momentum)

  # Assembles into kwargs dictionary.
  gp_layer_kwargs = dict(covmat_kwargs=covmat_kwargs)

  return gp_layer_kwargs


def main(argv):
  del argv  # unused arg

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
    output_dir = str(os.path.join(
      FLAGS.output_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
  else:
    wandb_run = None
    output_dir = FLAGS.output_dir

  tf.io.gfile.makedirs(output_dir)
  logging.info('Saving checkpoints at %s', output_dir)

  # Dataset Split Flags
  dist_shift = FLAGS.distribution_shift
  print(f'Distribution Shift: {dist_shift}.')
  if dist_shift == 'aptos':
    in_domain_dataset = 'ub_diabetic_retinopathy_detection'
    ood_dataset = 'aptos'
    train_split = 'train'
    in_domain_val_split = 'validation'
    ood_val_split = 'validation'
  elif dist_shift == 'severity':
    in_domain_dataset = 'diabetic_retinopathy_severity_shift_moderate'
    ood_dataset = 'diabetic_retinopathy_severity_shift_moderate'
    train_split = 'train'
    in_domain_val_split = 'in_domain_validation'
    ood_val_split = 'ood_validation'
  else:
    raise NotImplementedError

  # LR / Optimization Flags
  batch_size = FLAGS.batch_size
  grad_clip_norm = FLAGS.grad_clip_norm
  weight_decay = FLAGS.weight_decay
  lr_dict = {
    'base': FLAGS.lr_base,
    'warmup_steps': FLAGS.lr_warmup_steps,
    'decay_type': FLAGS.lr_decay_type
  }
  print('Standard wandb hyperparameters:')
  print({
    'batch_size': batch_size,
    'grad_clip_norm': grad_clip_norm,
    'weight_decay': weight_decay,
    'total_steps': FLAGS.total_steps,
    'lr': lr_dict
  })

  # Reweighting loss for class imbalance
  # class_reweight_mode = FLAGS.class_reweight_mode
  # if class_reweight_mode == 'constant':
  #   class_weights = utils.get_diabetic_retinopathy_class_balance_weights()
  # else:
  #   class_weights = None

  # Shows the number of available devices.
  # In a CPU/GPU runtime this will be a single device.
  # In a TPU runtime this will be 8 cores.
  print('Number of Jax local devices:', jax.local_devices())

  if FLAGS.pretrain_dataset == 'imagenet21k':
    config = get_i21k_sngp_config()
  else:
    raise NotImplementedError

  # SNGP Flags
  # Gaussian process layer section
  config.gp_layer = ml_collections.ConfigDict()
  config.gp_layer.ridge_penalty = FLAGS.sngp_ridge_penalty

  # Disable momentum in order to use exact covariance update for finetuning.
  config.gp_layer.covmat_momentum = FLAGS.sngp_covmat_momentum

  config.gp_layer.mean_field_factor = FLAGS.sngp_mean_field_factor
  print('SNGP Params:', config.gp_layer)

  # TODO(nband): fix sigmoid loss issues.
  assert config.get('loss', None) == 'softmax_xent'

  # config = FLAGS.config

  seed = FLAGS.seed
  rng = jax.random.PRNGKey(seed)
  tf.random.set_seed(seed)

  if config.get('data_dir'):
    logging.info('data_dir=%s', config.data_dir)
  logging.info('Output dir: %s', output_dir)

  save_checkpoint_path = None
  if config.get('checkpoint_steps'):
    gfile.makedirs(output_dir)
    save_checkpoint_path = os.path.join(output_dir, 'checkpoint.npz')

  # Create an asynchronous multi-metric writer.
  writer = metric_writers.create_default_writer(
    output_dir, just_logging=jax.process_index() > 0)

  # The pool is used to perform misc operations such as logging in async way.
  pool = multiprocessing.pool.ThreadPool()

  def write_note(note):
    if jax.host_id() == 0:
      logging.info('NOTE: %s', note)

  write_note('Initializing...')

  fillin = lambda *_: None
  # Verify settings to make sure no checkpoints are accidentally missed.
  if config.get('keep_checkpoint_steps'):
    assert config.get('checkpoint_steps'), 'Specify `checkpoint_steps`.'
    assert config.keep_checkpoint_steps % config.checkpoint_steps == 0, (
      f'`keep_checkpoint_steps` ({config.checkpoint_steps}) should be'
      f'divisible by `checkpoint_steps ({config.checkpoint_steps}).`')

  batch_size_eval = config.get('batch_size_eval', batch_size)
  if (batch_size % jax.device_count() != 0 or
      batch_size_eval % jax.device_count() != 0):
    raise ValueError(
      f'Batch sizes ({batch_size} and {batch_size_eval}) must '
      f'be divisible by device number ({jax.device_count()})')

  local_batch_size = batch_size // jax.host_count()
  local_batch_size_eval = batch_size_eval // jax.host_count()
  logging.info(
    'Global batch size %d on %d hosts results in %d local batch size. '
    'With %d devices per host (%d devices total), that\'s a %d per-device '
    'batch size.',
    batch_size, jax.host_count(), local_batch_size,
    jax.local_device_count(), jax.device_count(),
    local_batch_size // jax.local_device_count())

  write_note('Initializing preprocessing function...')
  # Same preprocessing function for training and evaluation
  preproc_fn = preprocess_spec.parse(
    spec=config.pp_train, available_ops=preprocess_utils.all_ops())

  write_note('Initializing train dataset...')
  rng, train_ds_rng = jax.random.split(rng)
  train_ds_rng = jax.random.fold_in(train_ds_rng, jax.process_index())
  train_base_dataset = ub.datasets.get(
    in_domain_dataset, split=train_split,
    data_dir=config.get('data_dir'))
  train_dataset_builder = train_base_dataset._dataset_builder
  train_ds = input_utils.get_data(
    dataset=train_dataset_builder,
    split=train_split,
    rng=train_ds_rng,
    process_batch_size=local_batch_size,
    preprocess_fn=preproc_fn,
    shuffle_buffer_size=config.shuffle_buffer_size,
    prefetch_size=config.get('prefetch_to_host', 2),
    data_dir=config.get('data_dir'))

  # Start prefetching already.
  train_iter = input_utils.start_input_pipeline(
    train_ds, config.get('prefetch_to_device', 1))

  write_note('Initializing val dataset(s)...')

  def _get_val_split(dataset,
                     split,
                     pp_eval,
                     data_dir=None):
    del pp_eval  # Same as pp_train for Diabetic Retinopathy.

    # We do ceil rounding such that we include the last incomplete batch.
    nval_img = input_utils.get_num_examples(
      dataset,
      split=split,
      process_batch_size=local_batch_size_eval,
      drop_remainder=False,
      data_dir=fillin(data_dir))
    val_steps = int(np.ceil(nval_img / batch_size_eval))
    logging.info('Running validation for %d steps for %s, %s', val_steps,
                 dataset, split)
    val_ds = input_utils.get_data(
      dataset=dataset,
      split=split,
      rng=None,
      process_batch_size=local_batch_size_eval,
      preprocess_fn=preproc_fn,
      cache=False,
      repeat_after_batching=True,
      shuffle=False,
      prefetch_size=config.get('prefetch_to_host', 2),
      drop_remainder=False,
      data_dir=config.get('data_dir'))
    val_iter = input_utils.start_input_pipeline(
      val_ds, config.get('prefetch_to_device', 1))
    return val_iter, val_steps

  # Load in-domain and OOD validation datasets.
  # Please specify the desired shift (Country Shift or Severity Shift)
  # in the config.
  in_domain_val_base_dataset = ub.datasets.get(
    in_domain_dataset, split=in_domain_val_split,
    data_dir=config.get('data_dir'))
  in_domain_val_dataset_builder = in_domain_val_base_dataset._dataset_builder
  ood_val_base_dataset = ub.datasets.get(
    ood_dataset, split=ood_val_split,
    data_dir=config.get('data_dir'))
  ood_val_dataset_builder = ood_val_base_dataset._dataset_builder
  val_iter_splits = {
    'in_domain_validation': _get_val_split(
      in_domain_val_dataset_builder,
      in_domain_val_split,
      pp_eval=config.pp_eval,
      data_dir=config.get('data_dir')),
    'ood_validation': _get_val_split(
      ood_val_dataset_builder,
      ood_val_split,
      pp_eval=config.pp_eval,
      data_dir=config.get('data_dir'))
  }
  ntrain_img = input_utils.get_num_examples(
    train_dataset_builder,
    split=train_split,
    process_batch_size=local_batch_size,
    data_dir=config.get('data_dir'))
  steps_per_epoch = ntrain_img / batch_size
  if config.get('num_epochs'):
    total_steps = int(config.num_epochs * steps_per_epoch)
    assert not config.get(
      'total_steps'), 'Set either num_epochs or total_steps'
  else:
    total_steps = FLAGS.total_steps

  logging.info(
    'Running for %d steps, that means %f epochs and %f steps per epoch',
    total_steps, total_steps * batch_size / ntrain_img, steps_per_epoch)

  write_note('Initializing model...')
  logging.info('config.model = %s', config.get('model'))

  # Specify Gaussian process layer configs.
  use_gp_layer = config.get('use_gp_layer', True)
  gp_config = config.get('gp_layer', {})
  gp_layer_kwargs = get_gp_kwargs(gp_config)

  # Process ViT backbone model configs.
  vit_kwargs = config.get('model')

  model = ub.models.vision_transformer_gp(
      num_classes=config.num_classes,
      use_gp_layer=use_gp_layer,
      vit_kwargs=vit_kwargs,
      gp_layer_kwargs=gp_layer_kwargs)

  # We want all parameters to be created in host RAM, not on any device, they'll
  # be sent there later as needed, otherwise we already encountered two
  # situations where we allocate them twice.
  @partial(jax.jit, backend='cpu')
  def init(rng):
    image_size = tuple(train_ds.element_spec['image'].shape[2:])
    logging.info('image_size = %s', image_size)
    dummy_input = jnp.zeros((local_batch_size,) + image_size, jnp.float32)
    variables = model.init(rng, dummy_input, train=False)
    # Split model parameters into trainable and untrainable collections.
    states, params = variables.pop('params')
    del variables

    # Set bias in the head to a low value, such that loss is small initially.
    params = flax.core.unfreeze(params)
    if use_gp_layer:
      # Modify the head parameter in the GP head.
      params['head']['output_layer']['bias'] = jnp.full_like(
          params['head']['output_layer']['bias'],
          config.get('init_head_bias', 0))
    else:
      params['head']['bias'] = jnp.full_like(
          params['head']['bias'], config.get('init_head_bias', 0))

    return params, states

  rng, rng_init = jax.random.split(rng)
  params_cpu, states_cpu = init(rng_init)

  if jax.process_index() == 0:
    num_params = sum(p.size for p in jax.tree_flatten(params_cpu)[0])
    parameter_overview.log_parameter_overview(params_cpu)
    writer.write_scalars(step=0, scalars={'num_params': num_params})

  @partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, states, images, labels):
    variable_dict = {'params': flax.core.freeze(params), **states}
    logits, out = model.apply(
        variable_dict,
        images,
        train=False,
        mean_field_factor=gp_config.get('mean_field_factor', -1.))
    losses = getattr(train_utils, config.get('loss', 'softmax_xent'))(
      logits=logits, labels=labels, reduction=False)
    loss = jax.lax.psum(losses, axis_name='batch')
    top1_idx = jnp.argmax(logits, axis=1)

    # Extracts the label at the highest logit index for each image.
    top1_correct = jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0]

    ncorrect = jax.lax.psum(top1_correct, axis_name='batch')
    n = batch_size_eval
    metric_args = jax.lax.all_gather([
      logits, labels, out['pre_logits']], axis_name='batch')
    return ncorrect, loss, n, metric_args

  # Load the optimizer from flax.
  opt_name = config.get('optim_name')
  write_note(f'Initializing {opt_name} optimizer...')
  opt_def = getattr(flax.optim, opt_name)(**config.get('optim', {}))

  # We jit this, such that the arrays that are created are created on the same
  # device as the input is, in this case the CPU. Else they'd be on device[0].
  opt_cpu = jax.jit(opt_def.create)(params_cpu)

  weight_decay_rules = weight_decay or []
  rescale_value = FLAGS.lr_base if config.get('weight_decay_decouple') else 1.
  weight_decay_fn = train_utils.get_weight_decay_fn(
      weight_decay_rules=weight_decay_rules, rescale_value=rescale_value)

  @partial(jax.pmap, axis_name='batch', donate_argnums=(0,))
  def update_fn(opt, states, lr, reset_covmat, images, labels, rng):
    """Update step."""
    measurements = {}

    # Get device-specific loss rng.
    rng, rng_model = jax.random.split(rng, 2)
    rng_model_local = jax.random.fold_in(rng_model, jax.lax.axis_index('batch'))

    def loss_fn(params, states, images, labels):
      # Specify mutable collection to update untrainable GP parameters.
      variable_dict = {'params': flax.core.freeze(params), **states}
      model_results, updated_states = model.apply(
          variable_dict,
          images,
          train=True,
          rngs={'dropout': rng_model_local},
          mutable=list(states.keys()),
          mean_field_factor=gp_config.get('mean_field_factor', -1.))

      logits, _ = model_results
      loss = getattr(train_utils, config.get('loss', 'sigmoid_xent'))(
          logits=logits, labels=labels)
      return loss, updated_states

    # Performs exact covariance update (i.e., reset precision matrix resetting
    # at begining of new epoch) if covmat_momentum is a null value.
    if use_gp_layer and gp_config.get('covmat_momentum', -1.) < 0:
      # Resets precision matrix to Identity * ridge_penalty if at the begining
      # of a new epoch. This should be done before accumulate gradient.
      ridge_penalty = gp_config.get('ridge_penalty', 1.)
      prec_mat_old = states['laplace_covariance']['head']['covmat_layer'][
        'precision_matrix']
      prec_mat_new = (
          (1. - reset_covmat) * prec_mat_old +
          reset_covmat * jnp.eye(prec_mat_old.shape[0]) * ridge_penalty)

      states = flax.core.unfreeze(states)
      states['laplace_covariance']['head']['covmat_layer'][
        'precision_matrix'] = prec_mat_new
      states = flax.core.freeze(states)

    # Implementation considerations compared and summarized at
    # https://docs.google.com/document/d/1g3kMEvqu1DOawaflKNyUsIoQ4yIVEoyE5ZlIPkIl4Lc/edit?hl=en#
    (l, s), g = accumulate_gradient_with_states(
        jax.value_and_grad(loss_fn, has_aux=True), opt.target, states, images,
        labels, config.get('grad_accum_steps'))
    l, g = jax.lax.pmean((l, g), axis_name='batch')

    # Log the gradient norm only if we need to compute it anyways (clipping)
    # or if we don't use grad_accum_steps, as they interact badly.
    if config.get('grad_accum_steps', 1) == 1 or grad_clip_norm is not None:
      grads, _ = jax.tree_flatten(g)
      l2_g = jnp.sqrt(sum([jnp.vdot(p, p) for p in grads]))
      measurements['l2_grads'] = l2_g

    # Optionally resize the global gradient to a maximum norm. We found this
    # useful in some cases across optimizers, hence it's in the main loop.
    if grad_clip_norm is not None:
      g_factor = jnp.minimum(1.0, grad_clip_norm / l2_g)
      g = jax.tree_map(lambda p: g_factor * p, g)
    opt = opt.apply_gradient(g, learning_rate=lr)
    opt = opt.replace(target=weight_decay_fn(opt.target, lr))

    params, _ = jax.tree_flatten(opt.target)
    measurements['l2_params'] = jnp.sqrt(sum([jnp.vdot(p, p) for p in params]))
    measurements['reset_covmat'] = reset_covmat

    return opt, s, l, rng, measurements

    #
    #
    # decay_rules = weight_decay or []
    # if isinstance(decay_rules, numbers.Number):
    #   decay_rules = [('.*kernel.*', decay_rules)]
    # # sched_m = lr / config.lr.base if config.get(
    # #   'weight_decay_decouple') else lr
    # sched_m = lr / FLAGS.lr_base if config.get(
    #   'weight_decay_decouple') else lr
    #
    # def decay_fn(v, wd):
    #   return (1.0 - sched_m * wd) * v
    #
    # opt = opt.replace(
    #   target=train_utils.tree_map_with_regex(decay_fn, opt.target,
    #                                          decay_rules))
    #
    # params, _ = jax.tree_flatten(opt.target)
    # measurements['l2_params'] = jnp.sqrt(
    #   sum([jnp.vdot(p, p) for p in params]))
    #
    # return opt, l, rng, measurements

  default_reinit_params = ('head/output_layer/kernel', 'head/output_layer/bias',
                           'head/kernel', 'head/bias')
  rng, train_loop_rngs = jax.random.split(rng)
  checkpoint_data = checkpoint_utils.maybe_load_checkpoint(
      train_loop_rngs=train_loop_rngs,
      save_checkpoint_path=save_checkpoint_path,
      init_optimizer=opt_cpu,
      init_params=params_cpu,
      init_fixed_model_states=states_cpu,
      default_reinit_params=default_reinit_params,
      config=config)
  train_loop_rngs = checkpoint_data.train_loop_rngs
  opt_cpu = checkpoint_data.optimizer
  states_cpu = checkpoint_data.fixed_model_states
  accumulated_train_time = checkpoint_data.accumulated_train_time

  write_note('Adapting the checkpoint model...')
  adapted_params = checkpoint_utils.adapt_upstream_architecture(
      init_params=params_cpu,
      loaded_params=opt_cpu.target)
  opt_cpu = opt_cpu.replace(target=adapted_params)

  write_note('Kicking off misc stuff...')
  first_step = int(opt_cpu.state.step)  # Might be a DeviceArray type.
  if first_step == 0 and jax.process_index() == 0:
    writer.write_hparams(dict(config))
  chrono = train_utils.Chrono(first_step, total_steps, batch_size,
                              accumulated_train_time)
  # Note: switch to ProfileAllHosts() if you need to profile all hosts.
  # (Xprof data become much larger and take longer to load for analysis)
  profiler = periodic_actions.Profile(
      # Create profile after every restart to analyze pre-emption related
      # problems and assure we get similar performance in every run.
      logdir=output_dir, first_profile=first_step + 10)

  # Prepare the learning-rate and pre-fetch it to device to avoid delays.
  lr_fn = train_utils.create_learning_rate_schedule(total_steps, **lr_dict)

  # TODO(dusenberrymw): According to flax docs, prefetching shouldn't be
  # necessary for TPUs.
  lr_iter = train_utils.prefetch_scalar(
      map(lr_fn, range(total_steps)), config.get('prefetch_to_device', 1))

  # Prepare the precision matrix resetting schedule, and pre-fetch it to device.
  reset_covmat_fn = lambda step: float(step % steps_per_epoch == 0)
  reset_covmat_iter = train_utils.prefetch_scalar(
      map(reset_covmat_fn, range(first_step, total_steps)),
      nprefetch=config.get('prefetch_to_device', 1))

  write_note(f'Replicating...\n{chrono.note}')
  opt_repl = flax_utils.replicate(opt_cpu)
  states_repl = flax_utils.replicate(states_cpu)

  checkpoint_writer = None

  # Note: we return the train loss, val loss, and fewshot best l2s for use in
  # reproducibility unit tests.
  train_loss = -jnp.inf
  val_loss = -jnp.inf
  results = {'dummy': {(0, 1): -jnp.inf}}

  write_note(f'First step compilations...\n{chrono.note}')
  logging.info('first_step = %s', first_step)
  # Advance the iterators if we are restarting from an earlier checkpoint.
  # TODO(dusenberrymw): Look into checkpointing dataset state instead.

  # Makes sure log_eval_steps is same as steps_per_epoch. This is because
  # the precision matrix needs to be updated fully (at the end of each epoch)
  # when eval takes place.
  log_eval_steps = steps_per_epoch
  if first_step > 0:
    write_note('Advancing iterators after resuming from a checkpoint...')
    lr_iter = itertools.islice(lr_iter, first_step, None)
    train_iter = itertools.islice(train_iter, first_step, None)
    # NOTE: Validation eval is only run on certain steps, so determine how many
    # times it was run previously.
    num_val_runs = sum(
        map(
            lambda i: train_utils.itstime(i, log_eval_steps, total_steps
                                          ), range(1, first_step + 1)))
    for val_name, (val_iter, val_steps) in val_iter_splits.items():
      val_iter = itertools.islice(val_iter, num_val_runs * val_steps, None)
      val_iter_splits[val_name] = (val_iter, val_steps)

  # Using a python integer for step here, because opt.state.step is allocated
  # on TPU during replication.
  for step, train_batch, lr_repl, reset_covmat_repl in zip(
      range(first_step + 1, total_steps + 1), train_iter, lr_iter,
      reset_covmat_iter):

    with jax.profiler.TraceAnnotation('train_step', step_num=step, _r=1):
      # TODO(jereliu): Expand to allow precision matrix resetting.
      (opt_repl, states_repl, loss_value, train_loop_rngs,
       extra_measurements) = update_fn(
           opt_repl,
           states_repl,
           lr_repl,
           reset_covmat_repl,
           train_batch['image'],
           train_batch['labels'],
           rng=train_loop_rngs)
    if jax.process_index() == 0:
      profiler(step)

    # Checkpoint saving
    if train_utils.itstime(
        step, config.get('checkpoint_steps'), total_steps, process=0):
      write_note('Checkpointing...')
      chrono.pause()
      train_utils.checkpointing_timeout(checkpoint_writer,
                                        config.get('checkpoint_timeout', 1))
      accumulated_train_time = chrono.accum_train_time
      # We need to transfer the weights over now or else we risk keeping them
      # alive while they'll be updated in a future step, creating hard to debug
      # memory errors (see b/160593526). Also, takes device 0's params only.
      # For GP layer, we will also do the same for untrainable parameters
      # (`states`). This is ok since `random features` are frozen throughout
      # pre-training, and `precision matrix` is a finetuning-specific parameters
      # that will be re-learned in the finetuning task.
      opt_cpu = jax.tree_map(lambda x: np.array(x[0]), opt_repl)
      states_cpu = jax.tree_map(lambda x: np.array(x[0]), states_repl)

      # Check whether we want to keep a copy of the current checkpoint.
      copy_step = None
      if train_utils.itstime(step, config.get('keep_checkpoint_steps'),
                             total_steps):
        write_note('Keeping a checkpoint copy...')
        copy_step = step

      # Checkpoint should be a nested dictionary or FLAX datataclasses from
      # `flax.struct`. Both can be present in a checkpoint.
      checkpoint_data = checkpoint_utils.CheckpointData(
          optimizer=opt_cpu,
          fixed_model_states=states_cpu,
          train_loop_rngs=train_loop_rngs,
          accumulated_train_time=accumulated_train_time)
      checkpoint_writer = pool.apply_async(
          checkpoint_utils.checkpoint_trained_model,
          (checkpoint_data, save_checkpoint_path, copy_step))
      chrono.resume()

    # Report training progress
    if train_utils.itstime(
        step, config.log_training_steps, total_steps, process=0):
      write_note('Reporting training progress...')
      train_loss = loss_value[0]  # Keep to return for reproducibility tests.
      timing_measurements, note = chrono.tick(step)
      write_note(note)
      train_measurements = {}
      train_measurements.update({
          'learning_rate': lr_repl[0],
          'training_loss': train_loss,
      })
      train_measurements.update(flax.jax_utils.unreplicate(extra_measurements))
      train_measurements.update(timing_measurements)
      writer.write_scalars(step, train_measurements)

    # Report validation performance
    if train_utils.itstime(step, config.log_eval_steps, total_steps):
      write_note('Evaluating on the validation set...')
      chrono.pause()

      all_val_results = {}

      for val_name, (val_iter, val_steps) in val_iter_splits.items():
        start_time = time()

        # Runs evaluation loop.
        results_arrs = {
          'y_true': [],
          'y_pred': [],
          'y_pred_entropy': []
        }

        for _, batch in zip(range(val_steps), val_iter):
          batch_ncorrect, batch_losses, batch_n, batch_metric_args = (
            evaluation_fn(
              opt_repl.target, states_repl, batch['image'], batch['labels']))

          # All results are a replicated array shaped as follows:
          # (local_devices, per_device_batch_size, elem_shape...)
          # with each local device's entry being identical as they got psum'd.
          # So let's just take the first one to the host as numpy.

          # Here we parse batch_metric_args to compute uncertainty metrics.
          logits, labels, _ = batch_metric_args
          logits = np.array(logits[0])
          probs = jax.nn.softmax(logits)

          # From one-hot to integer labels.
          int_labels = np.argmax(np.array(labels[0]), axis=-1)

          probs = np.reshape(probs, (probs.shape[0] * probs.shape[1], -1))
          int_labels = int_labels.flatten()
          y_pred = probs[:, 1]
          results_arrs['y_true'].append(int_labels)
          results_arrs['y_pred'].append(y_pred)

          # Entropy is computed at the per-epoch level (see below).
          results_arrs['y_pred_entropy'].append(probs)

        results_arrs['y_true'] = np.concatenate(results_arrs['y_true'],
                                                axis=0)
        results_arrs['y_pred'] = np.concatenate(
          results_arrs['y_pred'], axis=0).astype('float64')
        results_arrs['y_pred_entropy'] = entropy(
          np.concatenate(results_arrs['y_pred_entropy'], axis=0), axis=-1)

        time_elapsed = time() - start_time
        results_arrs['total_ms_elapsed'] = time_elapsed * 1e3
        results_arrs['dataset_size'] = val_steps * batch_size_eval

        all_val_results[val_name] = results_arrs

      per_pred_results, total_results = evaluate_vit_predictions(
        dataset_split_to_containers=all_val_results,
        is_deterministic=True,
        num_bins=15,
        return_per_pred_results=True
      )

      # Optionally log to wandb
      if FLAGS.use_wandb:
        wandb.log(total_results, step=step)

      # Save per-prediction metrics
      save_per_prediction_results(
        output_dir, step, per_pred_results, verbose=False)

      chrono.resume()

      # End of step.
    if config.get('testing_failure_step'):
      # Break early to simulate infra failures in test cases.
      if config.testing_failure_step == step:
        break

  write_note(f'Done!\n{chrono.note}')
  pool.close()
  pool.join()
  writer.close()

  if wandb_run is not None:
    wandb_run.finish()

  # Return final training loss, validation loss, and fewshot results for
  # reproducibility test cases.
  # return train_loss, val_loss, results
  # TODO(nband): fix result reporting for DR ViT-16 reproducibility unit tests


if __name__ == '__main__':
  app.run(main)
