# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Wide ResNet 28-10 trained on CIFAR10/100-N with access to PI first, and distilled without it afterwards.

Hyperparameters differ slightly from the original paper's code
(https://github.com/szagoruyko/wide-residual-networks) as TensorFlow uses, for
example, l2 instead of weight decay, and a different parameterization for SGD's
momentum.

D. Lopez-Paz, L. Bottou, B. Schoelkopf, and V. Vapnik. "Unifying distillation
and privileged information." ICLR 2016
"""

import os
import time
from absl import app
from absl import flags
from absl import logging
import robustness_metrics as rm
import tensorflow as tf
import tensorflow_datasets as tfds
import uncertainty_baselines as ub
import ood_utils  # local file import from baselines.cifar
import utils  # local file import from baselines.cifar
from tensorboard.plugins.hparams import api as hp

flags.DEFINE_float('label_smoothing', 0., 'Label smoothing parameter in [0,1].')
flags.register_validator(
    'label_smoothing',
    lambda ls: ls >= 0 and ls <= 1,
    message='--label_smoothing must be in [0, 1].')

flags.DEFINE_float(
    'max_validation_proportion', 0.05,
    'Maximum proportion of the training set to use as validation set.')
flags.register_validator(
    'max_validation_proportion',
    lambda p: p >= 0 and p < 1,
    message='--max_validation_proportion must be in [0, 1).')

# Data Augmentation flags.
flags.DEFINE_bool('augmix', False,
                  'Whether to perform AugMix [4] on the input data.')
flags.DEFINE_integer(
    'aug_count', 1, 'Number of augmentation operations in AugMix to perform '
    'on the input image. In the simgle model context, it'
    'should be 1. In the ensembles context, it should be'
    'ensemble_size if we perform random_augment only; It'
    'should be (ensemble_size - 1) if we perform augmix.')
flags.DEFINE_float('augmix_prob_coeff', 0.5, 'Augmix probability coefficient.')
flags.DEFINE_integer(
    'augmix_depth', -1,
    'Augmix depth, -1 meaning sampled depth. This corresponds'
    'to line 7 in the Algorithm box in [4].')
flags.DEFINE_integer(
    'augmix_width', 3,
    'Augmix width. This corresponds to the k in line 5 in the'
    'Algorithm box in [4].')

# Fine-grained specification of the hyperparameters (used when FLAGS.l2 is None)
flags.DEFINE_float('bn_l2', None, 'L2 reg. coefficient for batch-norm layers.')
flags.DEFINE_float('input_conv_l2', None,
                   'L2 reg. coefficient for the input conv layer.')
flags.DEFINE_float('group_1_conv_l2', None,
                   'L2 reg. coefficient for the 1st group of conv layers.')
flags.DEFINE_float('group_2_conv_l2', None,
                   'L2 reg. coefficient for the 2nd group of conv layers.')
flags.DEFINE_float('group_3_conv_l2', None,
                   'L2 reg. coefficient for the 3rd group of conv layers.')
flags.DEFINE_float('dense_kernel_l2', None,
                   'L2 reg. coefficient for the kernel of the dense layer.')
flags.DEFINE_float('dense_bias_l2', None,
                   'L2 reg. coefficient for the bias of the dense layer.')

flags.DEFINE_bool('collect_profile', False,
                  'Whether to trace a profile with tensorboard')

# Model flags
flags.DEFINE_integer('model_width_multiplier', 10,
                     'Factor used to resize the width of the model.')
flags.DEFINE_integer('model_depth', 28, 'Depth of the model.')
flags.DEFINE_integer('pi_tower_width', 128,
                     'Width of the hidden layers in the PI tower.')

# OOD flags.
flags.DEFINE_bool('eval_only', False,
                  'Whether to run only eval and (maybe) OOD steps.')
flags.DEFINE_bool('eval_on_ood', False,
                  'Whether to run OOD evaluation on specified OOD datasets.')
flags.DEFINE_list('ood_dataset', 'cifar100,svhn_cropped',
                  'list of OOD datasets to evaluate on.')
flags.DEFINE_string('saved_model_dir', None,
                    'Directory containing the saved model checkpoints.')
flags.DEFINE_bool('dempster_shafer_ood', False,
                  'Wheter to use DempsterShafer Uncertainty score.')

# CIFAR-PI flags.
flags.DEFINE_integer('num_annotators_per_example', 3,
                     'Number of annotators to load per training example.')
flags.DEFINE_enum(
    'annotator_sampling_strategy',
    'uniform', ['uniform', 'best', 'worst'],
    help='Strategy used to sample annotators.')
flags.DEFINE_integer(
    'num_annotators_per_example_and_step', None,
    'Number of annotators to load per example and step during training.')
flags.DEFINE_float(
    'max_reliability', 1.,
    'Maximum reliability threshold of the annotators. Any annotator with a higher reliability is discarded.'
)
flags.DEFINE_float(
    'min_reliability', 0.,
    'Minimum reliability threshold of the annotators. Any annotator with a lower reliability is discarded.'
)
flags.DEFINE_integer(
    'reliability_estimation_batch_size', 4096,
    'Number of examples to process in parallel when estimating reliability of ImageNet-PI.'
)
flags.DEFINE_bool(
    'disable_reliability_estimation', False,
    'Whether to disable the expensive reliability estimation step when loading the data.'
)
flags.DEFINE_bool(
    'use_annotator_labels', True,
    'Whether or not to use the annotation labels in the cross-entropy loss.')
flags.DEFINE_string(
    'pi_subset', 'annotator_ids,annotator_times',
    'Comma-separated string defining subset of additional information to use as PI.'
)
flags.DEFINE_integer('random_pi_length', 1, 'Length of random PI features.')
flags.DEFINE_integer('pi_seed', 42,
                     'Random seed to control PI processing randomness.')
flags.DEFINE_integer('num_adversarial_annotators_per_example', 0,
                     'Number of adversarial annotators to add to the dataset.')

# Distillation flags.
flags.DEFINE_float(
    'distillation_loss_weight', 0.5,
    'Weight of the distillation loss in the total loss during the distillation phase of training.'
)
flags.register_validator(
    'distillation_loss_weight',
    lambda dw: dw >= 0 and dw <= 1,
    message='--distillation_loss_weight must be in [0, 1].')
flags.DEFINE_float(
    'distillation_temperature', 1.,
    'Temperature parameter used to convert the logits of the teacher into soft-labels for distillation.'
)
flags.DEFINE_integer('distillation_epochs', 90,
                     'Number of epochs of the distillation phase.')

FLAGS = flags.FLAGS


def _extract_hyperparameter_dictionary():
  """Create the dictionary of hyperparameters from FLAGS."""
  flags_as_dict = FLAGS.flag_values_dict()
  hp_keys = ub.models.get_wide_resnet_hp_keys()
  hps = {k: flags_as_dict[k] for k in hp_keys}
  return hps


def _get_split_from_training_set(from_=None, to=None):
  """Define split as a tfds.core.ReadInstruction to avoid raising exceptions because we use the CIFAR10-H test set to train.
  """
  return tfds.core.ReadInstruction(
      split_name='train' if FLAGS.dataset != 'cifar10h' else 'test',
      from_=from_,
      to=to,
      unit='%')


def main(argv):
  fmt = '[%(filename)s:%(lineno)s] %(message)s'
  formatter = logging.PythonFormatter(fmt)
  logging.get_absl_handler().setFormatter(formatter)
  del argv  # unused arg

  pretrain_checkpoint_dir = f'{FLAGS.output_dir}/pretrain'
  distill_checkpoint_dir = f'{FLAGS.output_dir}/distill'
  tf.io.gfile.makedirs(pretrain_checkpoint_dir)
  tf.io.gfile.makedirs(distill_checkpoint_dir)
  logging.info('Saving checkpoints at %s', FLAGS.output_dir)
  tf.random.set_seed(FLAGS.seed)

  data_dir = FLAGS.data_dir
  if FLAGS.use_gpu:
    logging.info('Use GPU')
    strategy = tf.distribute.MirroredStrategy()
  else:
    logging.info('Use TPU at %s',
                 FLAGS.tpu if FLAGS.tpu is not None else 'local')
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
    tf.config.experimental_connect_to_cluster(resolver)
    tf.tpu.experimental.initialize_tpu_system(resolver)
    strategy = tf.distribute.TPUStrategy(resolver)

  batch_size = FLAGS.per_core_batch_size * FLAGS.num_cores

  aug_params = {
      'augmix': FLAGS.augmix,
      'aug_count': FLAGS.aug_count,
      'augmix_depth': FLAGS.augmix_depth,
      'augmix_prob_coeff': FLAGS.augmix_prob_coeff,
      'augmix_width': FLAGS.augmix_width,
  }

  # Note that stateless_{fold_in,split} may incur a performance cost, but a
  # quick side-by-side test seemed to imply this was minimal.
  seeds = tf.random.experimental.stateless_split([FLAGS.seed, FLAGS.seed + 1],
                                                 3)[:, 0]

  validation_proportion = min(1.0 - FLAGS.train_proportion,
                              FLAGS.max_validation_proportion)
  logging.info('Setting aside %f%% of the training set for validation',
               validation_proportion * 100)
  # The train_split goes up to train_proportion.
  train_split = _get_split_from_training_set(to=FLAGS.train_proportion * 100)

  train_builder = ub.datasets.get(
      FLAGS.dataset,
      data_dir=data_dir,
      download_data=FLAGS.download_data,
      split=train_split,
      seed=seeds[0],
      aug_params=aug_params,
      shuffle_buffer_size=FLAGS.shuffle_buffer_size,
      num_annotators_per_example=FLAGS.num_annotators_per_example,
      num_annotators_per_example_and_step=FLAGS
      .num_annotators_per_example_and_step,
      reliability_interval=(FLAGS.min_reliability, FLAGS.max_reliability),
      reliability_estimation_batch_size=FLAGS.reliability_estimation_batch_size,
      is_training=True,  # We need to set this to train on test set of CIFAR10-H
      pi_seed=FLAGS.pi_seed,
      num_adversarial_annotators_per_example=FLAGS
      .num_adversarial_annotators_per_example,
      annotator_sampling_strategy=FLAGS.annotator_sampling_strategy,
      random_pi_length=FLAGS.random_pi_length,
      disable_reliability_estimation=FLAGS.disable_reliability_estimation,
  )
  train_dataset = train_builder.load(batch_size=batch_size)

  # Load one example to infer pi_shape.
  dummy_example = train_builder.load(batch_size=1).take(1).get_single_element()

  validation_dataset = None
  steps_per_validation = 0
  if FLAGS.train_proportion < 1.0:

    # The validation_split starts from train_proportion.
    validation_split = _get_split_from_training_set(
        from_=-validation_proportion * 100)

    validation_builder = ub.datasets.get(
        FLAGS.dataset,
        split=validation_split,
        data_dir=data_dir,
        drop_remainder=FLAGS.drop_remainder_for_eval,
        num_annotators_per_example=FLAGS.num_annotators_per_example,
        num_annotators_per_example_and_step=FLAGS
        .num_annotators_per_example_and_step,
        reliability_interval=(FLAGS.min_reliability, FLAGS.max_reliability),
        reliability_estimation_batch_size=FLAGS
        .reliability_estimation_batch_size,
        is_training=False,
        pi_seed=FLAGS.pi_seed,
        num_adversarial_annotators_per_example=FLAGS
        .num_adversarial_annotators_per_example,
        annotator_sampling_strategy=FLAGS.annotator_sampling_strategy,
        disable_reliability_estimation=FLAGS.disable_reliability_estimation,
    )
    validation_dataset = validation_builder.load(batch_size=batch_size)
    validation_dataset = strategy.experimental_distribute_dataset(
        validation_dataset)
    steps_per_validation = validation_builder.num_examples // batch_size
  clean_test_builder = ub.datasets.get(
      'cifar10' if FLAGS.dataset in ['cifar10n', 'cifar10h'] else 'cifar100',
      split=tfds.Split.TEST
      if FLAGS.dataset != 'cifar10h' else tfds.Split.TRAIN,
      data_dir=data_dir,
      drop_remainder=FLAGS.drop_remainder_for_eval,
      is_training=False)
  clean_test_dataset = clean_test_builder.load(batch_size=batch_size)

  test_datasets = {
      'clean': strategy.experimental_distribute_dataset(clean_test_dataset),
  }

  train_dataset = strategy.experimental_distribute_dataset(train_dataset)

  steps_per_epoch = train_builder.num_examples // batch_size
  steps_per_eval = clean_test_builder.num_examples // batch_size
  num_classes = 100 if FLAGS.dataset == 'cifar100n' else 10

  if FLAGS.eval_on_ood:
    ood_dataset_names = FLAGS.ood_dataset
    ood_ds, steps_per_ood = ood_utils.load_ood_datasets(
        ood_dataset_names,
        clean_test_builder,
        validation_proportion,
        batch_size,
        drop_remainder=FLAGS.drop_remainder_for_eval)
    ood_datasets = {
        name: strategy.experimental_distribute_dataset(ds)
        for name, ds in ood_ds.items()
    }

  if FLAGS.corruptions_interval > 0:
    if FLAGS.dataset == 'cifar100':
      data_dir = FLAGS.cifar100_c_path
    corruption_types, _ = utils.load_corrupted_test_info(FLAGS.dataset)
    for corruption_type in corruption_types:
      for severity in range(1, 6):
        dataset = ub.datasets.get(
            f'{FLAGS.dataset}_corrupted',
            corruption_type=corruption_type,
            severity=severity,
            split=tfds.Split.TEST,
            data_dir=data_dir).load(batch_size=batch_size)
        test_datasets[f'{corruption_type}_{severity}'] = (
            strategy.experimental_distribute_dataset(dataset))

  def label_encoding_fn(labels):
    return tf.one_hot(
        tf.cast(labels, dtype=tf.int32), num_classes, dtype=tf.float32)

  def clean_labels_encoding(example):
    return pi_utils.clean_labels_encoding_fn(
        example=example,
        # pytype: disable=attribute-error
        num_annotators_per_example=train_builder
        .num_annotators_per_example_and_step,
        # pytype: enable=attribute-error
        label_encoding_fn=label_encoding_fn)

  def is_annotator_incorrect_encoding(example):
    return pi_utils.find_noisy_annotators(example, flatten_annotators=False)

  def annotator_label_if_incorrect_encoding(example):
    return pi_utils.annotator_label_if_incorrect_encoding_fn(
        example=example, label_encoding_fn=label_encoding_fn)

  def annotator_ids_encoding(example):
    # pytype: disable=attribute-error
    return pi_utils.annotator_ids_encoding_fn(
        example=example,
        num_dataset_annotators=train_builder
        .num_dataset_annotators)
    # pytype: enable=attribute-error

  def annotator_labels_encoding(example):
    return label_encoding_fn(example['pi_features']['annotator_labels'])

  encoding_fn_dict = {
      'clean_labels': clean_labels_encoding,
      'is_annotator_incorrect': is_annotator_incorrect_encoding,
      'annotator_label_if_incorrect': annotator_label_if_incorrect_encoding,
      'annotator_ids': annotator_ids_encoding,
      'annotator_labels': annotator_labels_encoding,
      'annotator_times': lambda e: e['pi_features']['annotator_times'],
      'trial_idx': lambda e: e['pi_features']['trial_idx'],
      'random_pi': lambda e: e['pi_features']['random_pi'],
  }
  privileged_information_fn = pi_utils.get_privileged_information_fn(
      pi_subset=FLAGS.pi_subset.split(','), encoding_fn_dict=encoding_fn_dict)

  pi_shape = privileged_information_fn(dummy_example).shape[1:]

  summary_writer = tf.summary.create_file_writer(
      os.path.join(FLAGS.output_dir, 'summaries'))

  with strategy.scope():
    logging.info('Building ResNet model with access to PI')
    teacher_model = ub.models.wide_resnet_pi_access(  # type: ignore
        input_shape=(32, 32, 3),
        depth=FLAGS.model_depth,
        width_multiplier=FLAGS.model_width_multiplier,
        pi_tower_width=FLAGS.pi_tower_width,
        num_classes=num_classes,
        l2=FLAGS.l2,
        hps=_extract_hyperparameter_dictionary(),
        pi_input_shape=pi_shape,
        seed=seeds[1])
    logging.info('Teacher model input shape: %s', teacher_model.input_shape)
    logging.info('Teacher model output shape: %s', teacher_model.output_shape)
    logging.info('Teacher model number of weights: %s',
                 teacher_model.count_params())

    logging.info('Building ResNet student model without access to PI.')
    distill_model = ub.models.wide_resnet(  # type: ignore
        input_shape=(32, 32, 3),
        depth=FLAGS.model_depth,
        width_multiplier=FLAGS.model_width_multiplier,
        num_classes=num_classes,
        l2=FLAGS.l2,
        hps=_extract_hyperparameter_dictionary(),
        seed=seeds[2])
    logging.info('Distillation model input shape: %s.',
                 distill_model.input_shape)
    logging.info('Distillation model output shape: %s.',
                 distill_model.output_shape)
    logging.info('Distillation model number of weights: %s.',
                 distill_model.count_params())

    def create_optimizer(epochs):
      base_lr = FLAGS.base_learning_rate * batch_size / 128
      lr_decay_epochs = [(int(start_epoch_str) * epochs) // 200
                         for start_epoch_str in FLAGS.lr_decay_epochs]
      lr_schedule = ub.schedules.WarmUpPiecewiseConstantSchedule(
          steps_per_epoch,
          base_lr,
          decay_ratio=FLAGS.lr_decay_ratio,
          decay_epochs=lr_decay_epochs,
          warmup_epochs=FLAGS.lr_warmup_epochs)
      return tf.keras.optimizers.SGD(
          lr_schedule, momentum=1.0 - FLAGS.one_minus_momentum, nesterov=True)

    pretrain_optimizer = create_optimizer(FLAGS.train_epochs)
    distill_optimizer = create_optimizer(FLAGS.distillation_epochs)

    metrics = {}
    for training_phase in ['pretrain', 'distill']:
      metrics.update({
          f'{training_phase}/train/loss':
              tf.keras.metrics.Mean(),
          f'{training_phase}/train/negative_log_likelihood':
              tf.keras.metrics.Mean(),
      })
      for noise_split in ['', '_clean', '_noisy']:
        metrics.update({
            f'{training_phase}/train/accuracy{noise_split}':
                tf.keras.metrics.SparseCategoricalAccuracy(),
            f'{training_phase}/train/ece{noise_split}':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        })
    metrics.update({
        'distill/train/negative_log_likelihood_distill':
            tf.keras.metrics.Mean()
    })

    metrics.update({
        'distill/test/negative_log_likelihood':
            tf.keras.metrics.Mean(),
        'distill/test/accuracy':
            tf.keras.metrics.SparseCategoricalAccuracy(),
        'distill/test/ece':
            rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
    })
    if validation_dataset:
      for validation_type in ['validation_clean', 'validation_noisy']:
        metrics.update({
            f'distill/{validation_type}/negative_log_likelihood':
                tf.keras.metrics.Mean(),
            f'distill/{validation_type}/accuracy':
                tf.keras.metrics.SparseCategoricalAccuracy(),
            f'distill/{validation_type}/ece':
                rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins),
        })
    if FLAGS.eval_on_ood:
      ood_metrics = ood_utils.create_ood_metrics(ood_dataset_names)
      metrics.update(ood_metrics)
    if FLAGS.corruptions_interval > 0:
      corrupt_metrics = {}
      for intensity in range(1, 6):
        for corruption in corruption_types:
          dataset_name = '{0}_{1}'.format(corruption, intensity)
          corrupt_metrics[f'distill/test/nll_{dataset_name}'] = (
              tf.keras.metrics.Mean())
          corrupt_metrics[f'distill/test/accuracy_{dataset_name}'] = (
              tf.keras.metrics.SparseCategoricalAccuracy())
          corrupt_metrics[f'distill/test/ece_{dataset_name}'] = (
              rm.metrics.ExpectedCalibrationError(num_bins=FLAGS.num_bins))

    pretrain_checkpoint = tf.train.Checkpoint(
        model=teacher_model, optimizer=pretrain_optimizer)
    distill_checkpoint = tf.train.Checkpoint(
        model=distill_model, optimizer=distill_optimizer)
    latest_pretrain_checkpoint = tf.train.latest_checkpoint(
        pretrain_checkpoint_dir)
    latest_distill_checkpoint = tf.train.latest_checkpoint(
        distill_checkpoint_dir)

    def restore_checkpoint(latest_checkpoint, checkpoint, optimizer):
      initial_epoch = 0
      if latest_checkpoint:
        # checkpoint.restore must be within a strategy.scope() so that optimizer
        # slot variables are mirrored.
        checkpoint.restore(latest_checkpoint)
        logging.info('Loaded checkpoint %s', latest_checkpoint)
        initial_epoch = optimizer.iterations.numpy() // steps_per_epoch

      if FLAGS.saved_model_dir:
        logging.info('Saved model dir : %s', FLAGS.saved_model_dir)
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.saved_model_dir)
        checkpoint.restore(latest_checkpoint)
        logging.info('Loaded checkpoint %s', latest_checkpoint)
      if FLAGS.eval_only:
        initial_epoch = FLAGS.train_epochs - 1  # Run just one epoch of eval

      return initial_epoch

    initial_pretrain_epoch = restore_checkpoint(latest_pretrain_checkpoint,
                                                pretrain_checkpoint,
                                                pretrain_optimizer)
    initial_distill_epoch = restore_checkpoint(latest_distill_checkpoint,
                                               distill_checkpoint,
                                               distill_optimizer)

  if not FLAGS.disable_reliability_estimation:
    mean_reliability = train_builder.mean_reliability  # type: ignore
    logging.info('Training on a slice of CIFAR-PI with mean reliability: %1.3f',
                 mean_reliability)
    with summary_writer.as_default():
      tf.summary.scalar('mean_reliability', mean_reliability, step=0)

    objective = work_unit.get_measurement_series('mean_reliability')
    objective.create_measurement(mean_reliability, 0)

  @tf.function
  def get_training_labels_and_features(inputs):
    """Loads the labels, images and privileged information used for training."""
    images = inputs['features']
    if FLAGS.use_annotator_labels:
      labels = inputs['pi_features'][
          'annotator_labels']  # We use the relabeled values to train.
      labels = pi_utils.flatten_annotator_axis(labels)

      # The pi_features might come padded with dummy annotators.
      # We should not compute the loss with them, so we remove them.
      annotator_ids = tf.reshape(inputs['pi_features']['annotator_ids'], [-1])
      non_empty_indices = tf.where(annotator_ids != -1)
      labels = tf.gather(labels, non_empty_indices)
    else:
      labels = inputs['clean_labels']
      # pytype: disable=attribute-error
      labels = pi_utils.repeat_across_annotators(
          labels,
          num_annotators_per_example=train_builder
          .num_annotators_per_example_and_step)
      # pytype: enable=attribute-error
      labels = pi_utils.flatten_annotator_axis(labels)
      non_empty_indices = None

    privileged_information = privileged_information_fn(inputs)
    if FLAGS.augmix and FLAGS.aug_count >= 1:
      # Index 0 at augmix processing is the unperturbed image.
      # We take just 1 augmented image from the returned augmented images.
      images = images[:, 1, ...]

    return labels, images, privileged_information, non_empty_indices

  @tf.function
  def update_train_metrics(labels,
                           logits,
                           negative_log_likelihood,
                           loss,
                           noise_split,
                           training_phase,
                           noisy_idx=None):
    """Computes the training metrics.

    Args:
      labels: tf.Tensor of target labels.
      logits: tf.Tensor of predicted logits.
      negative_log_likelihood: Scalar of NLL.
      loss: Scalar of training loss.
      noise_split: String defining the noise split for which the training
        metrics are computed. It can be ['clean', 'noisy', ''].
      training_phase: A string denoting whether the training metrics correspond
        to the 'pretrain' or distill' training phase.
      noisy_idx: tf.Tensor with shape (batch_size, ) whose entries, in {0,1}
        denote whether a given example is noisy (incorrectly annotated) or not.
        The `noisy_idx` are used to separate the noisy and the clean examples
        within the batch to log their metrics separately.
    """
    probs = tf.nn.softmax(logits)

    sample_weight = None
    if noise_split == '_clean':
      sample_weight = 1. - noisy_idx
    elif noise_split == '_noisy':
      sample_weight = noisy_idx
    metrics[f'{training_phase}/train/ece{noise_split}'].add_batch(
        probs, label=labels, sample_weight=sample_weight)
    metrics[f'{training_phase}/train/accuracy{noise_split}'].update_state(
        labels, logits, sample_weight=sample_weight)

    metrics[f'{training_phase}/train/negative_log_likelihood'].update_state(
        negative_log_likelihood)
    metrics[f'{training_phase}/train/loss'].update_state(loss)

  @tf.function
  def pretrain_step(iterator):
    """Pre-training StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      labels, images, privileged_information, non_empty_indices = get_training_labels_and_features(
          inputs)
      noisy_idx = pi_utils.find_noisy_annotators(inputs)

      with tf.GradientTape() as tape:
        logits = teacher_model((images, privileged_information), training=True)

        # Flatten the annotator axis.
        logits = pi_utils.flatten_annotator_axis(logits)

        if FLAGS.use_annotator_labels:
          logits = tf.gather(logits, non_empty_indices)

        if FLAGS.label_smoothing == 0.:
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  labels, logits, from_logits=True))

        else:
          one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.categorical_crossentropy(
                  one_hot_labels,
                  logits,
                  from_logits=True,
                  label_smoothing=FLAGS.label_smoothing))

        l2_loss = sum(teacher_model.losses)
        loss = negative_log_likelihood + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, teacher_model.trainable_variables)
      pretrain_optimizer.apply_gradients(
          zip(grads, teacher_model.trainable_variables))

      for noise_split in ['', '_clean', '_noisy']:
        update_train_metrics(
            labels=labels,
            logits=logits,
            negative_log_likelihood=negative_log_likelihood,
            loss=loss,
            noise_split=noise_split,
            training_phase='pretrain',
            noisy_idx=noisy_idx)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def distill_step(iterator):
    """Distillation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      labels, images, privileged_information, non_empty_indices = get_training_labels_and_features(
          inputs)
      noisy_idx = pi_utils.find_noisy_annotators(inputs)

      with tf.GradientTape() as tape:
        teacher_logits = teacher_model((images, privileged_information),
                                       training=False)
        logits = distill_model(images, training=True)

        # teacher_logits: (batch_size, num_annotators, num_classes) ->
        #                 (batch_size * num_annotators, num_classes)
        teacher_logits = pi_utils.flatten_annotator_axis(teacher_logits)
        # logits: (batch_size, num_classes) ->
        #         (batch_size, num_annotators, num_classes)
        # pytype: disable=attribute-error
        logits = pi_utils.repeat_across_annotators(
            logits,
            train_builder.
            num_annotators_per_example_and_step)
        # pytype: enable=attribute-error
        # logits: (batch_size, num_annotators, num_classes) ->
        #         (batch_size * num_annotators, num_classes)
        logits = pi_utils.flatten_annotator_axis(logits)

        if FLAGS.use_annotator_labels:
          teacher_logits = tf.gather(teacher_logits, non_empty_indices)
          logits = tf.gather(logits, non_empty_indices)

        distill_labels = tf.nn.softmax(teacher_logits /
                                       FLAGS.distillation_temperature)

        if FLAGS.label_smoothing == 0.:
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.sparse_categorical_crossentropy(
                  labels, logits, from_logits=True))
        else:
          one_hot_labels = tf.one_hot(tf.cast(labels, tf.int32), num_classes)
          negative_log_likelihood = tf.reduce_mean(
              tf.keras.losses.categorical_crossentropy(
                  one_hot_labels,
                  logits,
                  from_logits=True,
                  label_smoothing=FLAGS.label_smoothing))

        negative_log_likelihood_distill = tf.reduce_mean(
            tf.keras.losses.categorical_crossentropy(
                tf.stop_gradient(distill_labels), logits, from_logits=True))

        l2_loss = sum(distill_model.losses)
        loss = (
            1. - FLAGS.distillation_loss_weight
        ) * negative_log_likelihood + FLAGS.distillation_loss_weight * negative_log_likelihood_distill + l2_loss
        # Scale the loss given the TPUStrategy will reduce sum all gradients.
        scaled_loss = loss / strategy.num_replicas_in_sync

      grads = tape.gradient(scaled_loss, distill_model.trainable_variables)
      distill_optimizer.apply_gradients(
          zip(grads, distill_model.trainable_variables))

      for noise_split in ['', '_clean', '_noisy']:
        update_train_metrics(
            labels=labels,
            logits=logits,
            negative_log_likelihood=negative_log_likelihood,
            loss=loss,
            noise_split=noise_split,
            training_phase='distill',
            noisy_idx=noisy_idx)

    for _ in tf.range(tf.cast(steps_per_epoch, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def validation_step(iterator, num_steps, use_annotator_labels):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      if use_annotator_labels:
        labels = inputs['pi_features'][
            'annotator_labels']  # We use the relabeled values to train.
        labels = pi_utils.flatten_annotator_axis(labels)

        # The pi_features might come padded with dummy annotators.
        # We should not compute the loss with them, so we remove them.
        annotator_ids = tf.reshape(inputs['pi_features']['annotator_ids'], [-1])
        non_empty_indices = tf.where(annotator_ids != -1)
        labels = tf.gather(labels, non_empty_indices)
      else:
        labels = inputs['clean_labels']

      logits = distill_model(images, training=False)
      # pytype: disable=attribute-error
      logits = pi_utils.repeat_across_annotators(
          logits,
          num_annotators_per_example=train_builder.
          num_annotators_per_example_and_step)
      # pytype: enable=attribute-error
      logits = pi_utils.flatten_annotator_axis(logits)
      if use_annotator_labels:
        logits = tf.gather(logits, non_empty_indices)
      probs = tf.nn.softmax(logits)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      validation_type = 'validation_noisy' if use_annotator_labels else 'validation_clean'
      metrics[
          f'distill/{validation_type}/negative_log_likelihood'].update_state(
              negative_log_likelihood)
      metrics[f'distill/{validation_type}/accuracy'].update_state(labels, probs)
      metrics[f'distill/{validation_type}/ece'].add_batch(probs, label=labels)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  @tf.function
  def test_step(iterator, dataset_split, dataset_name, num_steps):
    """Evaluation StepFn."""

    def step_fn(inputs):
      """Per-Replica StepFn."""
      images = inputs['features']
      labels = inputs['labels']

      logits = distill_model(images, training=False)
      probs = tf.nn.softmax(logits)

      negative_log_likelihood = tf.reduce_mean(
          tf.keras.losses.sparse_categorical_crossentropy(labels, probs))

      if dataset_name == 'clean':
        metrics[
            f'distill/{dataset_split}/negative_log_likelihood'].update_state(
                negative_log_likelihood)
        metrics[f'distill/{dataset_split}/accuracy'].update_state(labels, probs)
        metrics[f'distill/{dataset_split}/ece'].add_batch(probs, label=labels)
      elif dataset_name.startswith('ood/'):
        ood_labels = 1 - inputs['is_in_distribution']
        if FLAGS.dempster_shafer_ood:
          ood_scores = ood_utils.DempsterShaferUncertainty(logits)
        else:
          ood_scores = 1 - tf.reduce_max(probs, axis=-1)

        # Edgecase for if dataset_name contains underscores
        for name, metric in metrics.items():
          if dataset_name in name:
            metric.update_state(ood_labels, ood_scores)
      else:
        corrupt_metrics[f'distill/test/nll_{dataset_name}'].update_state(
            negative_log_likelihood)
        corrupt_metrics[f'distill/test/accuracy_{dataset_name}'].update_state(
            labels, probs)
        corrupt_metrics[f'distill/test/ece_{dataset_name}'].add_batch(
            probs, label=labels)

    for _ in tf.range(tf.cast(num_steps, tf.int32)):
      strategy.run(step_fn, args=(next(iterator),))

  metrics.update({'pretrain/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'distill/ms_per_example': tf.keras.metrics.Mean()})
  metrics.update({'test/ms_per_example': tf.keras.metrics.Mean()})

  def log_eta(epoch, start_time, training_phase):
    """Estimates and logs the estimated time to finish the current training phase.
    """
    current_step = (epoch + 1) * steps_per_epoch
    total_epochs = FLAGS.train_epochs if training_phase == 'pretrain' else FLAGS.distillation_epochs
    max_steps = steps_per_epoch * total_epochs
    time_elapsed = time.time() - start_time
    steps_per_sec = float(current_step) / time_elapsed
    eta_seconds = (max_steps - current_step) / steps_per_sec
    message = ('({:s}) {:.1%} completion: epoch {:d}/{:d}. {:.1f} steps/s. '
               'ETA: {:.0f} min. Time elapsed: {:.0f} min'.format(
                   training_phase, current_step / max_steps, epoch + 1,
                   total_epochs, steps_per_sec, eta_seconds / 60,
                   time_elapsed / 60))
    logging.info(message)

  def write_metrics_to_summary(total_results, epoch):
    """Logs the metrics using the `summary_writer`."""

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

    if FLAGS.corruptions_interval > 0:
      for metric in corrupt_metrics.values():
        metric.reset_states()

  train_iterator = iter(train_dataset)
  start_time = time.time()
  tb_callback = None
  if FLAGS.collect_profile:
    tb_callback = tf.keras.callbacks.TensorBoard(
        profile_batch=(100, 102),
        log_dir=os.path.join(FLAGS.output_dir, 'logs'))
    tb_callback.set_model(teacher_model)
  logging.info('Starting pretraining phase.')
  for epoch in range(initial_pretrain_epoch, FLAGS.train_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    if not FLAGS.eval_only:
      train_start_time = time.time()
      pretrain_step(train_iterator)
      ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
      metrics['pretrain/ms_per_example'].update_state(ms_per_example)
      log_eta(epoch, start_time, training_phase='pretrain')
    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    logging.info('Train Loss: %.4f, Accuracy: %.2f%%',
                 metrics['pretrain/train/loss'].result(),
                 metrics['pretrain/train/accuracy'].result() * 100)
    total_results = {
        name: metric.result()
        for name, metric in metrics.items()
        if 'pretrain' in name
    }
    write_metrics_to_summary(total_results, epoch)

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = pretrain_checkpoint.save(
          os.path.join(pretrain_checkpoint_dir, 'checkpoint'))
      logging.info('Saved pretrain checkpoint to %s', checkpoint_name)

  logging.info('Starting distillation phase.')
  start_time = time.time()
  if FLAGS.collect_profile:
    tb_callback.set_model(distill_model)  # pytype: disable=attribute-error
  for epoch in range(initial_distill_epoch, FLAGS.distillation_epochs):
    logging.info('Starting to run epoch: %s', epoch)
    if tb_callback:
      tb_callback.on_epoch_begin(epoch)
    if not FLAGS.eval_only:
      train_start_time = time.time()
      distill_step(train_iterator)
      ms_per_example = (time.time() - train_start_time) * 1e6 / batch_size
      metrics['distill/ms_per_example'].update_state(ms_per_example)

      log_eta(epoch, start_time, training_phase='distill')

    if tb_callback:
      tb_callback.on_epoch_end(epoch)

    if validation_dataset:
      validation_iterator = iter(validation_dataset)
      validation_step(validation_iterator, steps_per_validation, True)
      validation_iterator = iter(validation_dataset)
      validation_step(validation_iterator, steps_per_validation, False)
    datasets_to_evaluate = {'clean': test_datasets['clean']}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      datasets_to_evaluate = test_datasets
    for dataset_name, test_dataset in datasets_to_evaluate.items():
      test_iterator = iter(test_dataset)
      logging.info('Testing on dataset %s', dataset_name)
      logging.info('Starting to run eval at epoch: %s', epoch)
      test_start_time = time.time()
      test_step(test_iterator, 'test', dataset_name, steps_per_eval)
      ms_per_example = (time.time() - test_start_time) * 1e6 / batch_size
      metrics['test/ms_per_example'].update_state(ms_per_example)

      logging.info('Done with testing on %s', dataset_name)

    if FLAGS.eval_on_ood:
      for ood_dataset_name, ood_dataset in ood_datasets.items():
        ood_iterator = iter(ood_dataset)
        logging.info('Calculating OOD on dataset %s', ood_dataset_name)
        logging.info('Running OOD eval at epoch: %s', epoch)
        test_step(ood_iterator, 'test', ood_dataset_name,
                  steps_per_ood[ood_dataset_name])

        logging.info('Done with OOD eval on %s', ood_dataset_name)

    corrupt_results = {}
    if (FLAGS.corruptions_interval > 0 and
        (epoch + 1) % FLAGS.corruptions_interval == 0):
      corrupt_results = utils.aggregate_corrupt_metrics(corrupt_metrics,
                                                        corruption_types)

    logging.info('Distillation Loss: %.4f, Accuracy: %.2f%%',
                 metrics['distill/train/loss'].result(),
                 metrics['distill/train/accuracy'].result() * 100)
    logging.info('Test NLL: %.4f, Accuracy: %.2f%%',
                 metrics['distill/test/negative_log_likelihood'].result(),
                 metrics['distill/test/accuracy'].result() * 100)

    total_results = {
        name: metric.result()
        for name, metric in metrics.items()
        if 'distill' in name
    }
    total_results.update(corrupt_results)
    write_metrics_to_summary(total_results, epoch)

    if (FLAGS.checkpoint_interval > 0 and
        (epoch + 1) % FLAGS.checkpoint_interval == 0):
      checkpoint_name = distill_checkpoint.save(
          os.path.join(distill_checkpoint_dir, 'checkpoint'))
      logging.info('Saved distillation checkpoint to %s', checkpoint_name)

  final_checkpoint_name = distill_checkpoint.save(
      os.path.join(FLAGS.output_dir, 'checkpoint'))
  logging.info('Saved last checkpoint to %s', final_checkpoint_name)
  with summary_writer.as_default():
    hp.hparams({
        'base_learning_rate': FLAGS.base_learning_rate,
        'one_minus_momentum': FLAGS.one_minus_momentum,
        'l2': FLAGS.l2,
    })


if __name__ == '__main__':
  app.run(main)
