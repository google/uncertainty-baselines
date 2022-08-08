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

r"""Model definition and binary for running Introspective Active Sampling.

Usage:

To train MLP on Cardiotoxicity Fingerprint dataset locally:
# pylint: disable=line-too-long
ml_python3 third_party/py/uncertainty_baselines/experimental/shoshin/introspective_active_learning.py \
  --adhoc_import_modules=uncertainty_baselines \
    -- \
    --xm_runlocal \
    --alsologtostderr \
    --dataset_name=cardiotoxicity \
    --model_name=mlp \
    --num_epochs=10 \
    --train_main_only=True
# pylint: enable=line-too-long
"""

from typing import Dict

from absl import app
from absl import flags
import tensorflow as tf

import data  # local file import from experimental.shoshin
import models  # local file import from experimental.shoshin


FLAGS = flags.FLAGS

flags.DEFINE_string('dataset_name', '', 'Name of registered TF dataset to use.')
flags.DEFINE_string('model_name', '', 'Name of registered model to use.')
# TODO(jihyeonlee): Use num_classes flag across files.
flags.DEFINE_integer('num_classes', 2, 'Number of classes for main task.')
flags.DEFINE_integer('batch_size', 64, 'Batch size.')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs.')
# Model parameter flags.
flags.DEFINE_integer(
    'num_splits', 5, 'Number of shards into which train and '
    'val will be split to train models used in bias label '
    'generation. Use a number that can divide 100 easily since we use '
    'TFDS functionality to split the dataset by percentage.')
flags.register_validator('num_splits',
                         lambda value: 100 % value == 0,
                         message='100 must be divisible by --num_splits.')
flags.DEFINE_list('hidden_sizes', '1024,512,128',
                  'Number and sizes of hidden layers for MLP model.')
flags.DEFINE_boolean('train_main_only', False,
                     'If True, trains only main task head, not bias head.')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('dropout_rate', 0.2, 'Dropout rate.')


class IntrospectiveActiveSampling(tf.keras.Model):
  """Defines Introspective Active Sampling method."""

  def __init__(self, model: tf.keras.Model, train_bias: bool, name: str):
    super(IntrospectiveActiveSampling, self).__init__(name=name)
    self.train_bias = train_bias
    if self.train_bias:
      self.id_to_bias_table = None

    self.model = model

  def update_id_to_bias_table(self, table):
    self.id_to_bias_table = table

  def train_step(self, inputs):
    features, labels, example_ids = inputs
    y_true_main = tf.one_hot(labels, depth=2)

    with tf.GradientTape() as tape:
      y_pred = self.model(features, training=True)

      y_true_bias = None
      if self.train_bias:
        if self.id_to_bias_table is None:
          raise ValueError('id_to_bias_table must not be None.')
        y_true_bias = self.id_to_bias_table.lookup(example_ids)
        y_true_bias = tf.one_hot(y_true_bias, depth=2)
      y_true = {
          'main': y_true_main,
          'bias': y_true_bias
      }
      total_loss = self.compiled_loss(y_true, y_pred)
      total_loss += sum(self.losses)  # Regularization loss

    gradients = tape.gradient(total_loss, self.model.trainable_variables)
    self.optimizer.apply_gradients(
        zip(gradients, self.model.trainable_variables))

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    return results

  def test_step(self, inputs):
    features, labels, example_ids = inputs
    y_true_main = tf.one_hot(labels, depth=2)
    y_pred = self.model(features, training=False)

    y_true_bias = None
    if self.train_bias:
      if self.id_to_bias_table is None:
        raise ValueError('id_to_bias_table must not be None.')
      y_true_bias = self.id_to_bias_table.lookup(example_ids)
      y_true_bias = tf.one_hot(y_true_bias, depth=2)

    y_true = {
        'main': y_true_main,
        'bias': y_true_bias
    }

    self.compiled_metrics.update_state(y_true, y_pred)
    results = {m.name: m.result() for m in self.metrics}
    return results


def compute_loss_main(y_true_main: tf.Tensor, y_pred_main: tf.Tensor):
  """Defines loss function for main classification task."""
  loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  return loss_func(y_true_main, y_pred_main)


def compute_loss_bias(y_true_bias: tf.Tensor, y_pred_bias: tf.Tensor):
  """Defines loss function for bias classification task."""
  loss_func = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
  return loss_func(y_true_bias, y_pred_bias)


def compile_model(model: tf.keras.Model):
  """Compiles model with optimizer, custom loss functions, and metrics."""
  model.compile(
      optimizer=tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate),
      loss={
          'main': compute_loss_main,
          'bias': compute_loss_bias
      },
      loss_weights={
          'main': 1,
          'bias': 1 if model.train_bias else 0
      },
      metrics={
          'main': [
              tf.keras.metrics.CategoricalAccuracy(name='acc'),
              tf.keras.metrics.AUC(name='auc')
          ],
          'bias': [
              tf.keras.metrics.CategoricalAccuracy(name='acc'),
              tf.keras.metrics.AUC(name='auc')
          ]
      })
  return model


def evaluate_model(model: tf.keras.Model, eval_ds: Dict[str, tf.data.Dataset]):
  """Evaluates model on given validation and/or test datasets.

  Args:
    model: Keras model to be evaluated.
    eval_ds: Dictionary mapping evaluation dataset name to the dataset.
  """
  # Evaluate on validation set.
  model.compile(metrics={
      'main': [tf.keras.metrics.CategoricalAccuracy(name='acc'),
               tf.keras.metrics.AUC(name='auc')],
      'bias': [tf.keras.metrics.CategoricalAccuracy(name='acc'),
               tf.keras.metrics.AUC(name='auc')]
  })
  for ds_name in eval_ds.keys():
    result = model.evaluate(eval_ds[ds_name], return_dict=True)
    print(f"Toxicity Acc: {result['main_acc']}")
    print(f"Toxicity AUC: {result['main_auc']}")
    print(f"Bias Acc: {result['bias_acc']}")
    print(f"Bias AUC: {result['bias_auc']}")


def main(_) -> None:

  if FLAGS.train_main_only:
    # Trains only the main classification task with no bias output head.
    dataset_builder = data.get_dataset(FLAGS.dataset_name)
    _, _, train_ds, eval_ds = dataset_builder(
        FLAGS.num_splits, FLAGS.batch_size)
    model_class = models.get_model(FLAGS.model_name)
    hidden_sizes = [int(size) for size in FLAGS.hidden_sizes]
    base_model = model_class(
        train_bias=False, name='main_only_model', hidden_sizes=hidden_sizes)

    introspective_model = IntrospectiveActiveSampling(model=base_model,
                                                      train_bias=False,
                                                      name='main_only')

    introspective_model = compile_model(introspective_model)

    introspective_model.fit(train_ds,
                            validation_data=eval_ds['val'],
                            epochs=FLAGS.num_epochs)

    evaluate_model(introspective_model, eval_ds)

  # TODO(jihyeonlee): In 'else' case, provide implementation of Introspective
  # Active Sampling method. Plan is to check in local version first, then XM.

  # TODO(jihyeonlee): Will save model checkpoints in a FLAGS.output_dir.

  # TODO(jihyeonlee): Will add Waterbirds dataloader and ResNet model to support
  # vision modality.

  # TODO(jihyeonlee): Create dataclass to be base class for dataloaders so that
  # expected return/properties are more clear.

if __name__ == '__main__':
  app.run(main)
