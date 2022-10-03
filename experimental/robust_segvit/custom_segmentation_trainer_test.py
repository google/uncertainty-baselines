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

"""Tests for the segmentation train script."""

import functools
import shutil
import tempfile

from absl.testing import absltest
from absl.testing import parameterized
import flax
from flax import jax_utils
import flax.linen as nn
import jax.numpy as jnp
import jax.random
import ml_collections
import numpy as np
from scenic.model_lib.base_models import segmentation_model
from scenic.train_lib_deprecated import optimizers
from scenic.train_lib_deprecated import train_utils
from sklearn import metrics as sk_metrics
import tensorflow as tf
import custom_segmentation_trainer  # local file import from experimental.robust_segvit
import custom_models

class SegmentationTrainerTest(parameterized.TestCase):
  """Tests the default trainer on single device setup."""

  def setUp(self):
    super(SegmentationTrainerTest, self).setUp()
    self.test_dir = tempfile.mkdtemp()
    # make sure tf does not allocate gpu memory
    tf.config.experimental.set_visible_devices([], 'GPU')

  def tearDown(self):
    shutil.rmtree(self.test_dir)
    super(SegmentationTrainerTest, self).tearDown()

  def get_train_state(self, rng, fake_batch_logits):
    """Generates the initial training state."""
    config = ml_collections.ConfigDict({
        'lr_configs': {
            'base_learning_rate': 0.1,
        },
        'optimizer': 'sgd',
    })

    # define a fake model that always outputs the same logits: fake_batch_logits
    class FakeFlaxModel(nn.Module):
      """A fake flax model."""

      @nn.compact
      def __call__(self, x, train=False, debug=False):
        del x
        del train
        del debug
        # FakeFlaxModule always predicts class 2.
        # The custom_segmentation_trainer assumes model gives two outputs
        return fake_batch_logits, None

    dummy_input = jnp.zeros((10, 10), jnp.float32)
    initial_params = FakeFlaxModel().init(rng, dummy_input).get(
        'params', flax.core.frozen_dict.FrozenDict({}))
    init_model_state = flax.core.frozen_dict.FrozenDict({})
    optimizer = optimizers.get_optimizer(config).create(initial_params)
    init_train_state = jax_utils.replicate(
        train_utils.TrainState(
            global_step=0,
            optimizer=optimizer,
            model_state=init_model_state,
            rng=jax.random.PRNGKey(0)))
    return FakeFlaxModel(), init_train_state

  def train_and_evaluation(self, model, train_state, fake_batches, metrics_fn):
    """Given the train_state, trains the model on fake batches."""
    eval_metrics = []
    fake_batches_replicated = jax_utils.replicate(fake_batches)
    # Fake config
    config = ml_collections.ConfigDict()
    config.eval_configs = ml_collections.ConfigDict()
    config.eval_configs.mode = 'standard'
    config.model = ml_collections.ConfigDict()
    config.model.backbone = ml_collections.ConfigDict()

    eval_step_pmapped = jax.pmap(
        functools.partial(
            custom_segmentation_trainer.eval_step,
            flax_model=model,
            metrics_fn=metrics_fn,
            config=config,
            debug=False),
        axis_name='batch',
        # We can donate both buffers of train_state and train_batch.
        donate_argnums=(0, 1),
    )
    for fake_batch in fake_batches_replicated:
      _, _, metrics, _, _ = eval_step_pmapped(
          train_state=train_state, batch=fake_batch)
      metrics = train_utils.unreplicate_and_get(metrics)
      eval_metrics.append(metrics)
    eval_metrics = train_utils.stack_forest(eval_metrics)
    eval_summary = jax.tree_map(lambda x: x.sum(), eval_metrics)
    for key, val in eval_summary.items():
      eval_summary[key] = val[0] / val[1]
    return eval_summary

  def test_segmentation_model_evaluate(self):
    """Test trainer evaluate end to end with segmentation model metrics."""
    # image and always outputs the same logits for all pixels
    height, width = 2, 2
    # define a fix output for the fake flax model
    fake_batch_logits = np.tile([.5, .2, .7, 0.0], (4, height, width, 1))

    # 4 evaluation batches of size 4.
    fake_batches = [
        {
            'inputs':
                None,
            'label':
                np.tile([[[3]], [[2]], [[1]], [[0]]], (1, height, width)),
            'batch_mask':
                np.tile([[[1]], [[1]], [[1]], [[1]]], (1, height, width)),
        },
        {
            'inputs':
                None,
            'label':
                np.tile([[[0]], [[3]], [[2]], [[0]]], (1, height, width)),
            'batch_mask':
                np.tile([[[1]], [[1]], [[1]], [[1]]], (1, height, width)),
        },
        {
            'inputs':
                None,
            'label':
                np.tile([[[0]], [[0]], [[0]], [[0]]], (1, height, width)),
            'batch_mask':
                np.tile([[[1]], [[1]], [[1]], [[1]]], (1, height, width)),
        },
        {
            'inputs':
                None,
            'label':
                np.tile([[[1]], [[1]], [[1]], [[1]]], (1, height, width)),
            'batch_mask':
                np.tile([[[1]], [[1]], [[1]], [[1]]], (1, height, width)),
        },
    ]

    rng = jax.random.PRNGKey(0)
    model, train_state = self.get_train_state(rng, fake_batch_logits)
    eval_summary = self.train_and_evaluation(
        model, train_state, fake_batches,
        functools.partial(
            segmentation_model.semantic_segmentation_metrics_function,
            target_is_onehot=False))

    def batch_loss(logits, targets):
      # softmax cross-entropy loss
      one_hot_targets = np.eye(4)[targets]
      loss = -np.sum(one_hot_targets * nn.log_softmax(logits), axis=-1)
      return loss

    expected_accuracy = 8.0 / 64.0  # FakeFlaxModule always predicts class 2.
    expected_loss = np.mean(
        [batch_loss(fake_batch_logits, b['label']) for b in fake_batches])

    self.assertEqual(expected_accuracy, eval_summary['accuracy'])
    np.testing.assert_allclose(expected_loss, eval_summary['loss'], atol=1e-6)

  @parameterized.parameters([(0, 0.0), (1, 0.01), (2, 0.5), (3, 0.99), (4, 1)])
  def test_get_confusion_matrix(self, seed, masked_fraction):
    """Test computation of mIoU metric."""
    np.random.seed(seed)

    # Create test data:
    num_classes = 3
    input_shape = [8, 1, 224, 224]
    logits_shape = input_shape + [num_classes]
    logits_np = np.random.rand(*logits_shape)
    logits = jnp.array(logits_np)

    # Note: We include label -1, which indicates excluded pixels:
    label = np.random.randint(0, num_classes, size=input_shape)
    label[:4] = np.argmax(logits_np[:4], axis=-1)  # Set half to correct.

    batch_np = {
        'label':
            label,
        'batch_mask':
            (np.random.rand(*input_shape) > masked_fraction) & (label != -1),
    }
    batch = {
        'label': jnp.array(batch_np['label']),
        'batch_mask': jnp.array(batch_np['batch_mask']),
    }

    cm_pmapped = jax.pmap(
        custom_segmentation_trainer.get_confusion_matrix, axis_name='batch')
    confusion_matrix = [
        cm_pmapped(labels=labels, logits=logits_, batch_mask=masks)
        for labels, logits_, masks in
        zip(batch['label'], logits, batch['batch_mask'])]
    confusion_matrix = jax.device_get(jax_utils.unreplicate(confusion_matrix))
    metrics_dict = segmentation_model.global_metrics_fn(
        confusion_matrix, dataset_metadata={'class_names': ['x'] * num_classes})
    labels_negative_ignored = np.maximum(batch_np['label'], 0)
    miou_np = sk_metrics.jaccard_score(
        y_true=labels_negative_ignored.ravel(),
        y_pred=np.argmax(logits_np, axis=-1).ravel(),
        average='macro',
        sample_weight=batch_np['batch_mask'].ravel())

    self.assertAlmostEqual(metrics_dict['mean_iou'], miou_np, places=4)

  @parameterized.parameters([(0, 0.0), (1, 0.01), (2, 0.5), (3, 0.99), (4, 1)])
  def test_unc_confusion_matrix(self, seed, masked_fraction):
    """Test computation of mIoU metric."""
    np.random.seed(seed)

    # Create test data:
    num_classes = 3
    input_shape = [8, 1, 224, 224]
    logits_shape = input_shape + [num_classes]
    logits_np = np.random.rand(*logits_shape)
    logits = jnp.array(logits_np)

    # when the uncertainty threshold is 100% or = 0
    # all labels are certain, and pavpu is the fraction of patches that are accurate.
    uncertainty_th = 0.0
    window_size = 1

    # Note: We include label -1, which indicates excluded pixels:
    label = np.random.randint(0, num_classes, size=input_shape)
    label[:4] = np.argmax(logits_np[:4], axis=-1)  # Set half to correct.

    batch_np = {
        'label':
            label,
        'batch_mask':
            (np.random.rand(*input_shape) > masked_fraction) & (label != -1),
    }
    batch = {
        'label': jnp.array(batch_np['label']),
        'batch_mask': jnp.array(batch_np['batch_mask']),
    }

    cm_pmapped = jax.pmap(
      functools.partial(
        custom_segmentation_trainer.get_uncertainty_confusion_matrix,
        uncertainty_th=uncertainty_th,
        window_size=window_size,
        uncertainty_measure='softmax',
      ), axis_name='batch')

    unc_confusion_matrix = [
        cm_pmapped(labels=labels, logits=logits_, weights=masks)
        for labels, logits_, masks in
        zip(batch['label'], logits, batch['batch_mask'])]
    unc_confusion_matrix = jax.device_get(jax_utils.unreplicate(unc_confusion_matrix))
    metrics_dict = custom_models.global_unc_metrics_fn(
        unc_confusion_matrix)
    labels_negative_ignored = np.maximum(batch_np['label'], 0)
    y_pred = np.argmax(logits_np, axis=-1)
    weights = batch_np['batch_mask']
    accurate = labels_negative_ignored == y_pred
    pavpu = np.sum(accurate * weights) / np.sum(weights)
    # all labels are certain, pavpu = fraction of patches that are accurate
    self.assertAlmostEqual(metrics_dict['pacc_cert'], jnp.nan_to_num(pavpu), places=2)


if __name__ == '__main__':
  absltest.main()
