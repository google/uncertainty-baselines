from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
import numpy as np
from flax import jax_utils

from metrics_multihost import ComputeAUCMetric
from metrics_multihost import ComputeOODAUCMetric
from metrics_multihost import ComputeScoreAUCMetric
import sklearn.metrics

from ood_metrics import get_ood_score

class OODMetricsMultiHostTest(parameterized.TestCase):

  def setUp(self):
    super(OODMetricsMultiHostTest, self).setUp()

  @parameterized.parameters([(0, 0.0), (1, 0.01), (2, 0.5), (3, 0.99), (4, 1)])
  def test_ComputeAUCMetric(self, seed, masked_fraction):
    """Test computation of AUC metric."""
    np.random.seed(seed)

    from_logits = False  # when set to True applies sigmoid to logits.
    num_thresholds = 100

    # Create test data:
    num_classes = 2
    input_shape = [8, 1, 224, 224]
    logits_shape = input_shape + [num_classes]
    logits_np = np.random.rand(*logits_shape)

    # Note: We include label -1, which indicates excluded pixels:
    label = np.random.randint(0, num_classes, size=input_shape)
    label[:4] = np.argmax(logits_np[:4], axis=-1)  # Set half to correct.

    batch_np = {
        'logits': logits_np,
        'label':
            label,
        'batch_mask':
            (np.random.rand(*input_shape) > masked_fraction) & (label != -1),
    }
    batch = {
        'logits': jnp.array(logits_np),
        'label': jnp.array(batch_np['label']),
        'batch_mask': jnp.array(batch_np['batch_mask']),
    }

    fake_batches_replicated = jax_utils.replicate([batch])

    auc_roc = ComputeAUCMetric(curve='ROC', num_thresholds=num_thresholds, from_logits=from_logits)

    for fake_batch in fake_batches_replicated:
      if from_logits:
        pred = jnp.max(fake_batch['logits'], axis=-1)
      else:
        pred = jnp.argmax(fake_batch['logits'], axis=-1)
      auc_roc.calculate_and_update_scores(logits=pred,
                               label=fake_batch['label'],
                               sample_weight=fake_batch['batch_mask'],
                               )

    auc_result = auc_roc.gather_metrics().numpy()

    # Numpy result:
    if np.all(batch_np['batch_mask'] == 0):
      auc_numpy = 0
    else:
      labels_negative_ignored = np.maximum(batch_np['label'], 0)
      y_pred = np.argmax(logits_np, axis=-1)
      auc_numpy = sklearn.metrics.roc_auc_score(labels_negative_ignored.ravel(),
                                              y_pred.ravel(),
                                              sample_weight=batch_np['batch_mask'].ravel())

    self.assertAlmostEqual(auc_result, auc_numpy, places=2)


  @parameterized.parameters([(0, 0.0), (1, 0.01), (2, 0.5), (3, 0.99), (4, 1)])
  def test_ComputeOODAUCMetric(self, seed, masked_fraction):
    """Test computation of OOD scored AUC metric."""
    np.random.seed(seed)
    num_thresholds = 1000

    ood_kwargs = {}
    # Create test data:
    num_classes = 2
    input_shape = [8, 1, 224, 224]
    logits_shape = input_shape + [num_classes]
    logits_np = np.random.rand(*logits_shape)

    # Note: We include label -1, which indicates excluded pixels:
    label = np.random.randint(0, num_classes, size=input_shape)
    label[:4] = np.argmax(logits_np[:4], axis=-1)  # Set half to correct.

    batch_np = {
        'logits': logits_np,
        'label':
            label,
        'batch_mask':
            (np.random.rand(*input_shape) > masked_fraction) & (label != -1),
    }
    batch = {
        'logits': jnp.array(logits_np),
        'label': jnp.array(batch_np['label']),
        'batch_mask': jnp.array(batch_np['batch_mask']),
    }

    fake_batches_replicated = jax_utils.replicate([batch])

    auc_roc = ComputeOODAUCMetric(curve='ROC', num_thresholds=num_thresholds)

    for fake_batch in fake_batches_replicated:
      pred = fake_batch['logits']
      ood_label = 1 - fake_batch['label']

      auc_roc.calculate_and_update_scores(logits=pred,
                               label=ood_label,
                               sample_weight=fake_batch['batch_mask'],
                               *ood_kwargs,
                               )
    auc_result = auc_roc.gather_metrics().numpy()

    # Numpy result:
    if np.all(batch_np['batch_mask'] == 0):
      auc_numpy = 0
    else:
      labels_negative_ignored = np.maximum(batch_np['label'], 0)
      ood_label_np = 1 - labels_negative_ignored
      ood_score =  get_ood_score(logits_np, **ood_kwargs)
      auc_numpy = sklearn.metrics.roc_auc_score(ood_label_np.ravel(),
                                              ood_score.ravel(),
                                              sample_weight=batch_np['batch_mask'].ravel())

    self.assertAlmostEqual(auc_result, auc_numpy, places=1)

  @parameterized.parameters([(0, 0.0), (1, 0.01), (2, 0.5), (3, 0.99), (4, 1)])
  def test_ComputeScoreAUCMetric(self, seed, masked_fraction):
    """Test computation of scored AUC metric."""
    np.random.seed(seed)
    num_thresholds = 10000
    summation_method = 'interpolation'
    ood_kwargs = {'method_name': 'msp'}
    # Create test data:
    num_classes = 2
    input_shape = [8, 1, 224, 224]
    logits_shape = input_shape + [num_classes]
    logits_np = np.random.rand(*logits_shape)
    # Note: We include label -1, which indicates excluded pixels:
    label = np.random.randint(0, num_classes, size=input_shape)
    label[:4] = np.argmax(logits_np[:4], axis=-1)  # Set half to correct.

    batch_np = {
        'logits': logits_np,
        'label':
            label,
        'batch_mask':
            (np.random.rand(*input_shape) > masked_fraction) & (label != -1),
    }
    batch = {
        'logits': jnp.array(logits_np),
        'label': jnp.array(batch_np['label']),
        'batch_mask': jnp.array(batch_np['batch_mask']),
    }


    fake_batches_replicated = jax_utils.replicate([batch])

    auc_roc = ComputeScoreAUCMetric(curve='ROC', num_thresholds=num_thresholds,
                                    summation_method=summation_method)

    for fake_batch in fake_batches_replicated:
      pred = fake_batch['logits']
      ood_label = 1 - fake_batch['label']

      auc_roc.calculate_and_update_scores(logits=pred,
                               label=ood_label,
                               sample_weight=fake_batch['batch_mask'],
                               **ood_kwargs,
                               )
    auc_result = auc_roc.gather_metrics().numpy()

    # Numpy result:
    if np.all(batch_np['batch_mask'] == 0):
      auc_numpy = 0
    else:
      labels_negative_ignored = np.maximum(batch_np['label'], 0)
      ood_label_np = 1 - labels_negative_ignored
      ood_score = get_ood_score(logits_np, **ood_kwargs)
      auc_numpy = sklearn.metrics.roc_auc_score(ood_label_np.ravel(),
                                              ood_score.ravel(),
                                              sample_weight=batch_np['batch_mask'].ravel())

    self.assertAlmostEqual(auc_result, auc_numpy, places=1)

if __name__ == '__main__':
  absltest.main()
