# coding=utf-8
# Copyright 2021 The Uncertainty Baselines Authors.
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

"""OOD utilities for CIFAR-10 and CIFAR-100."""

import tensorflow as tf
import uncertainty_baselines as ub


def DempsterShaferUncertainty(logits):
  """Defines the Dempster-Shafer Uncertainty for output logits.

  Under the Dempster-Shafer (DS) formulation of a multi-class model, the
  predictive uncertainty can be assessed as K/(K + sum(exp(logits))).
  This uncertainty metric directly measure the magnitude of the model logits,
  and is more properiate for a model that directly trains the magnitude of
  logits and uses this magnitude to quantify uncertainty (e.g., [1]).

  See Equation (1) of [1] for full detail.

  Args:
    logits: (tf.Tensor) logits of model prediction, shape (batch_size,
      num_classes).

  Returns:
    (tf.Tensor) DS uncertainty estimate, shape (batch_size, )
  """
  num_classes = tf.shape(logits)[-1]
  num_classes = tf.cast(num_classes, dtype=logits.dtype)

  belief_mass = tf.reduce_sum(tf.exp(logits), axis=-1)
  return num_classes / (belief_mass + num_classes)


def create_ood_metrics(ood_dataset_names, tpr_list=None):
  """Create OOD metrics."""
  ood_metrics = {}
  for dataset_name in ood_dataset_names:
    ood_dataset_name = f'ood/{dataset_name}'
    ood_metrics.update({
        f'{ood_dataset_name}_auroc':
            tf.keras.metrics.AUC(curve='ROC', num_thresholds=100000),
        f'{ood_dataset_name}_auprc':
            tf.keras.metrics.AUC(curve='PR', num_thresholds=100000),
    })
    if tpr_list:
      for tpr in tpr_list:
        tpr_percent = int(tpr * 100)
        ood_metrics.update({
            f'{ood_dataset_name}_(1-fpr)@{tpr_percent}tpr':
                tf.keras.metrics.SpecificityAtSensitivity(
                    tpr, num_thresholds=100000)
        })
  return ood_metrics


def load_ood_datasets(ood_dataset_names,
                      in_dataset_builder,
                      in_dataset_validation_percent,
                      batch_size,
                      drop_remainder=False):
  """Load OOD datasets."""
  steps = {}
  datasets = {}
  for ood_dataset_name in ood_dataset_names:
    ood_dataset_class = ub.datasets.DATASETS[ood_dataset_name]
    ood_dataset_class = ub.datasets.make_ood_dataset(ood_dataset_class)
    # If the OOD datasets are not CIFAR10/CIFAR100, we normalize by CIFAR
    # statistics, since all test datasets should be preprocessed the same.
    if 'cifar' not in ood_dataset_name:
      ood_dataset_builder = ood_dataset_class(
          in_dataset_builder,
          split='test',
          validation_percent=in_dataset_validation_percent,
          normalize_by_cifar=True,
          drop_remainder=drop_remainder)
    else:
      ood_dataset_builder = ood_dataset_class(
          in_dataset_builder,
          split='test',
          validation_percent=in_dataset_validation_percent,
          drop_remainder=drop_remainder)
    ood_dataset = ood_dataset_builder.load(batch_size=batch_size)
    steps[f'ood/{ood_dataset_name}'] = ood_dataset_builder.num_examples(
        'in_distribution') // batch_size + ood_dataset_builder.num_examples(
            'ood') // batch_size
    datasets[f'ood/{ood_dataset_name}'] = ood_dataset

  return datasets, steps
