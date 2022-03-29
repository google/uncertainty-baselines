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

"""Tests for colab_utils."""

from absl.testing import absltest
from absl.testing import parameterized

import pandas as pd
import colab_utils  # local file import from experimental.big_paper


def get_test_dataframe():
  # Experiment where we sweep `config.lr` over [.01, .02].
  # - config.lr = .01 has better validation loss
  # - config.lr = .02 has better AUROC.
  rows = [
      {
          'config.seed': 1,
          'config.lr': .01,
          'config.steps': 500,
          'val_loss': .1,
          'auroc': .8,
      },
      {
          'config.seed': 2,
          'config.lr': .01,
          'config.steps': 500,
          'val_loss': .08,
          'auroc': .6
      },
      {
          'config.seed': 1,
          'config.lr': .02,
          'config.steps': 500,
          'val_loss': .5,
          'auroc': .9
      },
      {
          'config.seed': 2,
          'config.lr': .02,
          'config.steps': 500,
          'val_loss': .8,
          'auroc': .95
      },
  ]
  df = pd.DataFrame(rows)
  df['model'] = 'Det'
  df['config.dataset'] = 'Cifar10'
  return df


class ColabUtilsTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name='_numerical_column',
          df=pd.DataFrame({
              'a': [2, 2, 2],
              'b': ['x', 'y', 'z']
          }),
          column='a',
          expected_value=2),
      dict(
          testcase_name='_single_row',
          df=pd.DataFrame({
              'a': [2,],
              'b': ['x']
          }),
          column='a',
          expected_value=2),
      dict(
          testcase_name='_text_column',
          df=pd.DataFrame({
              'a': [2, 3, 4],
              'b': ['z', 'z', 'z']
          }),
          column='b',
          expected_value='z'),
  )
  def test_get_unique_value(self, df, column, expected_value):
    self.assertEqual(colab_utils.get_unique_value(df, column), expected_value)

  @parameterized.named_parameters(
      dict(
          testcase_name='_empty_dataframe',
          df=pd.DataFrame({
              'a': [],
              'b': []
          }),
          column='a'),
      dict(
          testcase_name='_non_unique_values',
          df=pd.DataFrame({
              'a': [2, 2, 2],
              'b': ['x', 'y', 'z']
          }),
          column='b'),
  )
  def test_get_unique_value_fails(self, df, column):
    with self.assertRaisesRegex(ValueError,
                                'Expected unique value in column'):
      colab_utils.get_unique_value(df, column)

  @parameterized.parameters(('config.seed', (), True),
                            ('learning_rate', (), False),
                            ('learning_rate', ('learning_rate', 'model'), True),
                            ('random_seed', (), False))
  def test_is_hyperparameter(self, column, auxiliary_hparams, expected_result):
    self.assertEqual(
        colab_utils.is_hyperparameter(column, auxiliary_hparams),
        expected_result)

  @parameterized.parameters(
      ((), ['config.seed', 'config.lr']),
      (('config.seed'), ['config.lr']),
      (('config.seed', 'config.lr'), []),
  )
  def test_get_sweeped_hyperparameters(self, marginalization_hparams,
                                       expected_sweeped_params):
    df = get_test_dataframe()
    actual_sweeped_hparams = colab_utils.get_sweeped_hyperparameters(
        df, marginalization_hparams)
    self.assertSetEqual(
        set(actual_sweeped_hparams), set(expected_sweeped_params))

  @parameterized.named_parameters(
      dict(
          testcase_name='_tune_lr_on_loss',
          metric='val_loss',
          marginalization_hparams=('config.seed',),
          expected_hparams={'config.lr': .01},
      ),
      dict(
          testcase_name='_tune_lr_on_auroc',
          metric='auroc',
          marginalization_hparams=('config.seed',),
          expected_hparams={'config.lr': .02},
      ),
      dict(
          testcase_name='_tune_lr_and_seed_on_loss',
          metric='val_loss',
          marginalization_hparams=(),
          expected_hparams={
              'config.lr': .01,
              'config.seed': 2
          },
      ),
      dict(
          testcase_name='_no_tuning',
          metric='val_loss',
          marginalization_hparams=('config.lr', 'config.seed'),
          expected_hparams={},
      ),
  )
  def test_get_best_hyperparameters(self, metric, marginalization_hparams,
                                    expected_hparams):
    df = get_test_dataframe()
    actual_hparams = colab_utils.get_best_hyperparameters(
        df, metric, marginalization_hparams)
    self.assertDictEqual(actual_hparams, expected_hparams)

  @parameterized.parameters(('val_loss', .01), ('auroc', .02))
  def test_get_tuned_results(self, tuning_metric, best_lr):
    df = get_test_dataframe()
    actual_results = colab_utils.get_tuned_results(
        df, tuning_metric=tuning_metric)
    expected_results = actual_results[actual_results['config.lr'] == best_lr]
    pd.testing.assert_frame_equal(actual_results, expected_results)


if __name__ == '__main__':
  absltest.main()
