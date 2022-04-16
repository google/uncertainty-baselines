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

import numpy as np
import pandas as pd
import colab_utils  # local file import from experimental.big_paper


def get_test_untuned_dataframe():
  # Experiment where we sweep `config.lr` over [.01, .02].
  # - config.lr = .01 has better validation loss
  # - config.lr = .02 has better AUROC.
  rows = [
      {
          'config.seed': 1,
          'config.lr': .01,
          'config.steps': 500,
          'val_loss': .1,
          'msp_auroc': .8,
      },
      {
          'config.seed': 2,
          'config.lr': .01,
          'config.steps': 500,
          'val_loss': .08,
          'msp_auroc': .6
      },
      {
          'config.seed': 1,
          'config.lr': .02,
          'config.steps': 500,
          'val_loss': .5,
          'msp_auroc': .9
      },
      {
          'config.seed': 2,
          'config.lr': .02,
          'config.steps': 500,
          'val_loss': .8,
          'msp_auroc': .95
      },
  ]
  df = pd.DataFrame(rows)
  df['model'] = 'Det'
  df['config.dataset'] = 'Cifar10'
  return df


def get_test_tuned_dataframe():
  jft = 'jft/entity:1.0.0'
  inet = 'imagenet21k'
  return pd.DataFrame({
      'model': ['Det', 'Det', 'Det', 'BE', 'BE'],
      'config.dataset': [jft, jft, 'cifar10', inet, 'cifar10'],
      'exaflops': [100, 150, np.nan, 200, np.nan],
      'z/cars_5shot': [.6, .8, np.nan, .9, np.nan],
      'val_loss': [.3, .5, np.nan, .3, np.nan],
      'val_prec@1': [.45, .5, .9, .55, .85],
      'test_loss': [np.nan, np.nan, .2, np.nan, .01],
      # First value should be overwritten (upstream test value).
      'test_prec@1': [.2, np.nan, .85, np.nan, .85],
      'cifar_10h_ece': [np.nan, np.nan, .1, np.nan, .02],
  }).set_index('model')


def get_test_dataframe_for_scoring():
  df = pd.DataFrame({
      'model': ['Det', 'BE', 'GP'],
      ('test_prec@1', 'cifar10'): [.8, .9, .85],  # Prediction
      ('test_prec@1', 'imagenet2012'): [.7, .8, .75],  # Prediction
      ('ood_cifar100_msp_auroc_ece', 'cifar10'): [.7, .2, .1],  # Uncertainty
      ('test_calib_auc', 'imagenet2012'): [.3, .5, .6],  # Uncertainty
      ('1shot_prec@1', 'few-shot pets'): [.9, .8, np.nan],  # Adaptation
      ('5shot_prec@1', 'few-shot pets'): [.8, .9, np.nan],  # Adaptation
  }).set_index('model')
  df.columns = pd.MultiIndex.from_tuples(df.columns)
  return df


class ColabUtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      ('ood_cifar10_msp_auroc', 'auroc'),
      ('in_domain_test/accuracy', 'accuracy'),
      ('test_prec@1', 'prec@1'),
      ('ood_test/negative_log_likelihood', 'likelihood'),
      ('ms_step', 'ms_step'),
  )
  def test_get_base_metric(self, metric_name, expected_result):
    self.assertEqual(colab_utils.get_base_metric(metric_name), expected_result)

  @parameterized.parameters('Det', 'jft/entity:1.0.0', 'cifar10_nll')
  def test_get_base_metric_fails(self, metric_name):
    with self.assertRaisesRegex(ValueError, 'Unrecognized metric'):
      colab_utils.get_base_metric(metric_name)

  @parameterized.parameters(
      ('ood_cifar10_msp_auroc', colab_utils.MetricCategory.UNCERTAINTY),
      ('in_domain_test/accuracy', colab_utils.MetricCategory.PREDICTION),
      ('ood_test/negative_log_likelihood',
       colab_utils.MetricCategory.PREDICTION),
      ('test_prec@1', colab_utils.MetricCategory.PREDICTION),
      ('5shot_prec@1', colab_utils.MetricCategory.ADAPTATION),
  )
  def test_get_metric_score_category(self, metric_name, expected_result):
    self.assertEqual(
        colab_utils.get_metric_category(metric_name), expected_result)

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
    with self.assertRaisesRegex(ValueError, 'Expected unique value in column'):
      colab_utils.get_unique_value(df, column)

  @parameterized.named_parameters(
      dict(
          testcase_name='_no_nans',
          df=pd.DataFrame({
              'a': [2, 2, np.nan, np.nan],
              'b': [np.nan, np.nan, 'x', np.nan],
              'c': [np.nan, np.nan, np.nan, 'y']
          }),
          expected_series=pd.Series([2, 2, 'x', 'y'], name='a'),
      ),
      dict(
          testcase_name='_with_nans',
          df=pd.DataFrame({
              'a': [2, np.nan, 3, np.nan],
              'b': [np.nan, np.nan, np.nan, 'x']
          }),
          expected_series=pd.Series([2, np.nan, 3, 'x'], name='a')),
  )
  def test_row_wise_unique_non_nan(self, df, expected_series):
    pd.testing.assert_series_equal(
        colab_utils.row_wise_unique_non_nan(df), expected_series)

  def test_row_wise_unique_non_nan_fails(self):
    df = pd.DataFrame({'a': [1, 2, np.nan], 'b': [1, np.nan, 3]})
    with self.assertRaisesRegex(ValueError, 'have multiple set values'):
      colab_utils.row_wise_unique_non_nan(df)

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
    df = get_test_untuned_dataframe()
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
          metric='msp_auroc',
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
    df = get_test_untuned_dataframe()
    actual_hparams = colab_utils.get_best_hyperparameters(
        df, metric, marginalization_hparams)
    self.assertDictEqual(actual_hparams, expected_hparams)

  @parameterized.parameters(('val_loss', .01), ('msp_auroc', .02))
  def test_get_tuned_results(self, tuning_metric, best_lr):
    df = get_test_untuned_dataframe()
    actual_results = colab_utils.get_tuned_results(
        df, tuning_metric=tuning_metric)
    expected_results = actual_results[actual_results['config.lr'] == best_lr]
    pd.testing.assert_frame_equal(actual_results, expected_results)

  def test_fill_upstream_test_metrics(self):
    input_df = get_test_tuned_dataframe()
    expected_df = input_df.copy()
    expected_df['test_loss'] = [.3, .5, .2, .3, .01]
    expected_df['test_prec@1'] = [.45, .5, .85, .55, .85]

    pd.testing.assert_frame_equal(
        colab_utils._fill_upstream_test_metrics(input_df), expected_df)

  def test_processed_tuned_results(self):
    relevant_metrics = [
        'test_loss', 'cifar_10h_ece', 'z/cars_5shot', 'exaflops'
    ]
    input_df = get_test_tuned_dataframe()
    expected_be_results = {
        'model': 'BE',
        ('test_loss', 'imagenet21k'): .3,  # Filled to val_loss.
        ('test_loss', 'jft/entity:1.0.0'): np.nan,
        ('test_loss', 'cifar10'): .01,
        ('5shot_prec@1', 'few-shot cars'): .9,
        ('cifar_10h_ece', 'cifar10'): .02,
        ('exaflops', 'compute'): 200,
    }

    expected_det_results = {
        'model': 'Det',
        ('test_loss', 'jft/entity:1.0.0'): (.3 + .5) / 2,  # Filled to val_loss.
        ('test_loss', 'imagenet21k'): np.nan,
        ('test_loss', 'cifar10'): .2,
        ('cifar_10h_ece', 'cifar10'): .1,
        ('5shot_prec@1', 'few-shot cars'): (.6 + .8) / 2,
        ('exaflops', 'compute'): (100 + 150) / 2,
    }

    expected_df = pd.DataFrame([expected_be_results,
                                expected_det_results]).set_index('model')
    expected_df.columns = pd.MultiIndex.from_tuples(
        expected_df.columns, names=['metric', 'dataset'])
    result_df = colab_utils.process_tuned_results(input_df, relevant_metrics)
    pd.testing.assert_frame_equal(
        result_df.sort_index(axis=1), expected_df.sort_index(axis=1))

  def test_normalize_scores(self):
    entropy = np.log(10)
    input_df = pd.DataFrame({
        ('test_loss', 'cifar10'): [.2, .7],
        ('in_domain_test/ece', 'retina_country'): [.1, .3],
        ('exaflops', 'compute'): [100, 200],
    })
    input_df.columns = pd.MultiIndex.from_tuples(input_df.columns)

    expected_df = pd.DataFrame({
        ('test_loss', 'cifar10'): [1 - .2 / entropy, 1 - .7 / entropy],
        ('in_domain_test/ece', 'retina_country'): [.9, .7],
        ('exaflops', 'compute'): [100, 200],
    })
    pd.testing.assert_frame_equal(
        colab_utils._normalize_scores(input_df), expected_df)

  @parameterized.named_parameters(
      dict(
          testcase_name='_compute_1shot',
          drop_compute=True,
          drop_1shot=True,
          drop_incomplete_measurements=False,
          datasets=None,
          expected_columns=[('test_prec@1', 'cifar10')],
          expected_models=['Det', 'BE'],
      ),
      dict(
          testcase_name='_drop_incomplete_and_cifar',
          drop_compute=False,
          drop_1shot=False,
          drop_incomplete_measurements=True,
          datasets=['few-shot pets', 'compute'],
          expected_columns=[('1shot_prec@1', 'few-shot pets'),
                            ('exaflops', 'compute')],
          expected_models=['Det'],
      ),
  )
  def test_drop_unused_measurements(self, drop_compute, drop_1shot,
                                    drop_incomplete_measurements, datasets,
                                    expected_columns, expected_models):
    input_df = pd.DataFrame({
        'model': ['Det', 'BE'],
        ('test_prec@1', 'cifar10'): [.8, .85],
        ('1shot_prec@1', 'few-shot pets'): [.8, np.nan],
        ('exaflops', 'compute'): [100, 200],
    }).set_index('model')
    input_df.columns = pd.MultiIndex.from_tuples(input_df.columns)
    expected_df = input_df.loc[expected_models, expected_columns]
    result_df = colab_utils._drop_unused_measurements(
        input_df,
        drop_compute=drop_compute,
        drop_1shot=drop_1shot,
        drop_incomplete_measurements=drop_incomplete_measurements,
        datasets=datasets)
    pd.testing.assert_frame_equal(result_df, expected_df)

  @parameterized.named_parameters(
      dict(
          testcase_name='_drop_incomplete_measurements',
          drop_1shot=False,
          drop_incomplete_measurements=True,
          baseline_model=None,
          datasets=None,
          expected_df=pd.DataFrame({
              'model': ['Det', 'BE'],
              'score_prediction': [(.8 + .7) / 2, (.9 + .8) / 2],
              '#_best_prediction': [0., 2.],
              'mean_rank_prediction': [2., 1.],
              'score_uncertainty': [(.3 + .3) / 2, (.8 + .5) / 2],
              '#_best_uncertainty': [0., 2.],
              'mean_rank_uncertainty': [2., 1.],
              'score_adaptation': [(.9 + .8) / 2, (.8 + .9) / 2],
              '#_best_adaptation': [1., 1.],
              'mean_rank_adaptation': [1.5, 1.5],
              'score': [(.8 + .7 + .3 + .3 + .9 + .8) / 6,
                        (.9 + .8 + .8 + .5 + .8 + .9) / 6],
          }),
      ),
      dict(
          testcase_name='_keep_missing_measurements',
          drop_1shot=True,
          datasets=['imagenet2012', 'few-shot pets'],
          drop_incomplete_measurements=False,
          baseline_model=None,
          expected_df=pd.DataFrame({
              'model': ['Det', 'BE', 'GP'],
              'score_prediction': [.7, .8, .75],
              '#_best_prediction': [0, 1, 0],
              'mean_rank_prediction': [3, 1, 2],
              'score_uncertainty': [.3, .5, .6],
              '#_best_uncertainty': [0, 0, 1],
              'mean_rank_uncertainty': [3, 2, 1],
              'score_adaptation': [.8, .9, np.nan],
              '#_best_adaptation': [0., 1., np.nan],
              'mean_rank_adaptation': [2., 1., np.nan],
              'score': [(.7 + .3 + .8) / 3, (.8 + .5 + .9) / 3, np.nan]
          }),
      ),
      dict(
          testcase_name='_normalized',  # Only score cols change.
          baseline_model='Det',
          drop_1shot=False,
          drop_incomplete_measurements=True,
          datasets=['cifar10'],
          expected_df=pd.DataFrame({
              'model': ['Det', 'BE', 'GP'],  # No missing measurements on Cifar.
              'score_prediction': [1, .9 / .8, .85 / .8],
              '#_best_prediction': [0., 1., 0.],
              'mean_rank_prediction': [3., 1., 2.],
              'score_uncertainty': [1., .8 / .3, .9 / .3],
              '#_best_uncertainty': [0., 0., 1.],
              'mean_rank_uncertainty': [3., 2., 1.],
              'score': [1., (.9 / .8 + .8 / .3) / 2, (.85 / .8 + .9 / .3) / 2],
          }),
      )
  )
  def test_compute_score(self, drop_1shot, drop_incomplete_measurements,
                         baseline_model, datasets, expected_df):
    input_df = get_test_dataframe_for_scoring()

    result_df = colab_utils.compute_score(
        input_df,
        baseline_model=baseline_model,
        drop_1shot=drop_1shot,
        datasets=datasets,
        drop_incomplete_measurements=drop_incomplete_measurements)

    pd.testing.assert_frame_equal(
        result_df.sort_index(axis=0).sort_index(axis=1),
        expected_df.set_index('model').sort_index(axis=0).sort_index(axis=1),
        check_dtype=False)

  @parameterized.named_parameters(
      dict(
          testcase_name='_keep_all_measurements',
          drop_incomplete_measurements=False,
          expected_df=pd.DataFrame({
              'model': ['Det', 'BE', 'GP'],
              ('test_prec@1', 'cifar10'): [3, 1, 2],
              ('ood_cifar100_msp_auroc_ece', 'cifar10'): [3, 2, 1],
              ('5shot_prec@1', 'few-shot pets'): [2, 1, np.nan],
          })),
      dict(
          testcase_name='_drop_incomplete_measurements',
          drop_incomplete_measurements=True,
          expected_df=pd.DataFrame({
              'model': ['Det', 'BE'],
              ('test_prec@1', 'cifar10'): [2, 1],
              ('ood_cifar100_msp_auroc_ece', 'cifar10'): [2, 1],
              ('5shot_prec@1', 'few-shot pets'): [2, 1],
          })),
  )
  def test_rank_models(self, drop_incomplete_measurements, expected_df):
    input_df = get_test_dataframe_for_scoring()

    expected_df = expected_df.set_index('model').astype('float')
    expected_df.columns = pd.MultiIndex.from_tuples(expected_df.columns)
    result_df = colab_utils.rank_models(
        input_df,
        drop_1shot=True,
        datasets=['cifar10', 'few-shot pets'],
        drop_incomplete_measurements=drop_incomplete_measurements)

    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

  @parameterized.named_parameters(
      dict(
          testcase_name='_keep_all_measurements',
          drop_incomplete_measurements=False,
          expected_result={
              'prediction':
                  pd.DataFrame({
                      'model': ['Det', 'BE', 'GP'],
                      ('test_prec@1', 'cifar10'): [3, 1, 2],
                  }),
              'uncertainty':
                  pd.DataFrame({
                      'model': ['Det', 'BE', 'GP'],
                      ('ood_cifar100_msp_auroc_ece', 'cifar10'): [3, 2, 1],
                  }),
              'adaptation':
                  pd.DataFrame({
                      'model': ['Det', 'BE', 'GP'],
                      ('1shot_prec@1', 'few-shot pets'): [1, 2, np.nan],
                      ('5shot_prec@1', 'few-shot pets'): [2, 1, np.nan],
                  }),
          }),
      dict(
          testcase_name='_drop_incomplete_measurements',
          drop_incomplete_measurements=True,
          expected_result={
              'prediction':
                  pd.DataFrame({
                      'model': ['Det', 'BE'],
                      ('test_prec@1', 'cifar10'): [2, 1],
                  }),
              'uncertainty':
                  pd.DataFrame({
                      'model': ['Det', 'BE'],
                      ('ood_cifar100_msp_auroc_ece', 'cifar10'): [2, 1],
                  }),
              'adaptation':
                  pd.DataFrame({
                      'model': ['Det', 'BE'],
                      ('1shot_prec@1', 'few-shot pets'): [1, 2],
                      ('5shot_prec@1', 'few-shot pets'): [2, 1],
                  }),
          }),
  )
  def test_ranks_by_category(self, drop_incomplete_measurements,
                             expected_result):
    input_df = get_test_dataframe_for_scoring()

    models_by_category = colab_utils.rank_models_by_category(
        input_df,
        drop_1shot=False,
        datasets=['cifar10', 'few-shot pets'],
        drop_incomplete_measurements=drop_incomplete_measurements)

    self.assertSetEqual(set(models_by_category),
                        set(m.name.lower() for m in colab_utils.MetricCategory))

    for key, result_df in models_by_category.items():
      expected_df = expected_result[key].set_index('model')
      expected_df.columns = pd.MultiIndex.from_tuples(expected_df.columns)
      pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)


if __name__ == '__main__':
  absltest.main()
