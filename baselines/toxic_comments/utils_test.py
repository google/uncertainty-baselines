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

"""Tests for utils."""
import tensorflow as tf
import utils  # local file import from baselines.toxic_comments


class UtilsTest(tf.test.TestCase):

  def test_make_cv_train_and_eval_splits(self):
    num_folds = 10
    train_fold_ids = ['2', '5']

    (train_split, eval_split, train_folds, eval_folds,
     eval_fold_ids) = utils.make_cv_train_and_eval_splits(
         num_folds, train_fold_ids, return_individual_folds=True)

    expected_train_folds = ['train[20%:30%]', 'train[50%:60%]']
    expected_eval_folds = [
        'train[0%:10%]', 'train[10%:20%]', 'train[30%:40%]', 'train[40%:50%]',
        'train[60%:70%]', 'train[70%:80%]', 'train[80%:90%]', 'train[90%:100%]'
    ]
    expected_eval_fold_ids = [0, 1, 3, 4, 6, 7, 8, 9]

    self.assertEqual(train_split, 'train[20%:30%]+train[50%:60%]')
    self.assertEqual(eval_split, '+'.join(expected_eval_folds))
    self.assertListEqual(train_folds, expected_train_folds)
    self.assertListEqual(eval_folds, expected_eval_folds)
    self.assertListEqual(eval_fold_ids, expected_eval_fold_ids)


if __name__ == '__main__':
  tf.test.main()
