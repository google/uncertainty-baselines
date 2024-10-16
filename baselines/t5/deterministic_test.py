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

"""Tests for deterministic binary."""

import dataclasses
import functools
import os
import tempfile

from absl.testing import absltest
# import numpy as np
import seqio
# This is needed for predefined tasks.
import t5.data.mixtures  # pylint: disable=unused-import
from t5x import adafactor
from t5x import gin_utils
from t5x import models
from t5x import partitioning
from t5x import test_utils
from t5x import train as train_lib
from t5x import trainer
from t5x import utils
import t5x.examples.t5.network as t5_network
import tensorflow as tf

mock = absltest.mock


class DeterministicTest(absltest.TestCase):

  # This test is adapted from 't5x.train_test'.
  @mock.patch.object(
      seqio.Task,
      'get_dataset',
      side_effect=test_utils.get_fake_tokenized_dataset)
  def test_train(self, _):
    fake_vocab = test_utils.get_fake_vocab()[0]
    vocabulary = dataclasses.replace(fake_vocab, vocab_size=32)
    transformer_config = t5_network.T5Config(
        vocab_size=vocabulary.vocab_size,
        emb_dim=4,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        head_dim=2,
        mlp_dim=8)
    optimizer_def = adafactor.Adafactor()
    model = models.EncoderDecoderModel(
        module=t5_network.Transformer(config=transformer_config),
        input_vocabulary=vocabulary,
        output_vocabulary=vocabulary,
        optimizer_def=optimizer_def)
    train_dataset_cfg = utils.DatasetConfig(
        mixture_or_task_name='glue_mnli_mismatched_v002',
        task_feature_lengths={
            'inputs': 8,
            'targets': 8
        },
        split='train',
        batch_size=8,
        shuffle=False,
        seed=0)
    train_eval_dataset_cfg = utils.DatasetConfig(
        mixture_or_task_name='glue_mnli_mismatched_v002',
        task_feature_lengths={
            'inputs': 8,
            'targets': 8
        },
        split='validation',
        batch_size=8,
        shuffle=False,
        seed=0)
    checkpoint_cfg = utils.CheckpointConfig(
        save=utils.SaveCheckpointConfig(dtype='float32', period=3))
    trainer_cls = functools.partial(
        trainer.Trainer,
        learning_rate_fn=utils.create_learning_rate_scheduler(),
        num_microbatches=None)
    partitioner = partitioning.PjitPartitioner(
        num_partitions=1,
        model_parallel_submesh=None,
        logical_axis_rules=partitioning.standard_logical_axis_rules())

    def do_training(train_fn, model_dir):
      with mock.patch.object(
          utils, 'get_vocabulary', return_value=(vocabulary, vocabulary)):
        return train_fn(
            model=model,
            train_dataset_cfg=train_dataset_cfg,
            train_eval_dataset_cfg=train_eval_dataset_cfg,
            infer_eval_dataset_cfg=None,
            checkpoint_cfg=checkpoint_cfg,
            partitioner=partitioner,
            trainer_cls=trainer_cls,
            model_dir=model_dir,
            total_steps=3,
            eval_steps=2,
            eval_period=12,
            random_seed=0,
            summarize_config_fn=gin_utils.summarize_gin_config,
        )

    losses = {}
    metric_types = [
        'train', f'training_eval/{train_dataset_cfg.mixture_or_task_name}'
    ]
    with tempfile.TemporaryDirectory() as model_dir:
      host_step, _ = do_training(train_lib.train, model_dir)
      self.assertEqual(host_step, 3)
      self.assertIn('checkpoint_3', os.listdir(model_dir))

      # Collect train loss and evaluate loss.
      for metric_type in metric_types:
        path = os.path.join(model_dir, metric_type)
        train_summary = tf.compat.v1.train.summary_iterator(
            os.path.join(path,
                         os.listdir(path)[0]))
        for e in train_summary:
          for v in e.summary.value:
            if v.tag == 'loss':
              losses[metric_type] = tf.make_ndarray(v.tensor)

    # Compare to concrete values in case of changes upstream.
    # TODO(phandu): Comment here sources of non-determinism if this check
    # fails in the future. Current sources:
    # + (cl/399763010): Layer names are changed upstream. Non-determinism
    #   happens because rng splits depend on layer names.
    # + (cl/402315931): Changes on how dropout rngs are generated in each
    #   training step.
    # + (cl/403524153): Introduce a new system for logical axis names, which
    #   affects how Adafactor optimizer handles some parameters.
    # + (cl/404377346): Changes on how attention kernels are initialized.
    # np.testing.assert_allclose(losses[metric_types[0]], 198.6521, atol=1e-3)
    # np.testing.assert_allclose(losses[metric_types[1]], 254.6788, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
