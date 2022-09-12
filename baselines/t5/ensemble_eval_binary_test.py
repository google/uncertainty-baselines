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

"""Tests for ensemble_eval binary."""

import functools
import os
import tempfile

from absl.testing import absltest
import numpy as np
import seqio
import t5.data
from t5x import adafactor
from t5x import eval as eval_lib
from t5x import gin_utils
from t5x import partitioning
from t5x import train as train_lib
from t5x import trainer
from t5x import utils
import t5x.examples.t5.network as t5_network
import tensorflow as tf

from models import models as ub_models  # local file import from baselines.t5
import uncertainty_baselines.baselines.t5.utils as ub_utils


class EnsembleEvalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # pylint: disable=unused-import,g-import-not-at-top
    # Register necessary SeqIO Tasks/Mixtures.
    import data.mixtures  # local file import from baselines.t5
    import data.tasks.mnli  # local file import from baselines.t5
    # pylint: enable=unused-import,g-import-not-at-top

    self.mnli_cfg = utils.DatasetConfig(
        mixture_or_task_name='mnli_mismatched',
        task_feature_lengths={
            'inputs': 10,
            'targets': 1,
        },
        split='validation',
        batch_size=8,
        shuffle=False,
        seed=0)

    vocabulary = t5.data.get_default_vocabulary()
    transformer_config = t5_network.T5Config(
        vocab_size=vocabulary.vocab_size,
        emb_dim=4,
        num_heads=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        head_dim=2,
        mlp_dim=8)
    module = t5_network.Transformer(config=transformer_config)

    optimizer_def = adafactor.Adafactor()
    # We will save all checkpoints to perform ensembling.
    checkpoint_cfg = utils.CheckpointConfig(
        save=utils.SaveCheckpointConfig(dtype='float32', period=1))
    trainer_cls = functools.partial(
        trainer.Trainer,
        learning_rate_fn=utils.create_learning_rate_scheduler(),
        num_microbatches=None)
    self.partitioner = partitioning.PjitPartitioner(
        num_partitions=1,
        model_parallel_submesh=None,
        logical_axis_rules=partitioning.standard_logical_axis_rules())

    def do_training(model, model_dir, train_dataset_cfg):
      train_lib.train(
          model=model,
          train_dataset_cfg=train_dataset_cfg,
          train_eval_dataset_cfg=None,
          infer_eval_dataset_cfg=None,
          checkpoint_cfg=checkpoint_cfg,
          partitioner=self.partitioner,
          trainer_cls=trainer_cls,
          model_dir=model_dir,
          total_steps=2,
          eval_steps=4,
          eval_period=4,
          random_seed=0,
          summarize_config_fn=gin_utils.summarize_gin_config,
      )

    self.do_training = do_training
    self.model = ub_models.EncoderDecoderClassifierModel(
        label_tokens=('<extra_id_0>', '<extra_id_1>'),
        module=module,
        input_vocabulary=vocabulary,
        output_vocabulary=vocabulary,
        optimizer_def=optimizer_def)
    self.inference_evaluator_cls = functools.partial(
        seqio.Evaluator,
        logger_cls=[
            seqio.PyLoggingLogger, seqio.TensorBoardLogger, seqio.JSONLogger
        ])

  def test_binary_classification(self):
    train_dataset_cfg = self.mnli_cfg
    with tempfile.TemporaryDirectory() as model_dir:
      self.do_training(self.model, model_dir, train_dataset_cfg)

      output_dir = os.path.join(model_dir, 'ensemble')
      # Use all checkpoints.
      restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
          path=model_dir, mode='all')
      eval_lib.evaluate(
          model=self.model,
          dataset_cfg=train_dataset_cfg,
          restore_checkpoint_cfg=restore_checkpoint_cfg,
          partitioner=self.partitioner,
          output_dir=output_dir,
          inference_evaluator_cls=self.inference_evaluator_cls,
          train_state_initializer_cls=functools.partial(
              ub_utils.TrainStateEnsembleInitializer, ensemble_size=3))

      # Collect train loss and evaluate loss.
      path = os.path.join(output_dir, 'inference_eval',
                          train_dataset_cfg.mixture_or_task_name)
      train_summary = tf.compat.v1.train.summary_iterator(
          os.path.join(path,
                       os.listdir(path)[0]))
      for e in train_summary:
        for v in e.summary.value:
          if v.tag == 'eval/accuracy':
            acc = tf.make_ndarray(v.tensor)

    # TODO(phandu): Comment here sources of non-determinism if this check
    # fails in the future.
    np.testing.assert_allclose(acc, 37.8255, atol=1e-3)


if __name__ == '__main__':
  absltest.main()
