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
from unittest import mock
import warnings

from absl.testing import absltest
import jax
import jax.numpy as jnp
import numpy as np
import t5.data
from t5x import adafactor
from t5x import eval as eval_lib
from t5x import gin_utils
from t5x import partitioning
from t5x import train as train_lib
from t5x import trainer
from t5x import utils
import t5x.examples.t5.network as t5_network

from data import metrics  # local file import from baselines.t5
from decoding import beam_search  # local file import from baselines.t5
from models import models as ub_models  # local file import from baselines.t5
import uncertainty_baselines.baselines.t5.utils as ub_utils


class EnsembleEvalTest(absltest.TestCase):

  def setUp(self):
    super().setUp()

    # pylint: disable=unused-import,g-import-not-at-top
    # Register necessary SeqIO Tasks/Mixtures.
    import data.mixtures  # local file import from baselines.t5
    from uncertainty_baselines.baselines.t5.data.tasks.nalue import get_nalue_intent_tokens
    # pylint: enable=unused-import,g-import-not-at-top

    self.nalue_tokens = get_nalue_intent_tokens()
    self.nalue_cfg = utils.DatasetConfig(
        mixture_or_task_name='nalue_standard_oos',
        task_feature_lengths={
            'inputs': 10,
            'targets': 3,
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
          use_gda=False,
      )

    self.do_training = do_training

    # Decoding override to write Top-K beams and scores to output.
    class ClassifierModel(ub_models.EncoderDecoderClassifierModel):
      # Add a flag to check whether flat_ids accross different parameter
      # checkpoints are equal.
      deduplicate_flat_ids = False

      def score_batch(self, *args, **kwargs):
        kwargs['return_beam_predictions'] = True
        return super().score_batch(*args, **kwargs)

      def predict_batch_with_aux(self, *args, **kwargs):
        kwargs['num_decodes'] = 3
        kwargs['return_all_decodes'] = True
        return super().predict_batch_with_aux(*args, **kwargs)

      def _compute_logits_from_slice(self,
                                     decoding_state,
                                     params,
                                     encoded_inputs,
                                     raw_inputs,
                                     max_decode_length,
                                     rngs=None,
                                     ensemble_probs=True):
        if self.deduplicate_flat_ids:
          warnings.warn('Deduplicate flat_ids')
          k = jax.tree_util.tree_flatten(params)[0][0].shape[0]
          # Select 1 replica and populate it accross k replicas.
          replica = decoding_state.cur_token[:(
              decoding_state.cur_token.shape[0] // k)]
          flat_ids = jnp.broadcast_to(replica, (k,) + replica.shape)
          flat_ids = jnp.reshape(flat_ids, (-1,) + flat_ids.shape[2:])
          decoding_state.replace(cur_token=flat_ids)

        return super()._compute_logits_from_slice(
            decoding_state,
            params,
            encoded_inputs,
            raw_inputs,
            max_decode_length,
            rngs=rngs,
            ensemble_probs=ensemble_probs)

    # Remove the brevity penalty that is unnecessary for classification.
    decode_fn = functools.partial(
        beam_search, alpha=0., return_token_scores=True)
    self.model_class = functools.partial(
        ClassifierModel,
        module=module,
        input_vocabulary=vocabulary,
        output_vocabulary=vocabulary,
        optimizer_def=optimizer_def,
        decode_fn=decode_fn)

    def extract_beam_results(scores):
      beam_predictions = [beam_score[1] for beam_score in scores]
      beam_scores = [beam_score[2] for beam_score in scores]
      beam_scores = jnp.array(beam_scores, dtype=jnp.float32)
      self.global_metrics['beam_predictions'] = jnp.array(beam_predictions)
      self.global_metrics['beam_scores'] = beam_scores
      return beam_predictions, beam_scores

    self.global_metrics = {}
    self.extract_beam_results = extract_beam_results

  def test_beam_predictions_reshape_logic(self):
    # We want to test that the results obtained from a checkpoint is consistent
    # regardless the ensemble size.
    model = self.model_class(label_tokens=self.nalue_tokens)
    train_dataset_cfg = self.nalue_cfg
    beam_scores = {}
    beam_predictions = {}
    with tempfile.TemporaryDirectory() as model_dir:
      self.do_training(model, model_dir, train_dataset_cfg)
      checkpoint_path = os.path.join(model_dir, 'checkpoint_2')

      for ensemble_size in [1, 2]:
        metric_name = f'ensemble_{ensemble_size}'
        output_dir = os.path.join(model_dir, metric_name)
        restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
            path=[checkpoint_path] * ensemble_size,
            mode='specific',
            use_gda=False)

        with mock.patch.object(
            metrics,
            '_extract_beam_results_from_scores',
            side_effect=self.extract_beam_results):
          eval_lib.evaluate(
              model=model,
              dataset_cfg=train_dataset_cfg,
              restore_checkpoint_cfg=restore_checkpoint_cfg,
              partitioner=self.partitioner,
              output_dir=output_dir,
              train_state_initializer_cls=functools.partial(
                  ub_utils.TrainStateEnsembleInitializer,
                  ensemble_size=ensemble_size))

        self.assertIn('beam_predictions', self.global_metrics)
        self.assertIn('beam_scores', self.global_metrics)
        beam_predictions[metric_name] = self.global_metrics.pop(
            'beam_predictions')
        beam_scores[metric_name] = self.global_metrics.pop('beam_scores')

    np.testing.assert_allclose(beam_predictions['ensemble_1'],
                               beam_predictions['ensemble_2'])
    np.testing.assert_allclose(
        beam_scores['ensemble_1'], beam_scores['ensemble_2'], atol=1e-4)

  def test_flat_ids_are_constant_accross_replicas(self):
    model = self.model_class(label_tokens=self.nalue_tokens)
    train_dataset_cfg = self.nalue_cfg
    beam_scores = {}
    beam_predictions = {}
    with tempfile.TemporaryDirectory() as model_dir:
      self.do_training(model, model_dir, train_dataset_cfg)
      # Use all checkpoints.
      restore_checkpoint_cfg = utils.RestoreCheckpointConfig(
          path=model_dir, mode='all', use_gda=False)

      for deduplicate_flat_ids in [False, True]:
        metric_name = f'dedup_{deduplicate_flat_ids}'
        output_dir = os.path.join(model_dir, metric_name)

        with warnings.catch_warnings(record=True) as ws:
          model.deduplicate_flat_ids = deduplicate_flat_ids
          with mock.patch.object(
              metrics,
              '_extract_beam_results_from_scores',
              side_effect=self.extract_beam_results):
            eval_lib.evaluate(
                model=model,
                dataset_cfg=train_dataset_cfg,
                restore_checkpoint_cfg=restore_checkpoint_cfg,
                partitioner=self.partitioner,
                output_dir=output_dir,
                train_state_initializer_cls=functools.partial(
                    ub_utils.TrainStateEnsembleInitializer, ensemble_size=3))

          # Check if the deduplicate code actually triggers.
          if deduplicate_flat_ids:
            expected_msg = 'Deduplicate flat_ids'
            self.assertTrue(any(expected_msg in str(w.message) for w in ws))

        self.assertIn('beam_predictions', self.global_metrics)
        self.assertIn('beam_scores', self.global_metrics)
        beam_predictions[metric_name] = self.global_metrics.pop(
            'beam_predictions')
        beam_scores[metric_name] = self.global_metrics.pop('beam_scores')

    np.testing.assert_allclose(beam_predictions['dedup_True'],
                               beam_predictions['dedup_False'])
    np.testing.assert_allclose(
        beam_scores['dedup_True'], beam_scores['dedup_False'], atol=1e-4)


if __name__ == '__main__':
  absltest.main()
