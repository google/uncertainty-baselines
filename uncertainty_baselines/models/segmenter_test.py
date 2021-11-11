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

"""Tests for the Segmenter Model."""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import ml_collections
import uncertainty_baselines as ub

# TO DO debug updating parameters
class SegmenterTest(parameterized.TestCase):


  def test_Segmenter(self, ):

    """Experiment configuration for segmenter."""
    config = ml_collections.ConfigDict()
    config.model = ml_collections.ConfigDict()

    block_size = ml_collections.FieldReference((64, 128, 256, 512, 1024, 1024))
    use_norm = ml_collections.FieldReference(True)
    padding = ml_collections.FieldReference('SAME')

    # encoder configs
    config.model.encoder = ml_collections.ConfigDict()
    config.model.encoder.type = 'conv'
    config.model.encoder.block_size = block_size
    config.model.encoder.use_norm = use_norm
    config.model.encoder.padding = padding

    # decoder configs
    config.model.decoder = ml_collections.ConfigDict()
    config.model.decoder.type = 'conv'
    config.model.decoder.block_size = block_size
    config.model.decoder.use_norm = use_norm
    config.model.decoder.padding = padding
    config.model.decoder.last_norm = True

    # Optimizer.
    config.batch_size = 1
    config.num_training_epochs = 2
    config.optimizer = 'adam'

    config.rng_seed = 0

    # Create Dataset
    config.dataset_meta_data = {'num_clases': 34}
    config.dataset_configs = ml_collections.ConfigDict()
    config.target_size = [1024, 2028]
    # Update for JAX dataset
    seed = 83922
    tf.random.set_seed(seed)
    dataset_size = 10
    batch_size = 5
    input_shape = (1024, 2048, 3)
    num_classes = 34
    features = tf.random.normal((dataset_size,) + input_shape)
    labels = tf.ones((dataset_size,) + input_shape[:2] + (1,))
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset = dataset.repeat().shuffle(dataset_size).batch(batch_size)

    model = ub.models.segmenter.SegVitModel.build_flax_model(config)
    input_shape=input_shape,
    filters=[512, 256, 128, 64],
    num_classes=num_classes,
    seed=seed,
    )



if __name__ == '__main__':
  absltest.main()
