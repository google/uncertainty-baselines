import haiku as hk
import jax
import tensorflow as tf

import uncertainty_baselines as ub
from uncertainty_baselines.models.resnet50_fsvi import zero_padding_2D


class ResNet50FSVITest(tf.test.TestCase):

    def testCreateModel(self):
        batch_size = 2
        def forward(inputs, rng_key, stochastic, is_training):
            model = ub.models.ResNet50FSVI(
                output_dim=10,
                stochastic_parameters=True,
                dropout=False,
                dropout_rate=0.,)
            return model(inputs, rng_key, stochastic, is_training)

        init_fn, apply_fn = hk.transform_with_state(forward)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(batch_size, 32, 32, 3))
        params, state = init_fn(
            key, x, key, stochastic=True, is_training=True
        )
        output, new_state = apply_fn(
            params, state, key, x, key, stochastic=True, is_training=True
        )
        self.assertEqual(output.shape, (31, 10))

    def test_zero_padding_2D(self):
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, shape=(2, 32, 32, 3))
        padding = 3
        actual = zero_padding_2D(x, padding=padding)
        # TODO: make this work, currently have the following error
        # tensorflow.python.framework.errors_impl.InternalError: Cannot dlopen all CUDA libraries
        expected = tf.keras.layers.ZeroPadding2D(padding=3)(x)
        print(actual.shape)


if __name__ == '__main__':
    tf.test.main()
