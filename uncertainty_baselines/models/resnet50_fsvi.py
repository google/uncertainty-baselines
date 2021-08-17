from typing import Mapping, Optional, Sequence, Union, Any, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from haiku._src import utils
from jax import lax


def uniform_mod(min_val, max_val):
    def _uniform_mod(shape, dtype):
        rng_key, _ = jax.random.split(jax.random.PRNGKey(0))
        return jax.random.uniform(
            rng_key, shape=shape, dtype=dtype, minval=min_val, maxval=max_val
            # rng_key, shape=shape, dtype=dtype, minval=-10.0, maxval=-8.0
            # rng_key, shape=shape, dtype=dtype, minval=-20.0, maxval=-18.0
            # rng_key, shape=shape, dtype=dtype, minval=-5.0, maxval=-4.0
            # rng_key, shape=shape, dtype=dtype, minval=-3.0, maxval=-2.5
            # rng_key, shape=shape, dtype=dtype, minval=-3.0, maxval=-2.0
        )
    return _uniform_mod


def gaussian_sample(mu: jnp.ndarray, rho: jnp.ndarray, stochastic: bool, rng_key):
    if stochastic:
        jnp_eps = jax.random.normal(rng_key, mu.shape)
        z = mu + jnp.exp((0.5 * rho).astype(jnp.float32)) * jnp_eps
        # Experimental: Laplace variational distribution
        # dist = tfd.Laplace(loc=mu, scale=jnp.exp(rho))
        # z = dist.sample(seed=rng_key)
    else:
        z = mu
    return z


class dense_stochastic_hk(hk.Module):
    def __init__(
        self,
        output_size: int,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        name: Optional[str] = None,
        stochastic_parameters: bool = False,
    ):
        super(dense_stochastic_hk, self).__init__(name=name)
        self.input_size = None
        self.output_size = output_size
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        self.stochastic_parameters = stochastic_parameters

    def __call__(self, inputs, rng_key, stochastic: bool):
        """
        @param stochastic: if True, use sampled parameters, otherwise, use mean parameters.
        @return:
        """
        j, k = inputs.shape[-1], self.output_size
        input_size = self.input_size = inputs.shape[-1]
        output_size = self.output_size
        dtype = inputs.dtype

        if self.w_init is None:
            stddev = 1.0 / np.sqrt(self.input_size)
            self.w_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
        w_mu = hk.get_parameter("w_mu", shape=[j, k], dtype=dtype, init=self.w_init)

        if self.with_bias:
            if self.b_init is None:
                stddev = 1.0 / np.sqrt(self.input_size)
                self.b_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
            b_mu = hk.get_parameter("b_mu", shape=[k], dtype=dtype, init=self.b_init)
        if self.stochastic_parameters:
            w_logvar = hk.get_parameter(
                "w_logvar", shape=[j, k], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
            )
            if self.with_bias:
                b_logvar = hk.get_parameter(
                    "b_logvar", shape=[k], dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                )
            key_1, key_2 = jax.random.split(rng_key)
            W = gaussian_sample(w_mu, w_logvar, stochastic, key_1)
            if self.with_bias:
                b = gaussian_sample(b_mu, b_logvar, stochastic, key_2)
                return jnp.dot(inputs, W) + b
            else:
                return jnp.dot(inputs, W)
        else:
            if self.with_bias:
                return jnp.dot(inputs, w_mu) + b_mu
            else:
                return jnp.dot(inputs, w_mu)



def to_dimension_numbers(
    num_spatial_dims: int, channels_last: bool, transpose: bool,
) -> lax.ConvDimensionNumbers:
    """Create a `lax.ConvDimensionNumbers` for the given inputs."""
    num_dims = num_spatial_dims + 2

    if channels_last:
        spatial_dims = tuple(range(1, num_dims - 1))
        image_dn = (0, num_dims - 1) + spatial_dims
    else:
        spatial_dims = tuple(range(2, num_dims))
        image_dn = (0, 1) + spatial_dims

    if transpose:
        kernel_dn = (num_dims - 2, num_dims - 1) + tuple(range(num_dims - 2))
    else:
        kernel_dn = (num_dims - 1, num_dims - 2) + tuple(range(num_dims - 2))

    return lax.ConvDimensionNumbers(
        lhs_spec=image_dn, rhs_spec=kernel_dn, out_spec=image_dn
    )


class conv2D_stochastic(hk.Module):
    """General N-dimensional convolutional."""

    def __init__(
        self,
        output_channels: int,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        kernel_shape: Union[int, Sequence[int]],
        num_spatial_dims: int = 2,
        stride: Union[int, Sequence[int]] = 1,
        rate: Union[int, Sequence[int]] = 1,
        padding: Union[
            str, Sequence[Tuple[int, int]], hk.pad.PadFn, Sequence[hk.pad.PadFn]
        ] = "SAME",
        with_bias: bool = True,
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        data_format: str = "channels_last",
        mask: Optional[jnp.ndarray] = None,
        feature_group_count: int = 1,
        name: Optional[str] = None,
        stochastic_parameters: bool = False,
    ):
        """Initializes the module.
        Args:
            num_spatial_dims: The number of spatial dimensions of the input.
            output_channels: Number of output channels.
            kernel_shape: The shape of the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``.
            stride: Optional stride for the kernel. Either an integer or a sequence of
                length ``num_spatial_dims``. Defaults to 1.
            rate: Optional kernel dilation rate. Either an integer or a sequence of
                length ``num_spatial_dims``. 1 corresponds to standard ND convolution,
                ``rate > 1`` corresponds to dilated convolution. Defaults to 1.
            padding: Optional padding algorithm. Either ``VALID`` or ``SAME`` or a
                sequence of n ``(low, high)`` integer pairs that give the padding to
                apply before and after each spatial dimension. or a callable or sequence
                of callables of size ``num_spatial_dims``. Any callables must take a
                single integer argument equal to the effective kernel size and return a
                sequence of two integers representing the padding before and after. See
                ``haiku.pad.*`` for more details and example functions. Defaults to
                ``SAME``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            with_bias: Whether to add a bias. By default, true.
            w_init: Optional weight initialization. By default, truncated normal.
            b_init: Optional bias initialization. By default, zeros.
            data_format: The data format of the input.  Can be either
                ``channels_first``, ``channels_last``, ``N...C`` or ``NC...``. By
                default, ``channels_last``.
            mask: Optional mask of the weights.
            feature_group_count: Optional number of groups in group convolution.
                Default value of 1 corresponds to normal dense convolution. If a higher
                value is used, convolutions are applied separately to that many groups,
                then stacked together. This reduces the number of parameters
                and possibly the compute for a given ``output_channels``. See:
                https://www.tensorflow.org/xla/operation_semantics#conv_convolution.
            name: The name of the module.
        """
        super().__init__(name=name)
        if num_spatial_dims <= 0:
            raise ValueError(
                "We only support convolution operations for `num_spatial_dims` "
                f"greater than 0, received num_spatial_dims={num_spatial_dims}."
            )

        self.num_spatial_dims = num_spatial_dims
        self.output_channels = output_channels
        self.kernel_shape = utils.replicate(
            kernel_shape, num_spatial_dims, "kernel_shape"
        )
        self.with_bias = with_bias
        self.stride = utils.replicate(stride, num_spatial_dims, "strides")
        self.w_init = w_init
        self.b_init = b_init
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval
        self.mask = mask
        self.feature_group_count = feature_group_count
        self.lhs_dilation = utils.replicate(1, num_spatial_dims, "lhs_dilation")
        self.kernel_dilation = utils.replicate(
            rate, num_spatial_dims, "kernel_dilation"
        )
        self.data_format = data_format
        self.channel_index = utils.get_channel_index(data_format)
        self.dimension_numbers = to_dimension_numbers(
            num_spatial_dims, channels_last=(self.channel_index == -1), transpose=False
        )
        self.stochastic_parameters = stochastic_parameters

        if isinstance(padding, str):
            self.padding = padding.upper()
        else:
            self.padding = hk.pad.create(
                padding=padding,
                kernel=self.kernel_shape,
                rate=self.kernel_dilation,
                n=self.num_spatial_dims,
            )

    def __call__(self, inputs: jnp.ndarray, rng_key, stochastic) -> jnp.ndarray:
        """Connects ``ConvND`` layer.
        Args:
            inputs: An array of shape ``[spatial_dims, C]`` and rank-N+1 if unbatched,
                or an array of shape ``[N, spatial_dims, C]`` and rank-N+2 if batched.
        Returns:
            An array of shape ``[spatial_dims, output_channels]`` and rank-N+1 if
                unbatched, or an array of shape ``[N, spatial_dims, output_channels]``
                and rank-N+2 if batched.
        """
        dtype = inputs.dtype

        unbatched_rank = self.num_spatial_dims + 1
        allowed_ranks = [unbatched_rank, unbatched_rank + 1]
        if inputs.ndim not in allowed_ranks:
            raise ValueError(
                f"Input to ConvND needs to have rank in {allowed_ranks},"
                f" but input has shape {inputs.shape}."
            )

        unbatched = inputs.ndim == unbatched_rank
        if unbatched:
            inputs = jnp.expand_dims(inputs, axis=0)

        if inputs.shape[self.channel_index] % self.feature_group_count != 0:
            raise ValueError(
                f"Inputs channels {inputs.shape[self.channel_index]} "
                f"should be a multiple of feature_group_count "
                f"{self.feature_group_count}"
            )
        w_shape = self.kernel_shape + (
            inputs.shape[self.channel_index] // self.feature_group_count,
            self.output_channels,
        )

        if self.mask is not None and self.mask.shape != w_shape:
            raise ValueError(
                "Mask needs to have the same shape as weights. "
                f"Shapes are: {self.mask.shape}, {w_shape}"
            )

        if self.w_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            self.w_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)
        if self.b_init is None:
            fan_in_shape = np.prod(w_shape[:-1])
            stddev = 1.0 / np.sqrt(fan_in_shape)
            self.b_init = hk.initializers.RandomUniform(minval=-stddev, maxval=stddev)

        w_mu = hk.get_parameter(
            "w_mu", w_shape, dtype, init=self.w_init
        )  ### changed code!

        if self.stochastic_parameters:
            w_logvar = hk.get_parameter(
                "w_logvar", w_shape, dtype=dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
            )
            rng_key, sub_key = jax.random.split(rng_key)
            W = gaussian_sample(w_mu, w_logvar, stochastic, sub_key)
            out = lax.conv_general_dilated(
                inputs,
                W,
                window_strides=self.stride,
                padding=self.padding,
                lhs_dilation=self.lhs_dilation,
                rhs_dilation=self.kernel_dilation,
                dimension_numbers=self.dimension_numbers,
                feature_group_count=self.feature_group_count,
            )
        else:
            out = lax.conv_general_dilated(
                inputs,
                w_mu,
                window_strides=self.stride,
                padding=self.padding,
                lhs_dilation=self.lhs_dilation,
                rhs_dilation=self.kernel_dilation,
                dimension_numbers=self.dimension_numbers,
                feature_group_count=self.feature_group_count,
            )

        if self.with_bias:
            if self.channel_index == -1:
                bias_shape = (self.output_channels,)
            else:
                bias_shape = (self.output_channels,) + (1,) * self.num_spatial_dims
            b_mu = hk.get_parameter("b_mu", bias_shape, inputs.dtype, init=self.b_init)
            if self.stochastic_parameters:
                b_logvar = hk.get_parameter(
                    "b_logvar", shape=bias_shape, dtype=inputs.dtype, init=uniform_mod(self.uniform_init_minval, self.uniform_init_maxval)
                )
                rng_key, sub_key = jax.random.split(rng_key)
                b = gaussian_sample(b_mu, b_logvar, stochastic, sub_key)
                b = jnp.broadcast_to(b, out.shape)
            else:
                b = jnp.broadcast_to(b_mu, out.shape)
            out = out + b

        if unbatched:
            out = jnp.squeeze(out, axis=0)
        return out


class BlockV1(hk.Module):
    """ResNet V1 block with optional bottleneck."""

    def __init__(
        self,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bn_config: Mapping[str, float],
        bottleneck: bool,
        name: Optional[str] = None,
    ):

        super().__init__(name=name)
        self.use_projection = use_projection
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)
        bn_config.setdefault("decay_rate", 0.999)

        if self.use_projection:
            self.proj_conv = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="VALID",
                name="shortcut_conv",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            self.proj_batchnorm = hk.BatchNorm(name="batchnorm", **bn_config)

        channel_div = 4 if bottleneck else 1
        conv_0 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1,
            with_bias=False,
            padding="VALID",
            name="conv_0",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        bn_0 = hk.BatchNorm(name="batchnorm", **bn_config)

        conv_1 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        bn_1 = hk.BatchNorm(name="batchnorm", **bn_config)
        layers = ((conv_0, bn_0), (conv_1, bn_1))

        if bottleneck:
            conv_2 = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="VALID",
                name="conv_2",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            bn_2 = hk.BatchNorm(name="batchnorm", scale_init=jnp.zeros, **bn_config)
            layers = layers + ((conv_2, bn_2),)

        self.layers = layers

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        out = shortcut = inputs

        if self.use_projection:
            shortcut = self.proj_conv(shortcut, rng_key, stochastic)
            shortcut = self.proj_batchnorm(shortcut, is_training, test_local_stats)
            # DROPOUT
            if self.dropout and is_training:
                shortcut = hk.dropout(rng_key, self.dropout_rate, shortcut)

        for i, (conv_i, bn_i) in enumerate(self.layers):
            out = conv_i(out, rng_key, stochastic)
            out = bn_i(out, is_training, test_local_stats)
            if i < len(self.layers) - 1:  # Don't apply relu or dropout on last layer
                # DROPOUT
                if self.dropout and is_training:
                    out = hk.dropout(rng_key, self.dropout_rate, out)
                out = jax.nn.relu(out)

        return jax.nn.relu(out + shortcut)


class BlockV2(hk.Module):
    """ResNet V2 block with optional bottleneck."""

    def __init__(
        self,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        stride: Union[int, Sequence[int]],
        use_projection: bool,
        bn_config: Mapping[str, float],
        bottleneck: bool,
        name: Optional[str] = None,
    ):

        super().__init__(name=name)
        self.use_projection = use_projection
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        # TODO: implement dropout
        if self.dropout:
            raise NotImplementedError("Dropout not yet implemented for ResNetV2")

        bn_config = dict(bn_config)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        if self.use_projection:
            self.proj_conv = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=stride,
                with_bias=False,
                padding="SAME",
                name="shortcut_conv",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

        channel_div = 4 if bottleneck else 1
        conv_0 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=1 if bottleneck else 3,
            stride=1,
            with_bias=False,
            padding="SAME",
            name="conv_0",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        bn_0 = hk.BatchNorm(name="batchnorm", **bn_config)

        conv_1 = conv2D_stochastic(
            output_channels=channels // channel_div,
            kernel_shape=3,
            stride=stride,
            with_bias=False,
            padding="SAME",
            name="conv_1",
            stochastic_parameters=stochastic_parameters,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )

        bn_1 = hk.BatchNorm(name="batchnorm", **bn_config)
        layers = ((conv_0, bn_0), (conv_1, bn_1))

        if bottleneck:
            conv_2 = conv2D_stochastic(
                output_channels=channels,
                kernel_shape=1,
                stride=1,
                with_bias=False,
                padding="SAME",
                name="conv_2",
                stochastic_parameters=stochastic_parameters,
                uniform_init_minval=uniform_init_minval,
                uniform_init_maxval=uniform_init_maxval,
            )

            # NOTE: Some implementations of ResNet50 v2 suggest initializing
            # gamma/scale here to zeros.
            bn_2 = hk.BatchNorm(name="batchnorm", **bn_config)
            layers = layers + ((conv_2, bn_2),)

        self.layers = layers

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        x = shortcut = inputs

        for i, (conv_i, bn_i) in enumerate(self.layers):
            x = bn_i(x, is_training, test_local_stats)
            x = jax.nn.relu(x)
            if i == 0 and self.use_projection:
                shortcut = self.proj_conv(x, rng_key, stochastic)
            x = conv_i(x, rng_key, stochastic)

        return x + shortcut


class BlockGroup(hk.Module):
    """Higher level block for ResNet implementation."""

    def __init__(
        self,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        uniform_init_minval: float,
        uniform_init_maxval: float,
        channels: int,
        num_blocks: int,
        stride: Union[int, Sequence[int]],
        bn_config: Mapping[str, float],
        resnet_v2: bool,
        bottleneck: bool,
        use_projection: bool,
        name: Optional[str] = None,
    ):

        super().__init__(name=name)

        block_cls = BlockV2 if resnet_v2 else BlockV1

        self.blocks = []
        for i in range(num_blocks):
            self.blocks.append(
                block_cls(
                    stochastic_parameters=stochastic_parameters,
                    dropout=dropout,
                    dropout_rate=dropout_rate,
                    uniform_init_minval=uniform_init_minval,
                    uniform_init_maxval=uniform_init_maxval,
                    channels=channels,
                    stride=(1 if i else stride),
                    use_projection=(i == 0 and use_projection),
                    bottleneck=bottleneck,
                    bn_config=bn_config,
                    name="block_%d" % (i),
                )
            )

    def __call__(self, inputs, rng_key, stochastic, is_training, test_local_stats):
        out = inputs
        for block in self.blocks:
            out = block(out, rng_key, stochastic, is_training, test_local_stats)
        return out


def check_length(length, value, name):
    if len(value) != length:
        raise ValueError(f"`{name}` must be of length 4 not {len(value)}")


class ResNet(hk.Module):
    """ResNet model."""

    BlockGroup = BlockGroup  # pylint: disable=invalid-name
    BlockV1 = BlockV1  # pylint: disable=invalid-name
    BlockV2 = BlockV2  # pylint: disable=invalid-name

    def __init__(
        self,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        linear_model: bool,
        blocks_per_group: Sequence[int],
        num_classes: int,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        bottleneck: bool = True,
        channels_per_group: Sequence[int] = (256, 512, 1024, 2048),
        use_projection: Sequence[bool] = (True, True, True, True),
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        uniform_init_minval: float = -20.,
        uniform_init_maxval: float = -18.,
    ):
        """Constructs a ResNet model.
        Args:
        blocks_per_group: A sequence of length 4 that indicates the number of
        blocks created in each group.
        num_classes: The number of classes to classify the inputs into.
        bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
        passed on to the :class:`~haiku.BatchNorm` layers. By default the
        ``decay_rate`` is ``0.9`` and ``eps`` is ``1e-5``.
        resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults to
        ``False``.
        bottleneck: Whether the block should bottleneck or not. Defaults to
        ``True``.
        channels_per_group: A sequence of length 4 that indicates the number
        of channels used for each block in each group.
        use_projection: A sequence of length 4 that indicates whether each
        residual block should use projection.
        logits_config: A dictionary of keyword arguments for the logits layer.
        name: Name of the module.
        """
        super().__init__(name=name)
        self.resnet_v2 = resnet_v2
        self.linear_model = linear_model
        self.dropout = dropout
        self.dropout_rate = dropout_rate

        if self.linear_model:
            self.stochastic_parameters_feature_mapping = False
            self.stochastic_parameters_final_layer = stochastic_parameters
        else:
            self.stochastic_parameters_feature_mapping = stochastic_parameters
            self.stochastic_parameters_final_layer = stochastic_parameters

        # TODO: Maybe remove hardcoding here
        self.uniform_init_minval = uniform_init_minval
        self.uniform_init_maxval = uniform_init_maxval

        bn_config = dict(bn_config or {})
        bn_config.setdefault("decay_rate", 0.9)
        bn_config.setdefault("eps", 1e-5)
        bn_config.setdefault("create_scale", True)
        bn_config.setdefault("create_offset", True)

        logits_config = dict(logits_config or {})
        # logits_config.setdefault("w_init", jnp.zeros)
        logits_config.setdefault("w_init", None)  # TR: added
        logits_config.setdefault("name", "logits")
        logits_config.setdefault("with_bias", True)  # TR: added

        # Number of blocks in each group for ResNet.
        check_length(4, blocks_per_group, "blocks_per_group")
        check_length(4, channels_per_group, "channels_per_group")

        self.initial_conv = conv2D_stochastic(
            output_channels=64,
            kernel_shape=7,
            stride=2,
            with_bias=False,
            padding="VALID",
            name="initial_conv",
            stochastic_parameters=self.stochastic_parameters_feature_mapping,
            uniform_init_minval=self.uniform_init_minval,
            uniform_init_maxval=self.uniform_init_maxval,
        )

        if not self.resnet_v2:
            self.initial_batchnorm = hk.BatchNorm(name="batchnorm", **bn_config)

        self.block_groups = []
        strides = (1, 2, 2, 2)
        for i in range(4):
            self.block_groups.append(
                BlockGroup(
                    stochastic_parameters=self.stochastic_parameters_feature_mapping,
                    dropout=self.dropout,
                    dropout_rate=self.dropout_rate,
                    uniform_init_minval=self.uniform_init_minval,
                    uniform_init_maxval=self.uniform_init_maxval,
                    channels=channels_per_group[i],
                    num_blocks=blocks_per_group[i],
                    stride=strides[i],
                    bn_config=bn_config,
                    resnet_v2=resnet_v2,
                    bottleneck=bottleneck,
                    use_projection=use_projection[i],
                    name="block_group_%d" % (i),
                )
            )

        self.max_pool = hk.MaxPool(
            window_shape=(1, 3, 3, 1), strides=(1, 2, 2, 1), padding="SAME"
        )

        if self.resnet_v2:
            self.final_batchnorm = hk.BatchNorm(name="batchnorm", **bn_config)

        self.logits = dense_stochastic_hk(
            output_size=num_classes,
            uniform_init_minval=self.uniform_init_minval,
            uniform_init_maxval=self.uniform_init_maxval,
            stochastic_parameters=self.stochastic_parameters_final_layer,
            **logits_config,
        )

    def __call__(self, inputs, rng_key, stochastic, is_training):
        # TODO: remove hardcoded values
        test_local_stats = False
        out = inputs
        out = zero_padding_2D(out, padding=3)
        out = self.initial_conv(out, rng_key, stochastic)
        if not self.resnet_v2:
            out = self.initial_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)
            # DROPOUT
            if self.dropout and is_training:
                out = hk.dropout(rng_key, self.dropout_rate, out)
        # TODO: check why the max_pool is commented out
        # Yes, I was following the modifications Joost proposed in his paper here:
        # Uncertainty Estimation Using a Single Deep Deterministic Neural Network
        out = self.max_pool(out)

        for block_group in self.block_groups:
            out = block_group(out, rng_key, stochastic, is_training, test_local_stats)

        if self.resnet_v2:
            out = self.final_batchnorm(out, is_training, test_local_stats)
            out = jax.nn.relu(out)
        out = jnp.mean(out, axis=[1, 2])
        return self.logits(out, rng_key, stochastic)


def zero_padding_2D(x, padding: int):
    # assume x is of shape (batch_size, width, height, channels)
    return jnp.pad(x, pad_width=((0, 0), (padding, padding), (padding, padding), (0, 0)))


class ResNet18(ResNet):
    """ResNet18."""

    def __init__(
        self,
        output_dim: int,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        linear_model: bool = False,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
    ):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(
            stochastic_parameters=stochastic_parameters,
            linear_model=linear_model,
            dropout=dropout,
            dropout_rate=dropout_rate,
            blocks_per_group=(2, 2, 2, 2),
            num_classes=output_dim,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            bottleneck=False,
            channels_per_group=(64, 128, 256, 512),
            use_projection=(False, True, True, True),
            logits_config=logits_config,
            name=name,
        )


class ResNet50FSVI(ResNet):
    """ResNet18."""

    def __init__(
        self,
        output_dim: int,
        stochastic_parameters: bool,
        dropout: bool,
        dropout_rate: float,
        linear_model: bool = False,
        bn_config: Optional[Mapping[str, float]] = None,
        resnet_v2: bool = False,
        logits_config: Optional[Mapping[str, Any]] = None,
        name: Optional[str] = None,
        uniform_init_minval: float = -20.,
        uniform_init_maxval: float = -18.,
    ):
        """Constructs a ResNet model.

        Args:
          num_classes: The number of classes to classify the inputs into.
          bn_config: A dictionary of two elements, ``decay_rate`` and ``eps`` to be
            passed on to the :class:`~haiku.BatchNorm` layers.
          resnet_v2: Whether to use the v1 or v2 ResNet implementation. Defaults
            to ``False``.
          logits_config: A dictionary of keyword arguments for the logits layer.
          name: Name of the module.
        """
        super().__init__(
            stochastic_parameters=stochastic_parameters,
            linear_model=linear_model,
            dropout=dropout,
            dropout_rate=dropout_rate,
            blocks_per_group=(3, 4, 6, 3),
            num_classes=output_dim,
            bn_config=bn_config,
            resnet_v2=resnet_v2,
            bottleneck=True,
            channels_per_group=(256, 512, 1024, 2048),
            use_projection=(True, True, True, True),
            logits_config=logits_config,
            name=name,
            uniform_init_minval=uniform_init_minval,
            uniform_init_maxval=uniform_init_maxval,
        )
