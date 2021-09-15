"""
Jax with TPUs.

Make sure you have the appropriate version with TPU lib.

pip install "jax[tpu]>=0.2.16" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
https://cloud.google.com/tpu/docs/jax-quickstart-tpu-vm
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '0'

from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
import jax
import jax.numpy as jnp
import flax
import tqdm
import uncertainty_baselines as ub

from baselines.jft.experiments import imagenet21k_vit_base16_finetune_cifar10
from baselines.jft import checkpoint_utils

print(tf.config.experimental.get_visible_devices())
print(jax.local_devices()) # If this fails, see https://github.com/google/jax#pip-installation-gpu-cuda

config = imagenet21k_vit_base16_finetune_cifar10.get_config()
print(config)

checkpoint_path = "gs://ub-data/ImageNet21k_ViT-B16_ImagetNet21k_ViT-B_16_28592399.npz"

use_tpu = True

# Follows: https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/deterministic.py

batch_size = 8

seed = config.get('seed', 0)
rng = jax.random.PRNGKey(seed)

model = ub.models.vision_transformer(
      num_classes=config.num_classes, **config.get('model', {}))

if use_tpu:
  # * Load Pretrained From UB *
  from baselines.jft.experiments import imagenet21k_vit_base16_finetune_cifar10

  # Load a sensible config.
  model_config = imagenet21k_vit_base16_finetune_cifar10.get_config()

  image_size = (384, 384, 3)
  dummy_input = jnp.zeros((batch_size,) + image_size, jnp.float32)

  # Initialize random parameters.
  # This also compiles the model to XLA (takes some minutes the first time).
  variables = jax.jit(lambda: model.init(
    jax.random.PRNGKey(0),
    # Discard the "num_local_devices" dimension of the batch for initialization.
    # batch['image'][0, :1],
    dummy_input,
    train=False,
  ), backend='cpu')()

  # Load and convert pretrained checkpoint.
  # This involves loading the actual pre-trained model results, but then also also
  # modifying the parameters a bit, e.g. changing the final layers, and resizing
  # the positional embeddings.
  # For details, refer to the code and to the methods of the paper.
  params = checkpoint_utils.load_from_pretrained_checkpoint(
    variables['params'], checkpoint_path,
    model_config.model.representation_size,
    model_config.model.classifier,
    model_config.model.get('reinit_params', ('head/kernel', 'head/bias'))
  )

  print('Successfully loaded parameters.')
  # * Evaluate *

  # So far, all our data is in the host memory. Let's now replicate the arrays
  # into the devices.
  # This will make every array in the pytree params become a ShardedDeviceArray
  # that has the same data replicated across all local devices.
  # For TPU it replicates the params in every core.
  # For a single GPU this simply moves the data onto the device.
  # For CPU it simply creates a copy.
  params_repl = flax.jax_utils.replicate(params)
  print('params.cls:', type(params['head']['bias']).__name__,
        params['head']['bias'].shape)
  print('params_repl.cls:', type(params_repl['head']['bias']).__name__,
        params_repl['head']['bias'].shape)

  # Then map the call to our model's forward pass onto all available devices.

  # Andreas: changed to return [0]: (logits, out)[0]
  vit_apply_repl = jax.pmap(lambda params, inputs: model.apply(
    dict(params=params), inputs, train=False)[0])


else:
  @partial(jax.jit, backend='cpu')
  def init(rng):
  #     image_size = tuple(train_ds.element_spec['image'].shape[1:])
      image_size = (384, 384, 3)
      dummy_input = jnp.zeros((batch_size,) + image_size, jnp.float32)
      params = flax.core.unfreeze(model.init(rng, dummy_input,
                                             train=False))['params']

      # Set bias in the head to a low value, such that loss is small initially.
      params['head']['bias'] = jnp.full_like(
          params['head']['bias'], config.get('init_head_bias', 0))

      # init head kernel to all zeros for fine-tuning
      if config.get('model_init'):
          params['head']['kernel'] = jnp.full_like(params['head']['kernel'], 0)

      return params

  rng, rng_init = jax.random.split(rng)
  params_cpu = init(rng_init)

  params = checkpoint_utils.load_from_pretrained_checkpoint(
      params_cpu, checkpoint_path, config.model.representation_size,
      config.model.classifier,
      config.model.get('reinit_params', ('head/kernel', 'head/bias'))
  )

  params_repl = flax.jax_utils.replicate(params)

print('loaded ViT model')

# * CIFAR-10 PREPROCESSING! *

# TODO: use `pp_builder` once it's open sourced in UB
# preprocess_fn = pp_builder.get_preprocess_fn(config.pp_train)


image_size = 384 # from the config

# From: https://github.com/google-research/vision_transformer/blob/main/vit_jax/input_pipeline.py#L195

def get_pp(mode):
    def _pp(data):
    #     im = image_decoder(data['image'])
        im = data['image'] # CIFAR-10 is already decoded

        if im.shape[-1] == 1:
            im = tf.repeat(im, 3, axis=-1)
        if mode == 'train':
            channels = im.shape[-1]
            begin, size, _ = tf.image.sample_distorted_bounding_box(
                tf.shape(im),
                tf.zeros([0, 0, 4], tf.float32),
                area_range=(0.05, 1.0),
                min_object_covered=0,  # Don't enforce a minimum area.
                use_image_if_no_bounding_boxes=True)
            im = tf.slice(im, begin, size)
            # Unfortunately, the above operation loses the depth-dimension. So we
            # need to restore it the manual way.
            im.set_shape([None, None, channels])
            im = tf.image.resize(im, [image_size, image_size])
            if tf.random.uniform(shape=[]) > 0.5:
                im = tf.image.flip_left_right(im)
        else:
            im = tf.image.resize(im, [image_size, image_size])
        im = (im - 127.5) / 127.5
        label = tf.one_hot(data['label'], 10)
        return {'image': im, 'label': label}
    return _pp


data_builder = tfds.builder("cifar10")
ds_info = data_builder.info
data_builder.download_and_prepare()

train_ds = data_builder.as_dataset(split='train', shuffle_files=False)
train_ds = train_ds.map(get_pp("train"), tf.data.experimental.AUTOTUNE)
train_ds = train_ds.batch(batch_size, drop_remainder=True)

test_ds = data_builder.as_dataset(split='test', batch_size=-1)


if use_tpu:
  print('Getting accuracy on TPU ViT.')
  def get_accuracy(params_repl, dataset, split):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = ds_info.splits[split].num_examples // batch_size
    # steps = input_pipeline.get_dataset_info(dataset, 'test')[
    #           'num_examples'] // batch_size
    for _, batch in zip(tqdm.trange(steps), dataset.as_numpy_iterator()):
      predicted = vit_apply_repl(params_repl, batch['image'])

      # predicted
      is_same = predicted.argmax(axis=-1) == batch['label'].argmax(axis=-1)
      good += is_same.sum()
      total += len(is_same.flatten())
    return good / total

  # Random performance without fine-tuning.
  get_accuracy(params_repl, test_ds, 'test')
else:
  print('Getting accuracy on non-TPU ViT.')
  # * Evaluation Fn *
  # From: https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/deterministic.py

  # @partial(jax.pmap, axis_name='batch')
  def evaluation_fn(params, images, labels):
      logits, out = model.apply({'params': flax.core.freeze(params)},
                                images,
                                train=False)

      # "u" is also not open-sourced yet
  #     losses = getattr(u, config.get('loss', 'sigmoid_xent'))(
  #         logits=logits, labels=labels, reduction=False)

      top1_idx = jnp.argmax(logits, axis=1)
      # Extracts the label at the highest logit index for each image.
      top1_correct = jnp.sum(jnp.take_along_axis(labels, top1_idx[:, None], axis=1)[:, 0])

      return top1_correct


  for batch in train_ds.as_numpy_iterator():
      print(evaluation_fn(params, batch['image'], batch['label']))
