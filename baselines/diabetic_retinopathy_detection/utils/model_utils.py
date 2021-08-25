import logging
import os

import tensorflow as tf


"""Model initialization and checkpointing utils."""


# Model initialization.

def log_model_init_info(model):
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())


def load_input_shape(dataset_train: tf.data.Dataset):
  """Retrieve size of input to model using Shape tuple access.

  Depends on the number of distributed devices.

  Args:
    dataset_train: training dataset.

  Returns:
    list, input shape of model
  """
  try:
    shape_tuple = dataset_train.element_spec['features'].shape
  except AttributeError:  # Multiple TensorSpec in a (nested) PerReplicaSpec.
    tensor_spec_list = dataset_train.element_spec[  # pylint: disable=protected-access
        'features']._flat_tensor_specs
    shape_tuple = tensor_spec_list[0].shape

  return shape_tuple.as_list()[1:]


# Checkpoint write/load.


# TODO(nband): debug checkpoint issue with retinopathy models
#   (appears distribution strategy-related)
#   For now, we just reload from keras.models (and only use for inference)
#   using the method below (parse_keras_models)
def parse_checkpoint_dir(checkpoint_dir):
  """Parse directory of checkpoints.

  Intended for use with Deep Ensembles and ensembles of MC Dropout models.
  Currently not used, as per above bug.

  Args:
    checkpoint_dir: checkpoint dir.
  Returns:
    paths of checkpoints
  """
  paths = []
  subdirectories = tf.io.gfile.glob(checkpoint_dir)
  is_checkpoint = lambda f: ('checkpoint' in f and '.index' in f)
  for subdir in subdirectories:
    for path, _, files in tf.io.gfile.walk(subdir):
      if any(f for f in files if is_checkpoint(f)):
        latest_checkpoint_without_suffix = tf.train.latest_checkpoint(path)
        paths.append(os.path.join(path, latest_checkpoint_without_suffix))
        break

  return paths


def parse_keras_models(checkpoint_dir):
  """Parse directory of saved Keras models.

  Used for Deep Ensembles and ensembles of MC Dropout models.

  Args:
    checkpoint_dir: checkpoint dir.
  Returns:
    paths of saved Keras models
  """
  paths = []
  is_keras_model_dir = lambda dir_name: ('keras_model' in dir_name)
  for dir_name in tf.io.gfile.listdir(checkpoint_dir):
    dir_path = os.path.join(checkpoint_dir, dir_name)
    if tf.io.gfile.isdir(dir_path) and is_keras_model_dir(dir_name):
      paths.append(dir_path)

  return paths


def get_latest_checkpoint(file_names, return_epoch=False):
  """Get latest checkpoint from list of file names.

  Only necessary
  if manually saving/loading Keras models, i.e., not using the
  tf.train.Checkpoint API.

  Args:
    file_names: List[str], file names located with the `parse_keras_models`
      method.

  Returns:
    str, the file name with the most recent checkpoint or
    Tuple[int, str], the epoch and file name of the most recent checkpoint
  """
  if not file_names:
    return None

  checkpoint_epoch_and_file_name = []
  for file_name in file_names:
    try:
      checkpoint_epoch = file_name.split('/')[-2].split('_')[-1]
    except ValueError:
      raise Exception('Expected Keras checkpoint directory path of format '
                      'gs://path_to_checkpoint/keras_model_{checkpoint_epoch}/')
    checkpoint_epoch = int(checkpoint_epoch)
    checkpoint_epoch_and_file_name.append((checkpoint_epoch, file_name))

  checkpoint_epoch_and_file_name = sorted(
    checkpoint_epoch_and_file_name, reverse=True)

  most_recent_checkpoint_epoch_and_file_name = checkpoint_epoch_and_file_name[0]
  if return_epoch:
    return most_recent_checkpoint_epoch_and_file_name
  else:
    return most_recent_checkpoint_epoch_and_file_name[1]


def load_keras_model(checkpoint):
  model = tf.keras.models.load_model(checkpoint, compile=False)
  logging.info('Successfully loaded model from checkpoint %s.', checkpoint)
  logging.info('Model input shape: %s', model.input_shape)
  logging.info('Model output shape: %s', model.output_shape)
  logging.info('Model number of weights: %s', model.count_params())
  return model


def load_all_keras_checkpoints(checkpoint_dir):
  checkpoint_filenames = parse_keras_models(checkpoint_dir)
  if not checkpoint_filenames:
    raise Exception(
      f'Did not locate a Keras checkpoint in checkpoint directory '
      f'{checkpoint_dir}')

  checkpoint_epoch_and_file_name = []
  for i, file_name in enumerate(checkpoint_filenames):
    try:
      checkpoint_epoch = file_name.split('/')[-2].split('_')[-1]
    except ValueError:
      raise Exception('Expected Keras checkpoint directory path of format '
                      'gs://path_to_checkpoint/keras_model_{checkpoint_epoch}/')
    checkpoint_epoch = int(checkpoint_epoch)
    checkpoint_epoch_and_file_name.append((checkpoint_epoch, file_name))
    if i > 5:
      break

  checkpoint_epoch_and_file_name = list(sorted(checkpoint_epoch_and_file_name))

  for epoch, checkpoint_file in checkpoint_epoch_and_file_name:
    yield epoch, load_keras_model(checkpoint=checkpoint_file)


def load_keras_checkpoints(
    checkpoint_dir, load_ensemble=False, return_epoch=True):
  """Main checkpoint loading function.

  When not loading an ensemble, defaults to also return the epoch
    corresponding to the latest checkpoint.
  """
  # TODO(nband): debug, switch from keras.models.save to tf.train.Checkpoint
  checkpoint_filenames = parse_keras_models(checkpoint_dir)
  if not checkpoint_filenames:
    raise Exception(
      f'Did not locate a Keras checkpoint in checkpoint directory '
      f'{checkpoint_dir}')

  if load_ensemble:
    model = []
    for checkpoint_file in checkpoint_filenames:
      model.append(load_keras_model(checkpoint=checkpoint_file))
  else:
    if len(checkpoint_filenames) == 1 and not return_epoch:
      return load_keras_model(checkpoint_filenames[0])
    latest_checkpoint = get_latest_checkpoint(
      file_names=checkpoint_filenames, return_epoch=return_epoch)
    if return_epoch:
      epoch, latest_checkpoint = latest_checkpoint
      model = (epoch, load_keras_model(checkpoint=latest_checkpoint))
    else:
      model = load_keras_model(checkpoint=latest_checkpoint)

  return model
