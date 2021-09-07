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

r"""Speech commands covariate shift benchmark.

This dataset consists of labeled audio samples. It is based on the
publicly avilable Speech Commands dataset:
https://www.tensorflow.org/datasets/catalog/speech_commands

Reference:
* Warden P (2018) "Speech Commands: A Dataset for Limited-Vocabulary Speech
  Recognition" https://arxiv.org/abs/1804.03209
"""

from typing import Callable, Dict, Optional, Tuple, Union

import librosa
import numpy as np
from scipy import signal
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

from uncertainty_baselines.datasets import base

SAMPLE_RATE_HZ = 16000

WHITE_NOISE = 'white_noise'
PITCH_SHIFT = 'pitch_shift'
LOW_PASS = 'low_pass'
ROOM_REVERB = 'room_reverb'

SEMANTIC_SHIFT = 'semantic_shift'
IN_DISTRIBUTION_MAX_LABEL = 10
SEMANTIC_SHFIT_LABEL = 11


def mix_white_noise(audio, noise_level_db):
  _, variance = tf.nn.moments(audio, axes=[0])
  audio_rms = tf.math.sqrt(variance)
  noise_rms = tf.pow(10.0, noise_level_db / 10.0) * audio_rms
  noise = tf.random.normal(
      tf.shape(audio), mean=0.0, stddev=noise_rms, dtype=tf.float32)
  return audio + noise


def pitch_shift(audio, semitones):

  def librosa_pitch_shift(x, semitones):
    return librosa.effects.pitch_shift(
        x.numpy(), SAMPLE_RATE_HZ, n_steps=semitones)

  return tf.py_function(
      func=librosa_pitch_shift, inp=[audio, semitones], Tout=tf.float32)


IIR_FILTER_ORDER = 8
# Speech-commands dataset has a fixed sample rate of 16 kHz. So the Nyquist
# frequency is 8 kHz.
b_4khz, a_4khz = signal.butter(IIR_FILTER_ORDER, 0.5)
b_2khz, a_2khz = signal.butter(IIR_FILTER_ORDER, 0.25)
b_1khz, a_1khz = signal.butter(IIR_FILTER_ORDER, 0.125)


def low_pass_filter(audio, cutoff_frequency_khz):
  """Perform low-pass filtering at given cutoff frequency."""

  def scipy_filt(x, cutoff_frequency_khz):
    if cutoff_frequency_khz == 4:
      b, a = b_4khz, a_4khz
    elif cutoff_frequency_khz == 2:
      b, a = b_2khz, a_2khz
    elif cutoff_frequency_khz == 1:
      b, a = b_1khz, a_1khz
    else:
      raise ValueError(
          'Unsupported cutoff frequency (kHz): %s (Supported values: 1, 2, 4)' %
          cutoff_frequency_khz)
    return signal.filtfilt(b, a, x)

  return tf.py_function(
      func=scipy_filt, inp=[audio, cutoff_frequency_khz], Tout=tf.float32)


def add_room_reverb(audio, room_size):
  """Add simulated room reverberation to audio."""

  def pra_add_room_reverb(x, room_size):
    if room_size == 6:
      return signal.convolve(x, rir_6m)[:len(x)]
    elif room_size == 12:
      return signal.convolve(x, rir_12m)[:len(x)]
    else:
      raise ValueError(
          'Unsupported room size (m): %s (Supported value: 6, 12)' % room_size)

  return tf.py_function(
      func=pra_add_room_reverb, inp=[audio, room_size], Tout=tf.float32)


class _SpeechCommandsDatasetBuilder(tfds.core.DatasetBuilder):
  """Minimal TFDS DatasetBuilder for the Speech Commands dataset."""
  VERSION = tfds.core.Version('0.0.0')

  def __init__(
      self,
      tfds_dataset_builder: tfds.core.DatasetBuilder,
      label_filter_fn: Callable[[Dict[str, tf.Tensor]], bool],
      **kwargs):
    self._tfds_dataset_builder = tfds_dataset_builder
    self._label_filter_fn = label_filter_fn
    super().__init__(**kwargs)

  def _download_and_prepare(self, dl_manager, download_config=None):
    """Downloads and prepares dataset for reading."""
    return self._tfds_dataset_builder._download_and_prepare(  # pylint: disable=protected-access
        dl_manager, download_config)

  def _as_dataset(
      self,
      split: tfds.Split,
      decoders=None,
      read_config=None,
      shuffle_files=False) -> tf.data.Dataset:
    raise NotImplementedError

  # Note that we override `as_dataset` instead of `_as_dataset` to avoid any
  # `data_dir` reading logic.
  def as_dataset(
      self,
      split: tfds.Split,
      *,
      batch_size=None,
      decoders=None,
      read_config=None,
      shuffle_files=False,
      as_supervised=False) -> tf.data.Dataset:
    """Constructs a `tf.data.Dataset`, see parent class for documentation."""
    dataset = self._tfds_dataset_builder.as_dataset(
        split=split,
        batch_size=batch_size,
        decoders=decoders,
        read_config=read_config,
        shuffle_files=shuffle_files,
        as_supervised=as_supervised)
    return dataset.filter(self._label_filter_fn)

  def _info(self) -> tfds.core.DatasetInfo:
    raise NotImplementedError

  # Note that we are overriding info instead of _info() so that we properly
  # generate the full DatasetInfo.
  @property
  def info(self) -> tfds.core.DatasetInfo:
    """Returns the `tfds.core.DatasetInfo` object."""
    return self._tfds_dataset_builder.info


class SpeechCommandsDataset(base.BaseDataset):
  """A covariate/semantic shift benchmark dataset based on Speech Commmands.

  This shift benchmark is a based on the Speech Commands dataset:
    https://www.tensorflow.org/datasets/catalog/speech_commands

  This benchmark dataset supports two types of shifts.

  1. Covariate shifts, with audio features being perturbed versions of the
  original audio, and without any novel labels. In this category, there are
  the following types of perturbations, which can be specified through the
  `split` argument of the `build()` call.

    - split=('white_noise', noise_level_db): adding white noise to the original
      audio. `noise_level_db` is the level of the noise, in decibels relative
      to the level of the original audio.
    - split=('pitch_shift', semitones): spectral (pitch) shift, the amount of
      which is specified by `semitones`, which can be a negative or positive
      float number. This is based on `librosa.effects.pitch_shift()`.
    - split=('low_pass', cutoff_khz): performing low-pass filtering on the
      audio, with the cutoff frequency (in kHz) being `cutoff_khz`. The audio
      has a fixed sample rate of 16 kHz. The supported values of the
      `cutoff_khz` are: 4, 2, and 1.
    - split=('room_reverb', room_size): simulating room reverberation through
      convolution with a room impulse response inside a "shoebox" room of
      `room_size` meters in size. The supported values of `room_size` are:
      6 and 12.

  2. Semantic shift, with novel labels (words), not seen in the train
     and validation splits. This can be invoked with
     split=('semantic_shift',) during `build()` calls.
  """

  AUDIO_LENGTH = 16000

  def __init__(
      self,
      # TODO(cais): Maybe tighter typing for (str, float).
      split: Union[Tuple[str, float], str],
      shuffle_buffer_size: Optional[int] = None,
      num_parallel_parser_calls: int = 64,
      try_gcs: bool = False,
      download_data: bool = False,
      is_training: Optional[bool] = None):
    """Create a Speech commands tf.data.Dataset builder.

    Args:
      split: a dataset split, either a custom tfds.Split or one of the
        tfds.Split enums [TRAIN, VALIDAITON, TEST] or their lowercase string
        names.
      shuffle_buffer_size: the number of example to use in the shuffle buffer
        for tf.data.Dataset.shuffle().
      num_parallel_parser_calls: the number of parallel threads to use while
        preprocessing in tf.data.Dataset.map().
      try_gcs: Whether or not to try to use the GCS stored versions of dataset
        files. Currently unsupported.
      download_data: Whether or not to download data before loading. Currently
        unsupported.
      is_training: Whether or not the given `split` is the training split. Only
        required when the passed split is not one of ['train', 'validation',
        'test', tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST].
    """
    name = 'speech_commands'
    tfds_dataset_builder = tfds.builder(name, try_gcs=try_gcs)
    self._original_split = split
    tfds_split = split
    if split == (SEMANTIC_SHIFT,):
      tfds_split = 'test'
      label_filter_fn = lambda example: example['label'] == SEMANTIC_SHFIT_LABEL
    else:
      label_filter_fn = (
          lambda example: example['label'] <= IN_DISTRIBUTION_MAX_LABEL)

    if split not in [tfds.Split.TRAIN, tfds.Split.VALIDATION, tfds.Split.TEST]:
      tfds_split = tfds.Split.TEST

    dataset_builder = _SpeechCommandsDatasetBuilder(
        tfds_dataset_builder=tfds_dataset_builder,
        label_filter_fn=label_filter_fn)
    super().__init__(
        name=name,
        dataset_builder=dataset_builder,
        split=tfds_split,
        is_training=is_training,
        shuffle_buffer_size=shuffle_buffer_size,
        num_parallel_parser_calls=num_parallel_parser_calls,
        download_data=download_data)

  def _create_process_example_fn(self) -> base.PreProcessFn:

    def _example_parser(example: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
      audio = tf.cast(example['audio'], dtype=tf.float32)
      audio_length = tf.size(audio)
      if audio_length < self.AUDIO_LENGTH:
        audio = tf.concat([
            audio,
            tf.zeros(self.AUDIO_LENGTH - tf.size(audio), dtype=tf.float32)
        ],
                          axis=-1)
      elif audio_length > self.AUDIO_LENGTH:
        audio = tf.slice(audio, [0], [self.AUDIO_LENGTH])

      if isinstance(self._original_split, tuple):
        split_name = self._original_split[0]
        if split_name == WHITE_NOISE:
          noise_level_db = float(self._original_split[1])
          audio = mix_white_noise(audio, noise_level_db)
        elif split_name == PITCH_SHIFT:
          semitones = float(self._original_split[1])
          audio = pitch_shift(audio, semitones)
        elif split_name == LOW_PASS:
          cutoff_frequency_khz = float(self._original_split[1])
          audio = low_pass_filter(audio, cutoff_frequency_khz)
        elif split_name == ROOM_REVERB:
          room_size = float(self._original_split[1])
          audio = add_room_reverb(audio, room_size)
        elif split_name == SEMANTIC_SHIFT:
          pass
        else:
          raise ValueError(
              'Unrecognized shift split: {}'.format(self._original_split[0]))

      label = tf.cast(example['label'], tf.int32)
      parsed_example = {
          'features': audio,
          'labels': label,
      }
      return parsed_example

    return _example_parser


# Room impulse responses.
# Example code to generate these constant array impulse responses with:
# room_6m = pra.ShoeBox((6, 6, 6), fs=16000)
# room_6m.add_source((1, 1, 1))
# room_6m.add_microphone((3, 3, 3))
# room_6m.compute_rir()
# rir_6m = room_6m.rir[0][0]
rir_6m = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -3.42655409e-06, 1.40397144e-05,
    -3.23464140e-05, 5.88638497e-05, -9.41216445e-05, 1.38664674e-04,
    -1.93056691e-04, 2.57884907e-04, -3.33765739e-04, 4.21351967e-04,
    -5.21341628e-04, 6.34489047e-04, -7.61618535e-04, 9.03641428e-04,
    -1.06157736e-03, 1.23658093e-03, -1.42997539e-03, 1.64329541e-03,
    -1.87834190e-03, 2.13725290e-03, -2.42259633e-03, 2.73749260e-03,
    -3.08577920e-03, 3.47223458e-03, -3.90288827e-03, 4.38545825e-03,
    -4.92998118e-03, 5.54974277e-03, -6.26268922e-03, 7.09363795e-03,
    -8.07787071e-03, 9.26723564e-03, -1.07410721e-02, 1.26270804e-02,
    -1.51445908e-02, 1.87043900e-02, -2.41753856e-02, 3.37690552e-02,
    -5.52621319e-02, 1.49235490e-01, 2.15035990e-01, -6.20688346e-02,
    3.60249530e-02, -2.51831043e-02, 1.92005730e-02, -1.53836541e-02,
    1.27200685e-02, -1.07442389e-02, 9.21221783e-03, -7.98390668e-03,
    6.97316489e-03, -6.12415563e-03, 5.39912788e-03, -4.77166504e-03,
    4.22274434e-03, -3.73833024e-03, 3.30784757e-03, -2.92318059e-03,
    2.57799817e-03, -2.26728741e-03, 1.98702432e-03, -1.73393671e-03,
    1.50533062e-03, -1.29896122e-03, 1.11293555e-03, -9.45638301e-04,
    7.95674584e-04, -6.69382350e-04, 5.73977282e-04, -5.09614774e-04,
    4.76564526e-04, -4.75197793e-04, 5.05979603e-04, -5.69465508e-04,
    6.66302704e-04, -7.97235524e-04, 9.63115557e-04, -1.16491682e-03,
    1.40375671e-03, -1.68092371e-03, 1.99455616e-03, -2.34338973e-03,
    2.72999368e-03, -3.15731423e-03, 3.62877031e-03, -4.14837627e-03,
    4.72090080e-03, -5.35207479e-03, 6.04886668e-03, -6.81985203e-03,
    7.67571725e-03, -8.62995774e-03, 9.69986409e-03, -1.09079454e-02,
    1.22840336e-02, -1.38684831e-02, 1.57171924e-02, -1.79097900e-02,
    2.05635792e-02, -2.38586032e-02, 2.80857621e-02, -3.37471974e-02,
    4.17898371e-02, -5.42366910e-02, 7.63195787e-02, -1.27017118e-01,
    3.70874381e-01, 4.07325744e-01, -1.30488311e-01, 7.71800784e-02,
    -5.43815962e-02, 4.16437379e-02, -3.34574381e-02, 2.77171567e-02,
    -2.34443965e-02, 2.01228137e-02, -1.74543402e-02, 1.52549857e-02,
    -1.34051174e-02, 1.18236504e-02, -1.04537281e-02, 9.25433744e-03,
    -8.19517382e-03, 7.25337534e-03, -6.41137797e-03, 5.65546672e-03,
    -4.97477300e-03, 4.36056547e-03, -3.80573904e-03, 3.30444023e-03,
    -2.85178837e-03, 2.44366529e-03, -2.07655472e-03, 1.74741840e-03,
    -1.45359954e-03, 1.19274696e-03, -9.62755091e-04, 7.61716258e-04,
    -5.87882461e-04, 4.39634769e-04, -3.15458688e-04, 2.13924355e-04,
    -1.33670628e-04, 7.33923479e-05, -3.18301955e-05, 7.76269681e-06, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -2.38747338e-06, 9.78084796e-06,
    -2.25308894e-05, 4.09950309e-05, -6.55388060e-05, 9.65377286e-05,
    -1.34379712e-04, 1.79468131e-04, -2.32225669e-04, 2.93099104e-04,
    -3.62565268e-04, 4.41138428e-04, -5.29379443e-04, 6.27907151e-04,
    -7.37412564e-04, 8.58676648e-04, -9.92592721e-04, 1.14019486e-03,
    -1.30269424e-03, 1.48152599e-03, -1.67841036e-03, 1.89543337e-03,
    -2.13515466e-03, 2.40075373e-03, -2.69623175e-03, 3.02669482e-03,
    -3.39876026e-03, 3.82115283e-03, -4.30560367e-03, 4.86824762e-03,
    -5.53187459e-03, 6.32971336e-03, -7.31212081e-03, 8.55916230e-03,
    -1.02061784e-02, 1.25012298e-02, -1.59529621e-02, 2.17940034e-02,
    -3.39792070e-02, 7.59668925e-02, 3.33167613e-01, -5.16211323e-02,
    2.78021485e-02, -1.88732096e-02, 1.41659696e-02, -1.12397735e-02,
    9.23205932e-03, -7.76046629e-03, 6.62956316e-03, -5.72910917e-03,
    4.99223627e-03, -4.37605879e-03, 3.85182890e-03, -3.39957076e-03,
    3.00498365e-03, -2.65756834e-03, 2.34944770e-03, -2.07459889e-03,
    1.82833875e-03, -1.60696993e-03, 1.40753227e-03, -1.22762443e-03,
    1.06527387e-03, -9.18840394e-04, 7.86943755e-04, -6.68408515e-04,
    5.62221654e-04, -4.67499620e-04, 3.83462517e-04, -3.09413730e-04,
    2.44723730e-04, -1.88817158e-04, 1.41162467e-04, -1.01263604e-04,
    6.86533248e-05, -4.28878218e-05, 2.35424214e-05, -1.02081544e-05,
    2.48904614e-06, 0, 0
])
rir_12m = np.array([
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, -1.02631561e-07, 4.20417067e-07, -9.68370528e-07,
    1.76177987e-06, -2.81626657e-06, 4.14786421e-06, -5.77311964e-06,
    7.70922130e-06, -9.97416034e-06, 1.25869316e-05, -1.55677830e-05,
    1.89385256e-05, -2.27229167e-05, 2.69471374e-05, -3.16403864e-05,
    3.68356234e-05, -4.25705052e-05, 4.88885713e-05, -5.58407574e-05,
    6.34873472e-05, -7.19005124e-05, 8.11676589e-05, -9.13958911e-05,
    1.02718056e-04, -1.15301055e-04, 1.29357491e-04, -1.45162317e-04,
    1.63077184e-04, -1.83587008e-04, 2.07356558e-04, -2.35321148e-04,
    2.68838101e-04, -3.09952507e-04, 3.61892373e-04, -4.30063036e-04,
    5.24246885e-04, -6.64144351e-04, 8.96184611e-04, -1.36206139e-03,
    2.80012615e-03, 1.15072997e-01, -2.51785200e-03, 1.27969277e-03,
    -8.50826911e-04, 6.31821003e-04, -4.98049899e-04, 4.07292964e-04,
    -3.41290923e-04, 2.90862785e-04, -2.50889979e-04, 2.18294999e-04,
    -1.91117605e-04, 1.68050981e-04, -1.48191149e-04, 1.30893368e-04,
    -1.15685784e-04, 1.02215342e-04, -9.02127352e-05, 7.94689874e-05,
    -6.98194185e-05, 6.11324002e-05, -5.48429333e-05, 5.25571146e-05,
    -5.44336715e-05, 6.06481930e-05, -7.13917386e-05, 8.68703392e-05,
    -1.07305305e-04, 1.32934320e-04, -1.64013337e-04, 2.00819357e-04,
    -2.43654189e-04, 2.92849389e-04, -3.48772584e-04, 4.11835536e-04,
    -4.82504353e-04, 5.61312457e-04, -6.48877082e-04, 7.45920402e-04,
    -8.53296782e-04, 9.71925877e-04, -1.10295227e-03, 1.24790448e-03,
    -1.40867121e-03, 1.58761649e-03, -1.78774050e-03, 2.01290854e-03,
    -2.26818441e-03, 2.56032852e-03, -2.89856409e-03, 3.29579665e-03,
    -3.77063393e-03, 4.35089471e-03, -5.08006684e-03, 6.03007790e-03,
    -7.32897972e-03, 9.22877921e-03, -1.23029904e-02, 1.81988009e-02,
    -3.43591215e-02, 2.92416083e-01, 4.53664412e-02, -2.08959049e-02,
    1.34891000e-02, -9.88810347e-03, 7.74487914e-03, -6.31389140e-03,
    5.28432173e-03, -4.50358025e-03, 3.88802474e-03, -3.38799991e-03,
    2.97218879e-03, -2.61987936e-03, 2.31683855e-03, -2.05297195e-03,
    1.82092817e-03, -1.61523098e-03, 1.43172010e-03, -1.26717976e-03,
    1.11908539e-03, -9.85426674e-04, 8.64581570e-04, -7.55224800e-04,
    6.56260414e-04, -5.66771297e-04, 4.85980903e-04, -4.13223902e-04,
    3.47923440e-04, -2.89573357e-04, 2.37724170e-04, -1.91971964e-04,
    1.51949518e-04, -1.17319200e-04, 8.77672616e-05, -6.29992410e-05,
    4.27362663e-05, -2.67120863e-05, 1.46706950e-05, -6.36444438e-06,
    1.55256154e-06, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -8.14045566e-07,
    3.33482131e-06, -7.68173873e-06, 1.39764458e-05, -2.23433098e-05,
    3.29100517e-05, -4.58085628e-05, 6.11759391e-05, -7.91557804e-05,
    9.98998088e-05, -1.23569881e-04, 1.50340482e-04, -1.80401818e-04,
    2.13963667e-04, -2.51260164e-04, 2.92555807e-04, -3.38153018e-04,
    3.88401716e-04, -4.43711575e-04, 5.04567812e-04, -5.71551781e-04,
    6.45368117e-04, -7.26881005e-04, 8.17163336e-04, -9.17564439e-04,
    1.02980511e-03, -1.15611374e-03, 1.29942580e-03, -1.46368426e-03,
    1.65430568e-03, -1.87892983e-03, 2.14867658e-03, -2.48036175e-03,
    2.90065012e-03, -3.45445870e-03, 4.22372938e-03, -5.37536533e-03,
    7.30966286e-03, -1.12870931e-02, 2.44211607e-02, 1.57823163e-01,
    -1.84031593e-02, 9.71493353e-03, -6.54653603e-03, 4.89499637e-03,
    -3.87477969e-03, 3.17761594e-03, -2.66806097e-03, 2.27729466e-03,
    -1.96666031e-03, 1.71278483e-03, -1.50071519e-03, 1.32044779e-03,
    -1.16504284e-03, 1.02953888e-03, -9.10297536e-04, 8.04591759e-04,
    -7.10338248e-04, 6.25918303e-04, -5.50054829e-04, 4.81725970e-04,
    -4.20103244e-04, 3.64506449e-04, -3.14370274e-04, 2.69219220e-04,
    -2.28648532e-04, 1.92309542e-04, -1.59898280e-04, 1.31146558e-04,
    -1.05814934e-04, 8.36871230e-05, -6.45655380e-05, 4.82677178e-05,
    -3.46234569e-05, 2.34724997e-05, -1.46626874e-05, 8.04847385e-06,
    -3.48974253e-06, 8.50870964e-07, 0, 0
])
