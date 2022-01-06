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

"""Uncertainty baseline training datasets."""

from uncertainty_baselines.datasets.diabetic_retinopathy_detection import UBDiabeticRetinopathyDetectionDataset
from uncertainty_baselines.datasets.diabetic_retinopathy_severity_shift_mild import DiabeticRetinopathySeverityShiftMildDataset
from uncertainty_baselines.datasets.diabetic_retinopathy_severity_shift_moderate import DiabeticRetinopathySeverityShiftModerateDataset
from uncertainty_baselines.datasets.dialog_state_tracking import MultiWoZSynthDataset
from uncertainty_baselines.datasets.dialog_state_tracking import SGDSynthDataset
from uncertainty_baselines.datasets.dialog_state_tracking import SimDialDataset
from uncertainty_baselines.datasets.drug_cardiotoxicity import DrugCardiotoxicityDataset
from uncertainty_baselines.datasets.fashion_mnist import FashionMnistDataset
from uncertainty_baselines.datasets.genomics_ood import GenomicsOodDataset
from uncertainty_baselines.datasets.glue import ColaDataset
from uncertainty_baselines.datasets.glue import MrpcDataset
from uncertainty_baselines.datasets.glue import QnliDataset
from uncertainty_baselines.datasets.glue import QqpDataset
from uncertainty_baselines.datasets.glue import RteDataset
from uncertainty_baselines.datasets.glue import Sst2Dataset
from uncertainty_baselines.datasets.glue import WnliDataset
from uncertainty_baselines.datasets.imagenet import ImageNetDataset
from uncertainty_baselines.datasets.mnist import MnistDataset
from uncertainty_baselines.datasets.mnli import MnliDataset
from uncertainty_baselines.datasets.movielens import MovieLensDataset
from uncertainty_baselines.datasets.places import Places365Dataset
from uncertainty_baselines.datasets.random import RandomGaussianImageDataset
from uncertainty_baselines.datasets.random import RandomRademacherImageDataset
from uncertainty_baselines.datasets.svhn import SvhnDataset
from uncertainty_baselines.datasets.test_utils import DatasetTest
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsIdentitiesDataset
from uncertainty_baselines.datasets.toxic_comments import WikipediaToxicityDataset
# pylint: enable=g-bad-import-order

try:
  from uncertainty_baselines.datasets.smcalflow import MultiWoZDataset  # pylint: disable=g-import-not-at-top
  from uncertainty_baselines.datasets.smcalflow import SMCalflowDataset  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning(
      'Skipped importing the SMCalflow dataset due to ImportError. Try '
      'installing uncertainty baselines with the `datasets` extras.',
      exc_info=True)

try:
  # Try to import datasets depending on librosa.
  from uncertainty_baselines.datasets.speech_commands import SpeechCommandsDataset  # pylint: disable=g-import-not-at-top
except ImportError:
  logging.warning(
      'Skipped importing the Speech Commands dataset due to ImportError. Try '
      'installing uncertainty baselines with the `datasets` extras.',
      exc_info=True)
except OSError:
  logging.warning(
      'Skipped importing the Speech Commands dataset due to OSError.',
      exc_info=True)
