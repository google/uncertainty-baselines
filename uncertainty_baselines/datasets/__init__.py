# coding=utf-8
# Copyright 2020 The Uncertainty Baselines Authors.
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

from uncertainty_baselines.datasets import inception_preprocessing
from uncertainty_baselines.datasets.base import BaseDataset
from uncertainty_baselines.datasets.cifar import Cifar100Dataset
from uncertainty_baselines.datasets.cifar import Cifar10Dataset
from uncertainty_baselines.datasets.clinc_intent import ClincIntentDetectionDataset
from uncertainty_baselines.datasets.criteo import CriteoDataset
from uncertainty_baselines.datasets.datasets import get
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
from uncertainty_baselines.datasets.svhn import SvhnDataset
from uncertainty_baselines.datasets.test_utils import DatasetTest
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsDataset
from uncertainty_baselines.datasets.toxic_comments import CivilCommentsIdentitiesDataset
from uncertainty_baselines.datasets.toxic_comments import WikipediaToxicityDataset
