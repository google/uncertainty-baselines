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

"""Torch utilities."""

import torch


def count_parameters(model):
  r"""Count the number of parameters in the model.

  Due to Federico Baldassarre
  https://discuss.pytorch.org/t/how-do-i-check-the-number-of-
  parameters-of-a-model/4325/7

  Counts number of trainable model parameters.

  Args:
    model: the model to count the parameters of.

  Returns:
    The number of models in the parameter.
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_torch_model(model, optimizer, epoch, checkpoint_path):
  checkpoint_dict = {
      'epoch': epoch,
      'model_state_dict': model.state_dict(),
      'optimizer_state_dict': optimizer.state_dict()
  }
  torch.save(checkpoint_dict, checkpoint_path)
