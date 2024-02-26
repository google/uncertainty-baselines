# coding=utf-8
# Copyright 2024 The Uncertainty Baselines Authors.
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

"""Common configs for CLIP.

Adapted from
https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/clip/model.py.
"""

# Taken from: https://github.com/openai/CLIP/blob/main/clip/clip.py.
CLIP_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

IMAGE_RESOLUTION = {
    'resnet_50': 224,
    'resnet_101': 224,
    'resnet_50x4': 288,
    'resnet_50x16': 384,
    'resnet_50x64': 448,
    'vit_b32': 224,
    'vit_b16': 224,
    'vit_l14': 224,
    'vit_l14_336px': 336,
}

CONFIGS = {
    'vit_b32':
        dict(
            embed_dim=512,
            vocab_size=49408,
            vision_num_layers=12,
            vision_features=768,
            vision_patch_size=32,
            text_features=512,
            text_num_heads=8,
            text_num_layers=12),
    'vit_b16':
        dict(
            embed_dim=512,
            vocab_size=49408,
            vision_num_layers=12,
            vision_features=768,
            vision_patch_size=16,
            text_features=512,
            text_num_heads=8,
            text_num_layers=12),
    'vit_l14':
        dict(
            embed_dim=768,
            vocab_size=49408,
            vision_num_layers=24,
            vision_features=1024,
            vision_patch_size=14,
            text_features=768,
            text_num_heads=12,
            text_num_layers=12),
    'vit_l14_336px':
        dict(
            embed_dim=768,
            vocab_size=49408,
            vision_num_layers=24,
            vision_features=1024,
            vision_patch_size=14,
            text_features=768,
            text_num_heads=12,
            text_num_layers=12),
    'resnet_50':
        dict(
            embed_dim=1024,
            vocab_size=49408,
            vision_num_layers=(3, 4, 6, 3),
            vision_features=64,
            text_features=512,
            text_num_heads=8,
            text_num_layers=12),
    'resnet_50x4':
        dict(
            embed_dim=640,
            vocab_size=49408,
            vision_num_layers=(4, 6, 10, 6),
            vision_features=80,
            text_features=640,
            text_num_heads=10,
            text_num_layers=12),
    'resnet_50x16':
        dict(
            embed_dim=768,
            vocab_size=49408,
            vision_num_layers=(6, 8, 18, 8),
            vision_features=96,
            text_features=768,
            text_num_heads=12,
            text_num_layers=12),
    'resnet_50x64':
        dict(
            embed_dim=1024,
            vocab_size=49408,
            vision_num_layers=(3, 15, 36, 10),
            vision_features=128,
            text_features=1024,
            text_num_heads=16,
            text_num_layers=12),
    'resnet_101':
        dict(
            embed_dim=512,
            vocab_size=49408,
            vision_num_layers=(3, 4, 23, 3),
            vision_features=64,
            text_features=512,
            text_num_heads=8,
            text_num_layers=12)
}

# pylint: disable=line-too-long
CHECKPOINTS = {
}
# pylint: enable=line-too-long
