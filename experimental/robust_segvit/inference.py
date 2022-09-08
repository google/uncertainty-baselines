# coding=utf-8
# Copyright 2022 The Uncertainty Baselines Authors.
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

"""Run inference following torch model."""

import jax
import jax.numpy as jnp


def resize(x, smaller_size):
  """Resize image according to smaller size.

  Args:
    x: image tensor of shape [B, C, H, W]
    smaller_size: int
  Returns:
    im_res: image tensor reshaped.
  """
  im = x
  b, c, h, w = im.shape

  if h < w:
    ratio = w / h
    h_res, w_res = smaller_size, ratio * smaller_size
  else:
    ratio = h / w
    h_res, w_res = ratio * smaller_size, smaller_size

  if min(h, w) < smaller_size:
    im_res = jax.image.resize(im, [b, c, int(h_res), int(w_res)], "bilinear")

  else:
    im_res = im
  return im_res


def sliding_window(x, flip, window_size, window_stride):
  """Slice image in windows.

  Args:
    x: jnp.array, Image tensor.
    flip: boolean, to flip input.
    window_size: int,  size of windows in which to split im.
    window_stride: int, size of stride used to split im.

  Returns:
    windows: dict of windows and metadata.
  """
  im = x
  _, _, h, w = im.shape
  ws = window_size

  windows = {"crop": [], "anchors": []}
  h_anchors = range(0, h, window_stride)
  w_anchors = range(0, w, window_stride)

  h_anchors = [h_ for h_ in h_anchors if h_ < h - ws] + [h - ws]
  w_anchors = [w_ for w_ in w_anchors if w_ < w - ws] + [w - ws]

  for ha in h_anchors:
    for wa in w_anchors:
      window = im[:, :, ha : ha + ws, wa : wa + ws]
      windows["crop"].append(window)
      windows["anchors"].append((ha, wa))
  windows["flip"] = flip
  windows["shape"] = (h, w)
  return windows


def merge_windows(windows, window_size, ori_shape, apply_softmax=False):
  """Merges windows split for evaluation to img of ori_shape.

  Args:
    windows: dict of windows in which an image is split.
    window_size: int, size of window used to split image.
    ori_shape: int, size of original image.
    apply_softmax: flag to apply softmax to resized logit

  Returns:
    result: img of shape [C, H, W], where ori_shape=(H, W)
  """
  ws = window_size
  im_windows = windows["seg_maps"]
  anchors = windows["anchors"]
  c = im_windows[0].shape[0]
  h, w = windows["shape"]
  flip = windows["flip"]

  logit = jnp.zeros((c, h, w))
  count = jnp.zeros((1, h, w))
  for window, (ha, wa) in zip(im_windows, anchors):
    logit = logit.at[:, ha : ha + ws, wa : wa + ws].add(window)
    count = count.at[:, ha : ha + ws, wa : wa + ws].add(1)
  logit = logit / count
  logit = jax.image.resize(
      jnp.expand_dims(logit, 0), [1, c, ori_shape[0], ori_shape[1]],
      "bilinear")[0]

  if flip:
    logit = jnp.flip(logit, (2,))

  if apply_softmax:
    result = jax.nn.softmax(logit, 0)
  else:
    result = logit
  return result


def inference(
    model,
    variables,
    ims,
    ims_metas,
    ori_shape,
    window_size,
    window_stride,
    batch_size,
    apply_softmax=True,
    ens_size=1,
):
  """Inference function following opensegmm.

  Args:
    model: flax model we evaluate.
    variables: variables updated in flax_model.
    ims: list of tensors of shape ([BxCxHxW]).
    ims_metas: dict with metadata.
    ori_shape: original shape of image.
    window_size: size of window tensor is split into.
    window_stride: stride used to split img.
    batch_size: batch size for image.
    apply_softmax: apply softmax to segmentation map, True if len(im)>1.
    ens_size: size of batch ensemble.
  Returns:
    seg_map: segmentation map for image [B, C, H, W]
  """
  c = model.num_classes
  assert len(ims) == 1
  seg_map = jnp.zeros((ens_size, c, ori_shape[0], ori_shape[1]))

  for im, im_metas in zip(ims, ims_metas):
    im = resize(im, window_size)
    flip = im_metas["flip"]
    windows = sliding_window(im, flip, window_size, window_stride)
    crops = jnp.stack(windows.pop("crop"))[:, 0]  # (B, C, H, W)
    b = len(crops)
    wb = batch_size
    seg_maps = jnp.zeros((ens_size, b, c, window_size, window_size))

    # evaluate cropped image
    for i in range(0, b, wb):
      input_ = jnp.transpose(crops[i:i + wb], [0, 2, 3, 1])
      logit_ = model.apply(
          variables, input_, train=False, debug=False)[0]

      # split logit given batch ensemble size.
      logit_ = jnp.asarray(jnp.split(logit_, ens_size))
      for ens_id in range(ens_size):
        seg_maps = seg_maps.at[ens_id, i:i + wb].set(
            jnp.transpose(logit_[ens_id], [0, 3, 1, 2]))

    # merge windows per batch ensemble member
    for ens_id in range(ens_size):
      windows["seg_maps"] = seg_maps[ens_id]
      im_seg_map = merge_windows(
          windows, window_size, ori_shape, apply_softmax=apply_softmax)
      seg_map = seg_map.at[ens_id].add(im_seg_map)
  seg_map /= len(ims)

  return seg_map


def process_batch(*,
                  model,
                  variables,
                  inputs,
                  window_size=768,
                  window_stride=736,
                  window_batch_size=4,
                  ori_shape,
                  apply_softmax=False,
                  ens_size=1):
  """Process batch with eval function similar to segmm."""
  b, _, _, _ = inputs.shape
  c = model.num_classes

  seg_preds = jnp.zeros((ens_size, b, ori_shape[0], ori_shape[1], c))

  ims_metas = dict(flip=False)
  for im_idx, im in enumerate(inputs):
    im = jnp.expand_dims(jnp.transpose(im, [2, 0, 1]), 0)  # B x C x H x W
    seg_pred = inference(
        model=model,
        variables=variables,
        ims=[im],
        ims_metas=[ims_metas],
        ori_shape=ori_shape,
        window_size=window_size,
        window_stride=window_stride,
        batch_size=window_batch_size,
        apply_softmax=apply_softmax,
        ens_size=ens_size,
    )
    seg_pred = jnp.transpose(seg_pred, [0, 2, 3, 1])  # B, H, W, C
    seg_pred = jnp.asarray(jnp.split(seg_pred, ens_size))
    for ens_idx in range(ens_size):
      seg_preds = seg_preds.at[ens_idx, im_idx].set(seg_pred[ens_idx, im_idx])
  seg_preds = jnp.reshape(seg_preds,
                          [ens_size * b, ori_shape[0], ori_shape[1], c])

  return seg_preds
