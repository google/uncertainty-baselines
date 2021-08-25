import itertools
import logging
import os
import pathlib

import numpy as np
import tensorflow as tf
import torch
import torch.nn.functional as F
import tqdm


def flatten(lst):
  tmp = [i.contiguous().view(-1, 1) for i in lst]
  return torch.cat(tmp).view(-1)


def unflatten_like(vector, likeTensorList):
  # Takes a flat torch.tensor and unflattens it to a list of torch.tensors
  #    shaped like likeTensorList
  outList = []
  i = 0
  for tensor in likeTensorList:
    # n = module._parameters[name].numel()
    n = tensor.numel()
    outList.append(vector[:, i : i + n].view(tensor.shape))
    i += n
  return outList


def LogSumExp(x, dim=0):
  m, _ = torch.max(x, dim=dim, keepdim=True)
  return m + torch.log((x - m).exp().sum(dim=dim, keepdim=True))


def adjust_learning_rate(optimizer, lr):
  for param_group in optimizer.param_groups:
    param_group["lr"] = lr
  return lr


def schedule(swa_start, swa_lr, base_learning_rate, epoch):
  t = epoch / swa_start
  lr_ratio = swa_lr / base_learning_rate
  if t <= 0.5:
    factor = 1.0
  elif t <= 0.9:
    factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
  else:
    factor = lr_ratio
  return base_learning_rate * factor


def save_checkpoint(dir, epoch, name="checkpoint", **kwargs):
  state = {"epoch": epoch}
  state.update(kwargs)
  checkpoint_file_name = "%s-%d.pt" % (name, epoch)

  # Save to tmp
  swag_tmp_dir = '/tmp/torch_swag_checkpoints'
  pathlib.Path(swag_tmp_dir).mkdir(parents=True, exist_ok=True)
  tmp_path = os.path.join(swag_tmp_dir, checkpoint_file_name)
  torch.save(state, tmp_path)
  logging.info(f'Saved Torch model to tmp path: {tmp_path}')

  # Copy to remote
  remote_path = os.path.join(dir, checkpoint_file_name)
  tf.io.gfile.copy(tmp_path, remote_path, overwrite=True)
  assert tf.io.gfile.exists(remote_path)
  logging.info(f'Copied torch model to remote path: {remote_path}')

  # Remove tmp file
  tf.io.gfile.remove(tmp_path)
  assert not tf.io.gfile.exists(tmp_path)
  logging.info(f'Removed torch model from tmp path')


# def eval(loader, model, criterion, cuda=True, regression=False, verbose=False):
#   loss_sum = 0.0
#   correct = 0.0
#   num_objects_total = len(loader.dataset)
#
#   model.eval()
#
#   with torch.no_grad():
#     if verbose:
#       loader = tqdm.tqdm(loader)
#     for i, (input, target) in enumerate(loader):
#       if cuda:
#         input = input.cuda(non_blocking=True)
#         target = target.cuda(non_blocking=True)
#
#       loss, output = criterion(model, input, target)
#
#       loss_sum += loss.item() * input.size(0)
#
#       if not regression:
#         pred = output.data.argmax(1, keepdim=True)
#         correct += pred.eq(target.data.view_as(pred)).sum().item()
#
#   return {
#     "loss": loss_sum / num_objects_total,
#     "accuracy": None if regression else correct / num_objects_total * 100.0,
#   }


def predict(loader, model, verbose=False):
  predictions = list()
  targets = list()

  model.eval()

  if verbose:
    loader = tqdm.tqdm(loader)

  offset = 0
  with torch.no_grad():
    for input, target in loader:
      input = input.cuda(non_blocking=True)
      output = model(input)

      batch_size = input.size(0)
      predictions.append(F.softmax(output, dim=1).cpu().numpy())
      targets.append(target.numpy())
      offset += batch_size

  return {"predictions": np.vstack(predictions), "targets": np.concatenate(targets)}


def moving_average(net1, net2, alpha=1):
  for param1, param2 in zip(net1.parameters(), net2.parameters()):
    param1.data *= 1.0 - alpha
    param1.data += param2.data * alpha


def _check_bn(module, flag):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    flag[0] = True


def check_bn(model):
  flag = [False]
  model.apply(lambda module: _check_bn(module, flag))
  return flag[0]


def reset_bn(module):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.running_mean = torch.zeros_like(module.running_mean)
    module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    momenta[module] = module.momentum


def _set_momenta(module, momenta):
  if issubclass(module.__class__, torch.nn.modules.batchnorm._BatchNorm):
    module.momentum = momenta[module]


def bn_update(loader, model, num_train_steps,
              train_batch_size, image_h, image_w,
              device, verbose=False, subset=None, **kwargs
):
  """
      BatchNorm buffers update (if any).
      Performs 1 epochs to estimate buffers average using train dataset.

      :param: loader - tf.Data iterator
      # :param loader: train dataset loader for buffers average estimation.
      :param model: model being update
      :return: None
  """
  if not check_bn(model):
    return
  model.train()
  momenta = {}
  model.apply(reset_bn)
  model.apply(lambda module: _get_momenta(module, momenta))
  n = 0
  # num_batches = len(loader)

  with torch.no_grad():
    # if subset is not None:
    #   num_batches = int(num_batches * subset)
    #   loader = itertools.islice(loader, num_batches)
    if verbose:
      # loader = tqdm.tqdm(loader, total=num_batches)
      loader = tqdm.tqdm(loader, total=num_train_steps)

    for _ in range(num_train_steps):
    # for input, _ in loader:
      inputs = next(loader)
      input = inputs['features']
      input = torch.from_numpy(input._numpy()).view(train_batch_size, 3,  # pylint: disable=protected-access
                                                    image_h,
                                                    image_w).to(
        device, non_blocking=True)

      # input = input.cuda(non_blocking=True)
      input_var = torch.autograd.Variable(input)
      b = input_var.data.size(0)

      momentum = b / (n + b)
      for module in momenta.keys():
        module.momentum = momentum

      model(input_var, **kwargs)
      n += b

  model.apply(lambda module: _set_momenta(module, momenta))


def inv_softmax(x, eps=1e-10):
  return torch.log(x / (1.0 - x + eps))


def predictions(test_loader, model, seed=None, cuda=True, regression=False, **kwargs):
  # will assume that model is already in eval mode
  # model.eval()
  preds = []
  targets = []
  for input, target in test_loader:
    if seed is not None:
      torch.manual_seed(seed)
    if cuda:
      input = input.cuda(non_blocking=True)
    output = model(input, **kwargs)
    if regression:
      preds.append(output.cpu().data.numpy())
    else:
      probs = F.softmax(output, dim=1)
      preds.append(probs.cpu().data.numpy())
    targets.append(target.numpy())
  return np.vstack(preds), np.concatenate(targets)


# def schedule(epoch, lr_init, epochs, swa, swa_start=None, swa_lr=None):
#     t = (epoch) / (swa_start if swa else epochs)
#     lr_ratio = swa_lr / lr_init if swa else 0.01
#     if t <= 0.5:
#         factor = 1.0
#     elif t <= 0.9:
#         factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
#     else:
#         factor = lr_ratio
#     return lr_init * factor
