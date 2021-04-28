import torch


def count_parameters(model):
  r"""
  Due to Federico Baldassarre
  https://discuss.pytorch.org/t/how-do-i-check-the-number-of-
  parameters-of-a-model/4325/7

  Counts number of trainable model parameters.
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)


def checkpoint_torch_model(model, optimizer, epoch, checkpoint_path):
  checkpoint_dict = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict()}
  torch.save(checkpoint_dict, checkpoint_path)
