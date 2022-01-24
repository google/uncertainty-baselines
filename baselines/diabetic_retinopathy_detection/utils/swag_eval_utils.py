import torch
from absl import logging

from swag_utils import utils as swag_utils


def test_step_swag(
    model_to_eval, iterator, num_steps, eval_batch_size,
    sigmoid, device, image_h=512, image_w=512
):
  """T

  Need to BatchNorm over the training set for every stochastic sample.
  Therefore, we:
    1. Sample from the posterior,
    2. (here) Compute a full epoch, and return predictions and labels,
    3. Update metrics altogether at the end.
"""

  def step_fn(inputs):
    images = inputs['features']
    labels = inputs['labels']
    images = torch.from_numpy(images._numpy()).view(eval_batch_size, 3,  # pylint: disable=protected-access
                                                    image_h,
                                                    image_w).to(device)
    labels = torch.from_numpy(
      labels._numpy()).to(device).float().unsqueeze(
      -1)  # pylint: disable=protected-access
    with torch.no_grad():
      logits = model_to_eval(images)

    probs = sigmoid(logits)
    return labels, probs

  labels_list = []
  probs_list = []
  model_to_eval.eval()
  for _ in range(num_steps):
    labels, probs = step_fn(next(iterator))
    labels_list.append(labels)
    probs_list.append(probs)

  return {
    'labels': torch.cat(labels_list, dim=0),
    'probs': torch.cat(probs_list, dim=0)
  }

def evaluate_swag_on_datasets(
  train_iterator, train_batch_size, eval_datasets, steps, eval_model,
  eval_batch_size, num_samples, swag_is_active,
  scale, sample_with_cov, device, epoch, sigmoid, image_h=512, image_w=512
):
  labels = []
  probs = []
  # dataset_split_to_containers = {}

  for sample in range(num_samples):
    # Sample from approx posterior
    if swag_is_active:
      eval_model.sample(scale=scale, cov=sample_with_cov)
      # BN Update requires iteration over full train loop, hence we
      # put the MC sampling outside of the evaluation loops.
      swag_utils.bn_update(
        train_iterator, eval_model, num_train_steps=steps['train'],
        train_batch_size=train_batch_size, image_h=512, image_w=512,
        device=device)

    for dataset_key, eval_dataset in eval_datasets.items():
      dataset_iterator = iter(eval_dataset)
      logging.info(
        f'Evaluating SWAG on {dataset_key}, sample {sample} '
        f'at epoch: %s', epoch + 1)

      epoch_results = test_step_swag(
        model_to_eval=eval_model, iterator=dataset_iterator,
        num_steps=steps[dataset_key], eval_batch_size=eval_batch_size,
        sigmoid=sigmoid, device=device, image_h=image_h, image_w=image_w)

