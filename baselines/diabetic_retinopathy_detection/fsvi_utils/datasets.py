import pdb
from time import time

import jax.numpy as jnp
import numpy as np
import seqtools
import tensorflow_datasets as tfds
import torch
import logging

from tqdm import tqdm

try:
    import uncertainty_baselines as ub
except:
    print("WARNING: uncertainty_baselines could not be loaded.")

# TODO: remove this manual seed?
torch.manual_seed(0)


def load_data(
    batch_size,
    use_validation: bool,
    data_dir="/scratch/data/diabetic-retinopathy-detection",
    eval_batch_size: int = None,
    n_batches: int = None,
):
    # SELECT TRAINING AND TEST DATA LOADERS
    (_, output_dim, trainloader, x_train, _, _, _,) = get_diabetic_retinopathy(
        train_batch_size=batch_size,
        data_dir=data_dir,
        n_batches=n_batches,
        use_validation=use_validation,
        eval_batch_size=eval_batch_size,
    )

    n_train = x_train.shape[0]
    input_shape = list(x_train.shape)
    input_shape[0] = 1
    return (
        trainloader,
        input_shape,
        output_dim,
        n_train,
    )


def get_diabetic_retinopathy(
    train_batch_size, eval_batch_size, data_dir, use_validation, n_batches=None
):
    image_dim = 128
    output_dim = 2
    # input_shape = [1, image_dim, image_dim, 3]

    dataset_train_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="train", data_dir=data_dir
    )
    trainloader = dataset_train_builder.load(batch_size=train_batch_size)

    dataset_validation_builder = ub.datasets.get(
        "diabetic_retinopathy_detection",
        split="validation",
        data_dir=data_dir,
        is_training=not use_validation,
    )
    validation_batch_size = eval_batch_size if use_validation else train_batch_size
    valloader = dataset_validation_builder.load(batch_size=validation_batch_size)
    trainloader = trainloader.concatenate(valloader)

    dataset_test_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="test", data_dir=data_dir
    )
    testloader = dataset_test_builder.load(batch_size=eval_batch_size)

    print("Loading diabetic retinopathy dataset. This may take a few minutes.")
    train_iterator = iter(trainloader)
    ds_info = tfds.builder("diabetic_retinopathy_detection").info
    loader_n_batches = ds_info.splits["train"].num_examples // train_batch_size
    loader_n_batches = loader_n_batches if n_batches is None else n_batches

    logging.info("Finish getting data iterators")
    for i in tqdm(range(loader_n_batches), desc="loading data"):
        start = time()
        data = next(train_iterator)
        print(f"it takes {time() - start:.2f} seconds")
        if i == 0:
            x_train = data["features"]._numpy().transpose(0, 3, 1, 2)
            y_train = data["labels"]._numpy()
        else:
            x_train = np.concatenate(
                [x_train, data["features"]._numpy().transpose(0, 3, 1, 2)], 0
            )
            y_train = np.concatenate([y_train, data["labels"]._numpy()], 0)

    x_train = x_train.transpose(0, 2, 3, 1)
    train_dataset = seqtools.collate([x_train, y_train])
    trainloader = seqtools.batch(train_dataset, train_batch_size, collate_fn=collate_fn)

    test_iterator = iter(testloader)
    data = next(test_iterator)
    x_test = data["features"]._numpy()
    y_test = _one_hot(data["labels"]._numpy(), output_dim)

    return (
        image_dim,
        output_dim,
        trainloader,
        x_train,
        y_train,
        x_test,
        y_test,
    )


def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def collate_fn(batch):
    inputs = np.stack([x for x, _ in batch])
    targets = np.stack([y for _, y in batch])
    return inputs, targets
