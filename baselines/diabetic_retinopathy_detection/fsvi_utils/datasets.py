import pdb
from time import time

import jax.numpy as jnp
import numpy as np
import seqtools
import tensorflow_datasets as tfds
import torch
import logging

from tqdm import tqdm

from baselines.diabetic_retinopathy_detection.utils import load_input_shape

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
):
    # SELECT TRAINING AND TEST DATA LOADERS
    (
        dataset_train,
        dataset_validation,
        dataset_test,
        output_dim,
        n_train,
    ) = get_diabetic_retinopathy(
        train_batch_size=batch_size,
        data_dir=data_dir,
        use_validation=use_validation,
        eval_batch_size=eval_batch_size,
    )
    input_shape = load_input_shape(dataset_train=dataset_train)
    input_shape = [1] + input_shape

    return (
        dataset_train,
        input_shape,
        output_dim,
        n_train,
    )


def get_diabetic_retinopathy(
    train_batch_size, eval_batch_size, data_dir, use_validation
):
    output_dim = 2
    # input_shape = [1, image_dim, image_dim, 3]

    dataset_train_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="train", data_dir=data_dir
    )
    dataset_train = dataset_train_builder.load(batch_size=train_batch_size)

    dataset_validation_builder = ub.datasets.get(
        "diabetic_retinopathy_detection",
        split="validation",

        data_dir=data_dir,
        is_training=not use_validation,
    )
    validation_batch_size = eval_batch_size if use_validation else train_batch_size
    dataset_validation = dataset_validation_builder.load(
        batch_size=validation_batch_size
    )
    if not use_validation:
        dataset_train = dataset_train.concatenate(dataset_validation)

    dataset_test_builder = ub.datasets.get(
        "diabetic_retinopathy_detection", split="test", data_dir=data_dir
    )
    dataset_test = dataset_test_builder.load(batch_size=eval_batch_size)

    print("Loading diabetic retinopathy dataset. This may take a few minutes.")
    ds_info = tfds.builder("diabetic_retinopathy_detection").info
    n_train = ds_info.splits["train"].num_examples

    logging.info("Finish getting data iterators")
    # for i in tqdm(range(loader_n_batches), desc="loading data"):
    #     # start = time()
    #     data = next(train_iterator)
    # print(f"it takes {time() - start:.2f} seconds")

    return (
        dataset_train,
        dataset_validation,
        dataset_test,
        output_dim,
        n_train,
    )


def _one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return np.array(x[:, None] == np.arange(k), dtype)


def collate_fn(batch):
    inputs = np.stack([x for x, _ in batch])
    targets = np.stack([y for _, y in batch])
    return inputs, targets
