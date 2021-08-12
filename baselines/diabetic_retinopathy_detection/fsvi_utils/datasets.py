import logging

import tensorflow_datasets as tfds
import torch

from baselines.diabetic_retinopathy_detection.utils import load_input_shape

try:
    import uncertainty_baselines as ub
except:
    print("WARNING: uncertainty_baselines could not be loaded.")

# TODO: remove this manual seed?
torch.manual_seed(0)


def load_data(
    train_batch_size,
    use_validation: bool,
    data_dir="/scratch/data/diabetic-retinopathy-detection",
    eval_batch_size: int = None,
):
    # SELECT TRAINING AND TEST DATA LOADERS
    output_dim = 2
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

    n_valid = ds_info.splits["validation"].num_examples // eval_batch_size
    n_test = ds_info.splits["test"].num_examples // eval_batch_size

    logging.info("Finish getting data iterators")

    input_shape = load_input_shape(dataset_train=dataset_train)
    input_shape = [1] + input_shape

    return (
        dataset_train,
        dataset_validation,
        dataset_test,
        input_shape,
        output_dim,
        n_train,
        n_valid,
        n_test,
    )
