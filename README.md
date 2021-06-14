# Uncertainty Baselines

[![Travis](https://travis-ci.org/google/uncertainty-baselines.svg?branch=master)](https://travis-ci.org/google/uncertainty-baselines)

The goal of Uncertainty Baselines is to provide a template for researchers to
build on. The baselines can be a starting point for any new ideas, applications,
and/or for communicating with other uncertainty and robustness researchers. This
is done in three ways:

1. Provide high-quality implementations of standard and state-of-the-art methods
   on standard tasks.
2. Have minimal dependencies on other files in the codebase. Baselines should be
   easily forkable without relying on other baselines and generic modules.
3. Prescribe best practices for uncertainty and robustness benchmarking.

__Motivation.__ There are many uncertainty and robustness implementations acrossGitHub. However, they are typically one-off experiments for a specific paper
(many papers don't even have code). There are no clear examples that uncertainty
researchers can build on to quickly prototype their work. Everyone must
implement their own baseline. In fact, even on standard tasks, every project
differs slightly in their experiment setup, whether it be architectures,
hyperparameters, or data preprocessing. This makes it difficult to compare
properly against baselines.

## Installation

To install the latest development version, run

```sh
pip install "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"
```

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

## Usage

### Baselines

The
[`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines)
directory includes all the baselines, organized by their training dataset.
For example,
[`baselines/cifar/determinstic.py`](https://github.com/google/uncertainty-baselines/tree/master/baselines/cifar/deterministic.py)
is a Wide ResNet 28-10 obtaining 96.0% test accuracy on CIFAR-10.

__Launching with TPUs.__ You often need TPUs to reproduce baselines. There are three options:

1. __Colab.__
[Colab offers free TPUs](https://colab.research.google.com/notebooks/tpu.ipynb).
This is the most convenient and budget-friendly option. You can experiment with
a baseline by copying its script and running it from scratch. This works well for simple experimentation. However, be careful relying on Colab long-term: TPU access isn't guaranteed, and Colab can only go so far for managing multiple long experiments.

2. __Google Cloud.__
This is the most flexible option. First, you'll need to
create a virtual machine instance (details
[here](https://cloud.google.com/compute/docs/instances/create-start-instance)).

    Here's an example to launch the BatchEnsemble baseline on CIFAR-10. We assume
    a few environment variables which are set up with the cloud TPU (details
    [here](https://cloud.google.com/tpu/docs/quickstart)).

    ```sh
    export BUCKET=gs://bucket-name
    export TPU_NAME=ub-cifar-batchensemble
    export DATA_DIR=$BUCKET/tensorflow_datasets
    export OUTPUT_DIR=$BUCKET/model

    python baselines/cifar/batchensemble.py \
        --tpu=$TPU_NAME \
        --data_dir=$DATA_DIR \
        --output_dir=$OUTPUT_DIR
    ```

    Note the TPU's accelerator type must align with the number of cores for
    the baseline (`num_cores` flag). In this example, BatchEnsemble uses a
    default of `num_cores=8`. So the TPU must be set up with `accelerator_type=v3-8`.

3. __Change the flags.__ For example, go from 8 TPU cores to 8 GPUs, or reduce the number of cores to train the baseline.

    ```sh
    python baselines/cifar/batchensemble.py \
        --data_dir=/tmp/tensorflow_datasets \
        --output_dir=/tmp/model \
        --use_gpu=True \
        --num_cores=8
    ```

    Results may be similar, but ultimately all bets are off. GPU vs TPU may not make much of a difference in practice, especially if you use the same numerical precision. However, changing the number of cores matters a lot. The total batch size during each training step is often determined by `num_cores`, so be careful!

### Datasets

The
[`ub.datasets`](https://github.com/google/uncertainty-baselines/tree/master/uncertainty_baselines/datasets)
module consists of datasets following the
[TensorFlow Datasets](https://www.tensorflow.org/datasets) API.
They add minimal logic such as default data preprocessing.

```python
import uncertainty_baselines as ub

# Load CIFAR-10, holding out 10% for validation.
dataset_builder = ub.datasets.Cifar10Dataset(split='train',
                                             validation_percent=0.1)
train_dataset = dataset_builder.load(batch_size=FLAGS.batch_size)
for batch in train_dataset:
  # Apply code over batches of the data.
```

You can also use `get` to instantiate datasets from strings (e.g., commandline
flags).

```python
dataset_builder = ub.datasets.get(dataset_name, split=split, **dataset_kwargs)
```

To use the datasets in Jax and PyTorch:

```python
for batch in tfds.as_numpy(ds):
  train_step(batch)
```

Note that `tfds.as_numpy` calls `tensor.numpy()`. This invokes an unnecessary
copy compared to `tensor._numpy()`.

```python
for batch in iter(ds):
  train_step(jax.tree_map(lambda y: y._numpy(), batch))
```

### Models

The
[`ub.models`](https://github.com/google/uncertainty-baselines/tree/master/uncertainty_baselines/models)
module consists of models following the
[`tf.keras.Model`](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
API.

```python
import uncertainty_baselines as ub

model = ub.models.wide_resnet(input_shape=(32, 32, 3),
                              depth=28,
                              width_multiplier=10,
                              num_classes=10,
                              l2=1e-4)
```

You can also use `get` to instantiate models from strings (e.g., commandline
flags).

```python
model = ub.models.get(model_name, batch_size=FLAGS.batch_size)
```

## Metrics

We define metrics used across datasets below. All results are reported by roughly 3 significant digits and averaged over 10 runs.

1. __# Parameters.__ Number of parameters in the model to make predictions after training.
2. __Test Accuracy.__ Accuracy over the test set. For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, the accuracy is

    ```sh
    1/N \sum_{n=1}^N 1[ \argmax{ p(yn | xn) } = yn ],
    ```

    where `1` is the indicator function that is 1 when the model's predicted class is equal to the label and 0 otherwise.
3. __Test Cal. Error.__ Expected calibration error (ECE) over the test set ([Naeini et al., 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090)). ECE discretizes the probability interval `[0, 1]` under equally spaced bins and assigns each predicted probability to the bin that encompasses it. The calibration error is the difference between the fraction of predictions in the bin that are correct (accuracy) and the mean of the probabilities in the bin (confidence). The expected calibration error averages across bins.

    For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, ECE computes a weighted average

    ```sh
    \sum_{b=1}^B n_b / N | acc(b) - conf(b) |,
    ```

    where `B` is the number of bins, `n_b` is the number of predictions in bin `b`, and `acc(b)` and `conf(b)` is the accuracy and confidence of bin `b` respectively.
4. __Test NLL.__ Negative log-likelihood over the test set (measured in nats). For a dataset of `N` input-output pairs `(xn, yn)`, the negative log-likelihood is

    ```sh
    -1/N \sum_{n=1}^N \log p(yn | xn).
    ```

    It is equivalent up to a constant to the KL divergence from the true data distribution to the model, therefore capturing the overall goodness of fit to the true distribution ([Murphy, 2012](https://www.cs.ubc.ca/~murphyk/MLbook/)). It can also be intepreted as the amount of bits (nats) to explain the data ([Grunwald, 2004](https://arxiv.org/abs/math/0406077)).
5. __Train/Test Runtime.__ Training runtime is the total wall-clock time to train the model, including any intermediate test set evaluations. Test Runtime refers to the time it takes to run a forward pass on the GPU/TPU, i.e., the duration for which the device is not idle. Note that Test Runtime does not include time on the coordinator: this is more precise in comparing baselines because including the coordinator adds overhead in GPU/TPU scheduling and data fetching---producing high variance results.

__Viewing metrics.__
Uncertainty Baselines writes TensorFlow summaries to the `model_dir` which can
be consumed by TensorBoard. This includes the TensorBoard hyperparameters
plugin, which can be used to analyze hyperparamter tuning sweeps.

If you wish to upload to the *PUBLICLY READABLE* [tensorboard.dev](https://tensorboard.dev), use:

```sh
tensorboard dev upload --logdir MODEL_DIR --plugins "scalars,graphs,hparams" --name "My experiment" --description "My experiment details"
```

## References

If you'd like to cite Uncertainty Baselines, use the following BibTeX entry.

> Z. Nado, N. Band, M. Collier, J. Djolonga, M. Dusenberry,
> S. Farquhar, A. Filos, M. Havasi, R. Jenatton, G.
> Jerfel, J. Liu, Z. Mariet, J. Nixon, S. Padhy, J. Ren, T.
> Rudner, Y. Wen, F. Wenzel, K. Murphy, D. Sculley, B.
> Lakshminarayanan, J. Snoek, Y. Gal, and D. Tran.
> [Uncertainty Baselines:  Benchmarks for uncertainty & robustness in deep learning](https://arxiv.org/abs/2106.04015),
> _arXiv preprint arXiv:2106.04015_, 2021.

```
@article{nado2021uncertainty,
  author = {Zachary Nado and Neil Band and Mark Collier and Josip Djolonga and Michael Dusenberry and Sebastian Farquhar and Angelos Filos and Marton Havasi and Rodolphe Jenatton and Ghassen Jerfel and Jeremiah Liu and Zelda Mariet and Jeremy Nixon and Shreyas Padhy and Jie Ren and Tim Rudner and Yeming Wen and Florian Wenzel and Kevin Murphy and D. Sculley and Balaji Lakshminarayanan and Jasper Snoek and Yarin Gal and Dustin Tran},
  title = {{Uncertainty Baselines}:  Benchmarks for Uncertainty \& Robustness in Deep Learning},
  journal = {arXiv preprint arXiv:2106.04015},
  year = {2021},
}
```

### Adding a Baseline

1. Write a script that loads the fixed training dataset and model. Typically, this is forked from other baselines.
2. After tuning, set the default flag values to the best hyperparameters.
3. Add the baseline's performance to the table of results in the corresponding `README.md`.

### Adding a Dataset

1. Add the bibtex reference to [`references.md`](https://github.com/google/uncertainty-baselines/blob/master/references.md).
2. Add the dataset definition to the datasets/ dir. Every file should have a subclass of `datasets.base.BaseDataset`, which at a minimum requires implementing a constructor, a `tfds.core.DatasetBuilder`, and `_create_process_example_fn`.
3. Add a test that at a minimum constructs the dataset and checks the shapes of elements.
4. Add the dataset to `datasets/datasets.py` for easy access.
5. Add the dataset class to `datasets/__init__.py`.

For an example of adding a dataset, see [this pull request](https://github.com/google/uncertainty-baselines/pull/175).

### Adding a Model

1. Add the bibtex reference to [`references.md`](https://github.com/google/uncertainty-baselines/blob/master/references.md).
2. Add the model definition to the models/ dir. Every file should have a `create_model` function with the following signature:

    ```python
    def create_model(
        batch_size: int,
        ...
        **unused_kwargs: Dict[str, Any])
        -> tf.keras.models.Model:
    ```

3. Add a test that at a minimum constructs the model and does a forward pass.
4. Add the model to `models/models.py` for easy access.
5. Add the `create_model` function to `models/__init__.py`.
