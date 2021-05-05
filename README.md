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

__Motivation.__ There are many uncertainty and robustness implementations across
GitHub. However, they are typically one-off experiments for a specific paper
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

The
[`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines)
directory includes all the baselines, organized by their training dataset.
For example,
[`baselines/cifar/determinstic.py`](https://github.com/google/uncertainty-baselines/tree/master/baselines/cifar/deterministic.py)
is a Wide ResNet 28-10 obtaining 96.0% test accuracy on CIFAR-10.

The
[`experimental/`](https://github.com/google/uncertainty-baselines/tree/master/experimental)
directory is for active research projects.

There are two modules in Uncertainty Baselines' API which we describe next:
`import uncertainty_baselines as ub`.

### Datasets

The
[`ub.datasets`](https://github.com/google/uncertainty-baselines/tree/master/uncertainty_baselines/datasets)
module consists of datasets following the
[TensorFlow Datasets](https://www.tensorflow.org/datasets) API.
They add minimal logic such as default data preprocessing.

```python
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

## Contributors

Contributors (past and present):

* Angelos Filos
* Balaji Lakshminarayanan
* D Sculley
* Dustin Tran
* Florian Wenzel
* Ghassen Jerfel
* Jeremiah Liu
* Jeremy Nixon
* Jie Ren
* Jasper Snoek
* Josip Djolonga
* Kevin Murphy
* Marton Havasi
* Michael W. Dusenberry
* Neil Band
* Rodolphe Jenatton
* Sebastian Farquhar
* Shreyas Padhy
* Tim G. J. Rudner
* Yarin Gal
* Yeming Wen
* Zachary Nado

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
