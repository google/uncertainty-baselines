# Uncertainty Baselines

[![Travis](https://travis-ci.org/google/uncertainty-baselines.svg?branch=master)](https://travis-ci.org/google/uncertainty-baselines)

The goal of Uncertainty Baselines is to provide a template for researchers to build on. The baselines can be a starting point for any new ideas, applications, and/or for communicating with other uncertainty and robustness researchers. This is done in three ways:

1. Provide high-quality implementations of standard and state-of-the-art methods on standard tasks.
2. Have minimal dependencies on other files in the codebase. Baselines should be easily forkable without relying on other baselines and generic modules.
3. Prescribe best practices for training and evaluating uncertainty models.

__Motivation.__ There are many uncertainty implementations across GitHub. However, they are typically one-off experiments for a specific paper (many papers don't even have code). This raises three problems. First, there are no clear examples that uncertainty researchers can build on to quickly prototype their work. Everyone must implement their own baseline. Second, even on standard tasks such as CIFAR-10, projects differ slightly in their experiment setup, whether it be architectures, hyperparameters, or data preprocessing. This makes it difficult to compare properly across methods. Third, there is no clear guidance on which ideas and tricks necessarily contribute to getting best performance and/or are generally robust to hyperparameters.

All of our baselines are (so far) in TF2 Keras Models with tf.data pipelines. We welcome Jax and PyTorch users to use our datasets, for example via Python for loops:

```
for batch in tfds.as_numpy(ds):
  train_step(batch)
```
Although note that `tfds.as_numpy` calls `tensor.numpy()` which invokes an unnecessary copy compared to `tensor._numpy()`:

```
for batch in iter(ds):
  train_step(jax.tree_map(lambda y: y._numpy(), batch))
```

## Installation

To install the latest development version, run

```sh
pip install "git+https://github.com/google/uncertainty-baselines.git#egg=uncertainty_baselines"
```

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

## Usage

Access Uncertainty Baselines' API via `import uncertainty_baselines as ub`. To
run end-to-end examples with strong performance, see the
[`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines)
directory. For example,
[`baselines/cifar/determinstic.py`](https://github.com/google/uncertainty-baselines/tree/master/baselines/cifar/deterministic.py)
is a Wide ResNet 28-10 obtaining 96.0% test accuracy on CIFAR-10.

The
[`experimental/`](https://github.com/google/uncertainty-baselines/tree/master/experimental)
directory is for active research projects.

Below we outline modules in Uncertainty Baselines.

### Datasets

The [`ub.datasets`](https://github.com/google/uncertainty-baselines/tree/master/uncertainty_baselines/datasets) module consists of
datasets following the `tf.data.Dataset` and TFDS APIs. Typically, they add minimal logic
on top of TensorFlow Datasets such as default data preprocessing. Access it as:

```python
dataset_builder = ub.datasets.Cifar10Dataset(
    split='train', validation_percent=0.1)  # Use 5000 validation images.
train_dataset = dataset_builder.load(batch_size=FLAGS.batch_size)
```

Alternatively, use the getter command:

```python
dataset_builder = ub.datasets.get(
    dataset_name,
    split=split,
    **dataset_kwargs)
```

Supported datasets include:

- CIFAR-10
- CIFAR-100
- Civil Comments Toxicity Classification, [download](https://www.tensorflow.org/datasets/catalog/civil_comments)
- CLINC Intent Detection, [download](https://github.com/clinc/oos-eval/blob/master/data/data_full.json)
- Criteo Ads, [download](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
- GLUE, [download](https://gluebenchmark.com/)
- ImageNet
- MNIST
- MNLI
- Wikipedia Talk Toxicity Classification, [download](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes)

__Adding a new dataset.__

1. Add the bibtex reference to [`references.md`](https://github.com/google/uncertainty-baselines/blob/master/references.md).
2. Add the dataset definition to the datasets/ dir. Every file should have a subclass of `datasets.base.BaseDataset`, which at a minimum requires implementing a constructor, a `tfds.core.DatasetBuilder`, and `_create_process_example_fn`.
3. Add a test that at a minimum constructs the dataset and checks the shapes of elements.
4. Add the dataset to `datasets/datasets.py` for easy access.
5. Add the dataset class to `datasets/__init__.py`.

For an example of adding a dataset, see [this pull request](https://github.com/google/uncertainty-baselines/pull/175).

### Models

The
[`ub.models`](https://github.com/google/uncertainty-baselines/tree/master/uncertainty_baselines/models)
module consists of models following the `tf.keras.Model` API. Access it as:

```python
model = ub.models.ResNet20Builder(batch_size=FLAGS.batch_size, l2_weight=None)
```

Alternatively, use the getter command:

```python
model = ub.models.get(FLAGS.model_name, batch_size=FLAGS.batch_size)
```

Supported models include:

- ResNet-20 v1
- ResNet-50 v1
- Wide ResNet-*-*
- Criteo MLP
- Text CNN
- BERT

__Adding a new model.__

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

### Methods
The end-to-end baseline training scripts can be found in [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

Supported methods include:

- Deterministic
- BatchEnsemble
- Ensemble
- Hyper-batch Ensemble
- Hyper-deep Ensemble ([Quick Intro Notebook](https://github.com/google/uncertainty-baselines/blob/master/baselines/notebooks/Hyperparameter_Ensembles.ipynb))
- MIMO
- Rank-1 BNN
- SNGP
- Monte Carlo Dropout
- Variational Inference



## Metrics

We define metrics used across datasets below. All results are reported by roughly 3 significant digits and averaged over 10 runs.

1. __# Parameters.__ Number of parameters in the model to make predictions after training.
2. __Train/Test Accuracy.__ Accuracy over the train and test sets respectively. For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, the accuracy is

    ```sh
    1/N \sum_{n=1}^N 1[ \argmax{ p(yn | xn) } = yn ],
    ```

    where `1` is the indicator function that is 1 when the model's predicted class is equal to the label and 0 otherwise.
3. __Train/Test Cal. Error.__ Expected calibration error (ECE) over the train and test sets respectively ([Naeini et al., 2015](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4410090)). ECE discretizes the probability interval `[0, 1]` under equally spaced bins and assigns each predicted probability to the bin that encompasses it. The calibration error is the difference between the fraction of predictions in the bin that are correct (accuracy) and the mean of the probabilities in the bin (confidence). The expected calibration error averages across bins.

    For a dataset of `N` input-output pairs `(xn, yn)` where the label `yn` takes on 1 of `K` values, ECE computes a weighted average

    ```sh
    \sum_{b=1}^B n_b / N | acc(b) - conf(b) |,
    ```

    where `B` is the number of bins, `n_b` is the number of predictions in bin `b`, and `acc(b)` and `conf(b)` is the accuracy and confidence of bin `b` respectively.
4. __Train/Test NLL.__ Negative log-likelihood over the train and test sets respectively (measured in nats). For a dataset of `N` input-output pairs `(xn, yn)`, the negative log-likelihood is

    ```sh
    -1/N \sum_{n=1}^N \log p(yn | xn).
    ```

    It is equivalent up to a constant to the KL divergence from the true data distribution to the model, therefore capturing the overall goodness of fit to the true distribution ([Murphy, 2012](https://www.cs.ubc.ca/~murphyk/MLbook/)). It can also be intepreted as the amount of bits (nats) to explain the data ([Grunwald, 2004](https://arxiv.org/abs/math/0406077)).
5. __Train/Test Runtime.__ Training runtime is the total wall-clock time to train the model, including any intermediate test set evaluations. Wall-clock Test Runtime refers to the wall time of testing a batch of inputs. Compute Test Runtime refers to the time it takes to run a forward pass on the GPU/TPU i.e. the duration for which the device is not idle. Compute Test Runtime is lower than Wall-clock Test Runtime becuase it does not include the time it takes to schedule the job on the GPU/TPU and fetch the data.

__Viewing metrics.__
Uncertainty Baselines writes TensorFlow summaries to the `model_dir` which can
be consumed by TensorBoard. This included the TensorBoard hyperparameters
plugin, which can be used to analyze hyperparamter tuning sweeps.

If you wish to upload to the *PUBLICLY READABLE* [tensorboard.dev](https://tensorboard.dev/) you can use the following command:

```sh
tensorboard dev upload --logdir MODEL_DIR --plugins "scalars,graphs,hparams" --name "My experiment" --description "My experiment details"
```


## Contributors

Contributors (past and present):

* Angelos Filos
* Dustin Tran
* Florian Wenzel
* Ghassen Jerfel
* Jeremiah Liu
* Jeremy Nixon
* Jie Ren
* Josip Djolonga
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
