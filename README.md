# Uncertainty Baselines

[![Travis](https://travis-ci.org/google/uncertainty-baselines.svg?branch=master)](https://travis-ci.org/google/uncertainty-baselines)

The goal of Uncertainty Baselines is to provide a template for researchers to build on. The baselines can be a starting point for any new ideas, applications, and/or for communicating with other uncertainty and robustness researchers. This is done in three ways:

1. Provide high-quality implementations of standard and state-of-the-art methods on standard tasks.
2. Have minimal dependencies on other files in the codebase. Baselines should be easily forkable without relying on other baselines and generic modules.
3. Prescribe best practices for training and evaluating uncertainty models.

__Motivation.__ There are many uncertainty implementations across GitHub. However, they are typically one-off experiments for a specific paper (many papers don't even have code). This raises three problems. First, there are no clear examples that uncertainty researchers can build on to quickly prototype their work. Everyone must implement their own baseline. Second, even on standard tasks such as CIFAR-10, projects differ slightly in their experiment setup, whether it be architectures, hyperparameters, or data preprocessing. This makes it difficult to compare properly across methods. Third, there is no clear guidance on which ideas and tricks necessarily contribute to getting best performance and/or are generally robust to hyperparameters.

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
datasets following the `tf.data.Dataset` API. Typically, they add minimal logic
on top of TensorFlow Datasets such as default data preprocessing. Access it as:

```python
dataset_builder = ub.datasets.Cifar10Dataset(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    validation_percent=0.1)  # Use 5000 validation images.
train_dataset = ub.utils.build_dataset(
    dataset_builder, strategy, 'train', as_tuple=True) # as_tuple for model.fit()
```

Alternatively, use the getter command:

```python
dataset_builder = ub.datasets.get(
    dataset_name,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
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

1. Add the bibtex reference to the `References` section below.
2. Add the dataset definition to the datasets/ dir. Every file should have a subclass of `datasets.base.BaseDataset`, which at a minimum requires implementing a constructor, `_read_examples`, and `_create_process_example_fn`.
3. Add a test that at a minimum constructs the dataset and checks the shapes of elements.
4. Add the dataset to `datasets/datasets.py` for easy access.
5. Add the dataset class to `datasets/__init__.py`.

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

1. Add the bibtex reference to the `References` section below.
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

## References

We don't yet have a recommended BibTeX entry if you'd like to cite our work. If
you're using specific datasets, models, or baselines, see below.

### Datasets

```
# CIFAR-10
@article{cifar10,
title = {CIFAR-10 (Canadian Institute for Advanced Research)},
author = {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
url = {http://www.cs.toronto.edu/~kriz/cifar.html},
}

# CIFAR-100
@article{cifar100,
title = {CIFAR-100 (Canadian Institute for Advanced Research)},
author = {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
url = {http://www.cs.toronto.edu/~kriz/cifar.html},
}

# Civil Comments Toxicity Classification
@article{civil_comments,
  title     = {Nuanced Metrics for Measuring Unintended Bias with Real Data for Text
               Classification},
  author    = {Daniel Borkan and Lucas Dixon and Jeffrey Sorensen and Nithum Thain and Lucy Vasserman},
  journal   = {CoRR},
  volume    = {abs/1903.04561},
  year      = {2019},
  url       = {http://arxiv.org/abs/1903.04561},
  archivePrefix = {arXiv},
  eprint    = {1903.04561},
  timestamp = {Sun, 31 Mar 2019 19:01:24 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1903-04561},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

# CLINIC
@article{clinic,
  title = {An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction},
  author = {Larson, Stefan and Mahendran, Anish and Peper, Joseph J and Clarke, Christopher and Lee, Andrew and Hill, Parker and Kummerfeld, Jonathan K and Leach, Kevin and Laurenzano, Michael A and Tang, Lingjia and others},
  journal = {arXiv preprint arXiv:1909.02027},
  year = {2019}
}

# Criteo
@article{criteo,
title = {Display Advertising Challenge},
url = {https://www.kaggle.com/c/criteo-display-ad-challenge.},
}

# GLUE
@inproceedings{glue,
    title = "{GLUE}: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding",
    author = "Wang, Alex  and
      Singh, Amanpreet  and
      Michael, Julian  and
      Hill, Felix  and
      Levy, Omer  and
      Bowman, Samuel",
    booktitle = "Proceedings of the 2018 {EMNLP} Workshop {B}lackbox{NLP}: Analyzing and Interpreting Neural Networks for {NLP}",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-5446",
    doi = "10.18653/v1/W18-5446",
    pages = "353--355"
}

# ImageNet
@article{imagenet,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = {{ImageNet Large Scale Visual Recognition Challenge}},
    Year = {2015},
    journal = {International Journal of Computer Vision (IJCV)},
    volume = {115},
    number = {3},
    pages = {211-252}
    }

# MNIST
@article{mnist,
  author = {LeCun, Yann and Cortes, Corinna},
  title = {{MNIST} handwritten digit database},
  url = {http://yann.lecun.com/exdb/mnist/},
  year = 2010
}

# MNLI
@InProceedings{N18-1101,
  author = "Williams, Adina
            and Nangia, Nikita
            and Bowman, Samuel",
  title = "A Broad-Coverage Challenge Corpus for
           Sentence Understanding through Inference",
  booktitle = "Proceedings of the 2018 Conference of
               the North American Chapter of the
               Association for Computational Linguistics:
               Human Language Technologies, Volume 1 (Long
               Papers)",
  year = "2018",
  publisher = "Association for Computational Linguistics",
  pages = "1112--1122",
  location = "New Orleans, Louisiana",
  url = "http://aclweb.org/anthology/N18-1101"
}

# Wikipedia Talk Toxicity Classification
@inproceedings{wikipedia_talk,
  author = {Wulczyn, Ellery and Thain, Nithum and Dixon, Lucas},
  title = {Ex Machina: Personal Attacks Seen at Scale},
  year = {2017},
  isbn = {9781450349130},
  publisher = {International World Wide Web Conferences Steering Committee},
  address = {Republic and Canton of Geneva, CHE},
  url = {https://doi.org/10.1145/3038912.3052591},
  doi = {10.1145/3038912.3052591},
  booktitle = {Proceedings of the 26th International Conference on World Wide Web},
  pages = {1391-1399},
  numpages = {9},
  keywords = {online discussions, wikipedia, online harassment},
  location = {Perth, Australia},
  series = {WWW '17}
}
```

### Models

```
# Residual Networks
@misc{resnet,
  title={Deep residual learning for image recognition. CoRR abs/1512.03385 (2015)},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year={2015}
}

# Criteo MLP
@inproceedings{uncertaintybenchmark,
  title={Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift},
  author={Snoek, Jasper and Ovadia, Yaniv and Fertig, Emily and Lakshminarayanan, Balaji and Nowozin, Sebastian and Sculley, D and Dillon, Joshua and Ren, Jie and Nado, Zachary},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13969--13980},
  year={2019}
}

# Text CNN
@inproceedings{textcnn,
    title = "Convolutional Neural Networks for Sentence Classification",
    author = "Kim, Yoon",
    booktitle = "Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing ({EMNLP})",
    month = oct,
    year = "2014",
    address = "Doha, Qatar",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D14-1181",
    doi = "10.3115/v1/D14-1181",
    pages = "1746--1751",
}

# BERT
@inproceedings{bert,
    title = "{BERT}: Pre-training of Deep Bidirectional Transformers for Language Understanding",
    author = "Devlin, Jacob  and
      Chang, Ming-Wei  and
      Lee, Kenton  and
      Toutanova, Kristina",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    address = "Minneapolis, Minnesota",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/N19-1423",
    doi = "10.18653/v1/N19-1423",
    pages = "4171--4186",
}

# Wide ResNet-*-*
@article{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}
```

## Contributors

Contributors (past and present):

* Dustin Tran
* Florian Wenzel
* Ghassen Jerfel
* Jeremiah Liu
* Jie Ren
* Marton Havasi
* Michael W. Dusenberry
* Rodolphe Jenatton
* Shreyas Padhy
* Yeming Wen
* Zachary Nado
