# Uncertainty Baselines
Uncertainty Baselines is a set of common benchmarks for uncertainty calibration
and robustness research.

## Installation
WIP, but soon it will be available via `pip install`!

There is not yet a stable version (nor an official release of this library).
All APIs are subject to change.

## Usage
We access Uncertainty baselines via `import uncertainty_baselines as ub`. To
view a fully worked CIFAR-10 ResNet-20 example, see
`experiments/cifar10_resnet20/main.py`.

### Datasets
We implement datasets using the `tf.data.Dataset` API, available via the code
below:

```
dataset_builder = ub.datasets.Cifar10Dataset(
    batch_size=FLAGS.batch_size,
    eval_batch_size=FLAGS.eval_batch_size,
    validation_percent=0.1)  # Use 5000 validation images.
train_dataset = ub.utils.build_dataset(
    dataset_builder, strategy, 'train', as_tuple=True) # as_tuple for model.fit()
```

or via our getter method

```
dataset_builder = ub.datasets.get(
    dataset_name,
    batch_size=batch_size,
    eval_batch_size=eval_batch_size,
    **dataset_kwargs)
```

We support the following datasets:

- CIFAR-10
- CIFAR-100
- Clinc Intent Detection, [download](https://github.com/clinc/oos-eval/blob/master/data/data_full.json)
- Criteo Ads, [download](https://www.kaggle.com/c/criteo-display-ad-challenge/data)
- ImageNet
- Jigsaw Toxic Comment classification, [download](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)
- MNIST
- GLUE
- MNLI

#### Adding a dataset

To add a new dataset:

1. Add the bibtex reference to the `References` section below.
2. Add the dataset definition to the datasets/ dir. Every file should have a subclass of `datasets.base.BaseDataset`, which at a minimum requires implementing a constructor, `_read_examples`, and `_create_process_example_fn`.
3. Add a test that at a minimum constructs the dataset and checks the shapes of elements.
4. Add the dataset to `datasets/datasets.py` for easy access.
5. Add the dataset class to `datasets/__init__.py`.

### Models
We implement models using the `tf.keras.Model` API, available via the code
below:

```
model = ub.models.ResNet20Builder(batch_size=FLAGS.batch_size, l2_weight=None)
```

or via our getter method

```
model = ub.models.get(FLAGS.model_name, batch_size=FLAGS.batch_size)
```

We support the following models:

- ResNet-20 v1
- ResNet-50 v1
- Wide ResNet-*-*
- Criteo MLP
- Text CNN
- BERT

#### Adding a model

To add a new model:

1. Add the bibtex reference to the `References` section below.
2. Add the model definition to the models/ dir. Every file should have a `create_model` function with the following signature:
```
def create_model(
    batch_size: int,
    ...
    **unused_kwargs: Dict[str, Any])
    -> tf.keras.models.Model:
```

3. Add a test that at a minimum constructs the model and does a forward pass.
4. Add the model to `models/models.py` for easy access.
5. Add the `create_model` function to `models/__init__.py`.

## Experiments
The `experiments/` directory is for projects that use the codebase that the
authors believe others in the community will find usedul

## References

### Datasets
CIFAR10

```
@article{cifar10,
title = {CIFAR-10 (Canadian Institute for Advanced Research)},
author = {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
url = {http://www.cs.toronto.edu/~kriz/cifar.html},
}
```

CIFAR100

```
@article{cifar100,
title = {CIFAR-100 (Canadian Institute for Advanced Research)},
author = {Alex Krizhevsky and Vinod Nair and Geoffrey Hinton},
url = {http://www.cs.toronto.edu/~kriz/cifar.html},
}
```

CLINIC

```
@article{clinic,
  title = {An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction},
  author = {Larson, Stefan and Mahendran, Anish and Peper, Joseph J and Clarke, Christopher and Lee, Andrew and Hill, Parker and Kummerfeld, Jonathan K and Leach, Kevin and Laurenzano, Michael A and Tang, Lingjia and others},
  journal = {arXiv preprint arXiv:1909.02027},
  year = {2019}
}
```

Criteo

```
@article{criteo,
title = {Display Advertising Challenge},
url = {https://www.kaggle.com/c/criteo-display-ad-challenge.},
}
```

ImageNet

```
@article{imagenet,
    Author = {Olga Russakovsky and Jia Deng and Hao Su and Jonathan Krause and Sanjeev Satheesh and Sean Ma and Zhiheng Huang and Andrej Karpathy and Aditya Khosla and Michael Bernstein and Alexander C. Berg and Li Fei-Fei},
    Title = {{ImageNet Large Scale Visual Recognition Challenge}},
    Year = {2015},
    journal = {International Journal of Computer Vision (IJCV)},
    volume = {115},
    number = {3},
    pages = {211-252}
    }
```

Jigsaw

```
@inproceedings{jigsaw,
    title = "Challenges for Toxic Comment Classification: An In-Depth Error Analysis",
    author = {van Aken, Betty  and
      Risch, Julian  and
      Krestel, Ralf  and
      L{\"o}ser, Alexander},
    booktitle = "Proceedings of the 2nd Workshop on Abusive Language Online ({ALW}2)",
    month = oct,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/W18-5105",
    doi = "10.18653/v1/W18-5105",
    pages = "33--42",
}
```

MNIST

```
@article{mnist,
  author = {LeCun, Yann and Cortes, Corinna},
  title = {{MNIST} handwritten digit database},
  url = {http://yann.lecun.com/exdb/mnist/},
  year = 2010
}
```

GLUE

```
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
```

MNLI

```
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
```

### Models
ResNet-20, ResNet-50

```
@misc{resnet,
  title={Deep residual learning for image recognition. CoRR abs/1512.03385 (2015)},
  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  year={2015}
}
```

Criteo MLP

```
@inproceedings{uncertaintybenchmark,
  title={Can you trust your model's uncertainty? Evaluating predictive uncertainty under dataset shift},
  author={Snoek, Jasper and Ovadia, Yaniv and Fertig, Emily and Lakshminarayanan, Balaji and Nowozin, Sebastian and Sculley, D and Dillon, Joshua and Ren, Jie and Nado, Zachary},
  booktitle={Advances in Neural Information Processing Systems},
  pages={13969--13980},
  year={2019}
}
```

Text CNN

```
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
```

BERT

```
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
```

Wide ResNet-*-*

```
@article{zagoruyko2016wide,
  title={Wide residual networks},
  author={Zagoruyko, Sergey and Komodakis, Nikos},
  journal={arXiv preprint arXiv:1605.07146},
  year={2016}
}
```


## Contributors

Contributors (past and present):

*   Dustin Tran
*   Jeremiah Liu
*   Zachary Nado
