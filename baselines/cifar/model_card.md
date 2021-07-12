# Model Card: Diabetic Retinopathy Baseline

Below we give a summary of the models trained on the CIFAR-10 and CIFAR-100 datasets, following the guidance of [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).

## Model Details

The models covered by this card were trained to classify the label of images.

### Model Training Date

November 2020

### Model Types

All the models in this baseline use a backbone Wide ResNet 28-10 model. For hyper-deep ensembles we additionally use a backbone ResNet-20 model.

## Model Use

### Intended Use

These models are intended for research purposes, primarily for developing better calibration or out-of-distribution detection methods. They serve as easily-reproducible and well-documented baselines for comparisons in research papers.

### Out-of-Scope Use Cases

These models are not intended to be used in any user-facing or production settings.

## Data

The data used for these models are the publicly availabe datasets [CIFAR-10](https://www.tensorflow.org/datasets/catalog/cifar10), [CIFAR-100](https://www.tensorflow.org/datasets/catalog/cifar100). To measure the robustness of the methods we use [CIFAR-10-C](https://www.tensorflow.org/datasets/catalog/cifar10_corrupted) where corruptions were applied to the examples of CIFAR-10. We apply the same corruptions on CIFAR-100 to produce a CIFAR-100-C.

## Performance and Limitations

### Performance

All models released were selected for their validation accuracy, with final numbers on the combined train/validation and test sets.

### Limitations

The models have only been trained and evaluated on the CIFAR-10 and CIFAR-100 datasets and their corrupted variants CIFAR-10-C and CIFAR-100-C, so they should not be used with any other data distributions. Additionally, while they reach competitive performance on these datasets, they are not perfect.

### Bias and Fairness

Our models were not tested with respect to bias and fairness.

## Feedback

Please open a [Pull Request](https://github.com/google/uncertainty-baselines/pulls) to contact the maintainers!

