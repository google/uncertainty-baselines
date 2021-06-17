# Model Card: Diabetic Retinopathy Baseline

Below we give a summary of the models trained on the Diabetic Retinopathy Deterministic dataset, following the guidance of [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).

## Model Details

The models covered by this card were trained to classify whether or not images of blood vessels in the eye indicate a patient has diabetic retinopathy.

### Model Training Date

March 2021

### Model Types

All the models in this baseline use a backbone ResNet-50 v1.5 model, with slight modifications described below:

- [Monte-Carlo Dropout](https://arxiv.org/abs/1506.02142): we run test images through the model `FLAGS.num_dropout_samples_eval` times, each with a different dropout mask, to do approximate Bayesian inference.
- [Radial Bayesian Neural Net](https://arxiv.org/abs/1907.00865): we use a Radial Posterior as a variational approximate posterior.
- Variational Inference: we perform variational inference on the model weights.

## Model Use

### Intended Use

These models are intended for research purposes, primarily for developing better calibration or out-of-distribution detection methods. They serve as easily-reproducible and well-documented baselines for comparisons in research papers.

### Out-of-Scope Use Cases

These models are not intended to be used in any user-facing or production settings.

## Data

The data used for these models is from the [Kaggle Diabetic Retinopathy Detection challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data), hosted on [Tensorflow Datasets](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection).

## Performance and Limitations

### Performance

All models released were selected for their validation AUC, with final numbers on the combined train/validation and test sets released on [tensorboard.dev](tensorboard.dev). All models reach between 0.8-0.9 AUC on the test split.

### Limitations

The models have only been trained and evaluated on the Kaggle Diabetic Retinopathy Detection challenge dataset, so they should not be used with any other data distributions. Additionally, while they reach competitive AUC performance on this dataset, they are not perfect.

### Bias and Fairness

There are always ethical considerations to keep in mind for any medical dataset, such as the conscious or unconscious biases of the people performing data collection and labelling. There was never any patient data associated with this dataset, so we cannot test for whether or not any of these issues are present in these models.

## Feedback

Please open a [Pull Request](https://github.com/google/uncertainty-baselines/pulls) to contact the maintainers!

