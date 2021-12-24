# Model Card: Diabetic Retinopathy Baseline

Below we give a summary of the models trained on the Diabetic Retinopathy Detection dataset, following the guidance of [Model Cards for Model Reporting (Mitchell et al.)](https://arxiv.org/abs/1810.03993).

## Model Details

The models covered by this card were trained to classify whether or not images of blood vessels in the eye indicate a patient has diabetic retinopathy.

### Model Training Date

August 2021

### Model Types

All the models in this baseline use a backbone ResNet-50 v1.5 model, with implementation and modification details provided in Section 5 of the benchmark whitepaper located [here](https://openreview.net/forum?id=jyd4Lyjr2iB). 

## Model Use

### Intended Use

These models are intended for research purposes, primarily for developing better calibration or out-of-distribution detection methods. They serve as easily-reproducible and well-documented baselines for comparisons in research papers.

### Out-of-Scope Use Cases

These models are not intended to be used in any user-facing or production settings.

## Data

The data used for these models is from the [Kaggle Diabetic Retinopathy Detection](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and the [Kaggle APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection) challenges.
 
See `uncertainty-baselines/uncertainty_baselines/datasets` for dataloading utilites.

The images from either competition may be downloaded for free with a Kaggle account at the links above.

## Performance and Limitations

### Performance

Models were selected on one of two possible validation metrics:
* In-domain validation AUC. 
* Area under the "balanced" accuracy referral curve, constructed on in-domain and distributionally shifted data.

See Appendix B.3 [whitepaper](https://openreview.net/forum?id=jyd4Lyjr2iB) for more details on the latter metric.

### Limitations

No single set of benchmarking tasks is a panacea.

We hope that the tasks, evaluation methods, and uncertainty quantification methods presented in this benchmarking suite will significantly lower the barrier for assessing the reliability of Bayesian deep learning methods on safety-critical real-world prediction tasks.

We further discuss limitations of these models (and this benchmark in general) in the [whitepaper](https://openreview.net/forum?id=jyd4Lyjr2iB). 

### Bias and Fairness

There are always ethical considerations to keep in mind for any medical dataset, such as the conscious or unconscious biases of the people performing data collection and labelling. There was never any patient data associated with this dataset, so we cannot test for whether or not any of these issues are present in these models.

## Feedback

Please open a [Pull Request](https://github.com/google/uncertainty-baselines/pulls) to contact the maintainers!
