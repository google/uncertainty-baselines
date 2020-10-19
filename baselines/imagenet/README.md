# ImageNet

## ResNet-50

| Method | ImageNet | ImageNet-C | ImageNet-A | ImageNetV2 | ImageNet-Vid-Robust | YTBB-Robust | ObjectNet | Train Runtime (hours) | Wall-clock/Compute Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.939 / 76.2% / 0.032 | 3.21 / 40.5% / 0.103 (75.4%) | 8.09 / 0.7% / 0.425 | 1.58 / 64.4% / 0.074 | 29.9% | 21.7% | 25.9% | 5 (32 TPUv2 cores) | 1.60 / 0.020 (32 TPUv2 cores) | 25.6M |
| BatchEnsemble | 0.922 / 76.8% / 0.037 | 3.09 / 41.9% / 0.089 (73.5%) | - | - | - | - | - | 17.5 (32 TPUv2 cores) | 8.33 / 0.081 (32 TPUv2 cores) | 25.8M |
| MIMO | 0.887 / 77.5% / 0.037 | 3.03 / 43.3% / 0.106 (71.7%) | 7.76 / 1.4% / 0.432 | 1.51 / 65.7% / 0.084 | 31.8% | 22.2% | 28.1% | - | - | 27.7M |
| Rank-1 BNN (Gaussian, size=4) | 0.886 / 77.3% / 0.0166 | 2.95 / 42.9% / 0.054 (72.12%) | - | - | - | - | - | 21.1 (32 TPUv2 cores) | - | 26.0M |
| Rank-1 BNN (Cauchy, size=4, 4 samples) | 0.897 / 77.2% / 0.0192 | 2.98 / 42.5 / 0.059 (72.66%) | - | - | - | - | - | 21.1 (32 TPUv2 cores) | - | 26.0M |
| Monte Carlo Dropout (size=10)  | 0.919 / 76.6% / 0.026 | 2.96 / 42.4% / 0.046 (72.9%) | 7.58 / 0.4% / 0.344 | 1.53 / 64.6% / 0.021 | 31.0% | 23.1% | 26.2% | 6 (32 TPUv2 cores) | 1.79 / 0.205 (32 TPUv2 cores) | 25.6M |
| SNGP | 0.937 / 76.0% / 0.014 | 3.06 / 40.9% / 0.050 (75.0%) | 7.18 / 0.75% / 0.364 | 1.53 / 64.1% / 0.045 | 29.7% | 21.1% | 26.1% | 5 (32 TPUv3 cores) | 1.74 / 0.017 (32 TPUv3 cores) | 25.6M |
| SNGP, with MC Dropout (size=10) | 0.913 / 76.6% / 0.020 | 3.05 / 41.2% / 0.058 (74.5%) | - | - | - | - | - | 5 (32 TPUv3 cores) | 1.80 / 0.126 (32 TPUv3 cores) | 25.6M |
| SNGP, with BatchEnsemble (size=4) | 0.913 / 76.55% / 0.014 | 3.08 / 40.58% / 0.047 (75.2%) | - | - | - | - | - | - (32 TPUv3 cores) | - / - (32 TPUv3 cores) | - |
| SNGP Ensemble (size=4) | 0.851 / 78.1% / 0.039 | 2.77 / 44.9% / 0.050 (69.73%) | - | - | - | - | - | 17.5 (128 TPUv3 cores) | 6.52 / 0.055 (32 TPUv3 cores) | 102.4M |
| Ensemble (size=4) | 0.877 / 77.5% / 0.031 | 2.99 / 42.1% / 0.051 (73.3%) | - | - | - | - | - | 17.5 (128 TPUv2 cores) | 6.40 / 0.082 (32 TPUv2 cores) | 102.4M |

## EfficientNet

| Method | ImageNet | ImageNet-C | ImageNet-A | ImageNetV2 | ImageNet-Vid-Robust | YTBB-Robust | ObjectNet | Train Runtime (hours) | Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic (B0) | 1.04 / 75.6% / 0.497 | - | - | - | - | - | - | 5 (32 TPUv3 cores) | - | 5.3M |
| Deterministic (B1) | 1.00 / 77.2% / - | - | - | - | - | - | - | 6.5 (32 TPUv3 cores) | - | 7.8M |
| Deterministic (B2) | 0.973 / 78.0% / - | - | - | - | - | - | - | 9 (32 TPUv3 cores) | - | 9.2M |

## Metrics

We define metrics specific to ImageNet below. For general metrics, see [the top-level `README.md`](https://github.com/google/uncertainty-baselines).

1. __ImageNet__. Negative-log-likelihood, (top-1) accuracy, and calibration error on the ImageNet test set.
2. __ImageNet-C__. Negative-log-likelihood, (top-1) accuracy, and calibration error on [ImageNet-C](https://arxiv.org/abs/1903.12261). Results take the mean across corruption intensities and corruption types. Parentheses denotes mean corruption error, which measures misclassification error and instead of taking the mean, it computes a weighted mean with weights given by AlexNet's performance.
4. __ImageNet-A.__ Negative-log-likelihood, (top-1) accuracy, and calibration error on [ImageNet-A](https://arxiv.org/abs/1907.07174).
5. __ImageNetV2.__ Negative-log-likelihood, (top-1) accuracy, and calibration error on the matched frequency variant of [ImageNetV2](https://arxiv.org/abs/1902.10811).
6. __ImageNet-Vid-Robust.__ Accuracy (pm-k) on [ImageNet-Vid-Robust](https://arxiv.org/abs/1906.02168).
7. __YTBB-Robust.__ Accuracy (pm-k) on [YTBB-Robust](https://arxiv.org/abs/1906.02168).
8. __ObjectNet.__ Accuracy (top-1) on [ObjectNet](https://papers.nips.cc/paper/9142-objectnet-a-large-scale-bias-controlled-dataset-for-pushing-the-limits-of-object-recognition-models.pdf).

## Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | ImageNet | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- |
| [`facebookarchive/fb.resnet.torch `](https://github.com/facebookarchive/fb.resnet.torch ) | Deterministic | - / 75.99% / - | - | 25.6M |
| [`KaimingHe/deep-residual-networks`](https://github.com/KaimingHe/deep-residual-networks) | Deterministic | - / 75.3% / - | - | 25.6M |
| [`keras-team/keras`](https://keras.io/applications/#resnet) | Deterministic | - / 74.9% / - | - | 25.6M |
| | Deterministic (ResNet-152v2) | - / 78.0% / - | - | 60.3M |
| [`tensorflow/tpu`](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)<sup>1</sup> | Deterministic | - / 76% / - | 17 (8 TPUv2) | 25.6M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>2</sup> | Adaptive Thermostat Monte Carlo (single sample) | 1.08 / 74.2% / - | 1000 epochs (8 TPUv3 cores) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | 0.883 / 77.5% / - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | 1.15 / 73.1% / - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | 0.941 / 76.4% / - | 1000 epochs (8 TPUv3 cores) | - |
| [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476)<sup>3</sup> | Deterministic (ResNet-152) | 0.8716 / 78.39% / - | pretrained+10 epochs | 60.3M |
| | SWA | 0.8682 / 78.92% / - | pretrained+10 epochs | 60.3M |
| | SWAG | 0.8205 / 79.08% / - | pretrained+10 epochs | 1.33B |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>4</sup>  | Variational Online Gauss-Newton | 1.37 / 67.38% / - | 1.90 (128 P100 GPUs) | - |
| [Zhang et al. (2019)](https://openreview.net/forum?id=rkeS1RVtPS)<sup>5</sup> | Deterministic (ResNet-50) | 0.960 / 76.046% / - | - | 25.6M |
| | cSGHMC | 0.888 / 77.11% / 93.524% | - | 307.2M |

1. See documentation for differences from original paper, e.g., preprocessing.
2. Modifies architecture. Cyclical learning rate.
3. Uses ResNet-152. Training uses pre-trained SGD solutions. SWAG uses rank 20 which requires 20 + 2 copies of the model parameters, and 30 samples at test time.
4. Uses ResNet-18. Scales KL by an additional factor of 5.
5. cSGHMC uses a total of 9 copies of the full size of weights for prediction. The authors use a T=1/200 temperature scaling on the log-posterior (see the newly added appendix I at https://openreview.net/forum?id=rkeS1RVtPS)

TODO(trandustin): Add column for Checkpoints.
