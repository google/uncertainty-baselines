# ImageNet

## ResNet-50

| Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Cal. Error | cNLL/cA/cCE | mCE | Train Runtime (hours) | Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- | ----------- |
| Deterministic | 0.900 / 0.943 | 77.9% / 76.1% | 0.0411 / 0.0392 | 3.22 / 40.3% / 0.104 | 75.6% | 5 (32 TPUv3 cores) | 1.60 (32 TPUv2 cores) | 25.6M |
| BatchEnsemble<sup>1</sup> | 0.861 / 0.944 | 78.9% / 76.7% | 0.0313 / 0.0494 | 3.18 / 41.8% / 0.110 | 73.7% | 17.5 (32 TPUv2 cores) | 8.33 (32 TPUv2 cores) | 25.8M |
| Monte Carlo Dropout (size=10) | 1.034 / 0.925 | 74.9% / 76.4% | 0.0417 / 0.0252 | 2.95 / 42.4% / 0.045 | 72.9% | 6 (32 TPUv3 cores) | 1.79 (32 TPUv2 cores) | 25.6M |
| SNGP | 0.955 / 0.937 | 77.1% / 76.0% | 0.0538 / 0.0138 | 3.06 / 40.9% / 0.050 | 75.0% | 5 (32 TPUv3 cores) | 1.74 (32 TPUv2 cores) | 25.6M |
| SNGP, with MC Dropout (size=10) | 0.846 / 0.913 | 79.7% / 76.6% | 0.0521 / 0.0197 | 3.05 / 41.2% / 0.058 | 74.5% | 5 (32 TPUv3 cores)  | 1.80 (32 TPUv3 cores) | 25.6M |
| SNGP Ensemble (size=4) | - / 0.851 | - / 78.1% | - / 0.0389 | 2.77 / 44.9% / 0.050 | 69.73% | 17.5 (128 TPUv2 cores) | 6.52 (32 TPUv2 cores) | 102.4M |
| Ensemble (size=4) | - / 0.877 | - / 77.5% | - / 0.0305 | 2.99 / 42.1% / 0.051 | 73.3% | 17.5 (128 TPUv2 cores) | 6.40 (32 TPUv2 cores) | 102.4M |

## EfficientNet

| Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Cal. Error | cNLL/cA/cCE | mCE | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic (B0) | - / 1.04 | - / 75.6% | - / 0.0497 | - | - | 5 (32 TPUv3 cores) | 5.3M |
| Deterministic (B1) | - / 1.00 | - / 77.2% | - | - | - | 6.5 (32 TPUv3 cores) | 7.8M |
| Deterministic (B2) | - / 0.973 | - / 78.0% | - | - | - | 9 (32 TPUv3 cores) | 9.2M |

## Metrics

We define metrics specific to ImageNet below. For general metrics, see [`baselines/`](https://github.com/google/edward2/tree/master/baselines).

1. __cNLL/cA/cCE__. Negative-log-likelihood, accuracy, and calibration error on [ImageNet-C](https://arxiv.org/abs/1903.12261). `c` stands for corrupted. Results take the mean across corruption intensities and corruption types.
2. __mCE__. Mean corruption error, which measures misclassification error across corruption intensities and corruption types on ImageNet-C. However, instead of taking the mean, it computes a weighted mean where weights are given by AlexNet's performance.

## Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Top-1 Accuracy | Train/Test Top-5 Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`facebookarchive/fb.resnet.torch `](https://github.com/facebookarchive/fb.resnet.torch ) | Deterministic | - | - / 75.99% | - / 92.98% | - | 25.6M |
| [`KaimingHe/deep-residual-networks`](https://github.com/KaimingHe/deep-residual-networks) | Deterministic | - | - / 75.3% | - | - | 25.6M |
| [`keras-team/keras`](https://keras.io/applications/#resnet) | Deterministic | - | - / 74.9% | - / 92.1% | - | 25.6M |
| | Deterministic (ResNet-152v2) | - | - / 78.0% | - / 94.2% | - | 60.3M |
| [`tensorflow/tpu`](https://github.com/tensorflow/tpu/tree/master/models/official/resnet)<sup>2</sup> | Deterministic | - | - / 76% | - | 17 (8 TPUv2) | 25.6M |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>3</sup> | Adaptive Thermostat Monte Carlo (single sample) | - / 1.08 | - / 74.2% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.883 | - / 77.5% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 1.15 | - / 73.1% | - | 1000 epochs (8 TPUv3 cores) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.941 | - / 76.4% | - | 1000 epochs (8 TPUv3 cores) | - |
| [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476)<sup>4</sup> | Deterministic (ResNet-152) | - / 0.8716 | - / 78.39% | - | pretrained+10 epochs | 60.3M |
| | SWA | - / 0.8682 | - / 78.92% | - | pretrained+10 epochs | 60.3M |
| | SWAG | - / 0.8205 | - / 79.08% | - | pretrained+10 epochs | 1.33B |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>5</sup>  | Variational Online Gauss-Newton | - / 1.37 | 73.87% / 67.38% | | 1.90 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530) | Deterministic | - | - | - | - | - |
| | Dropout | - | - | - | - | - |
| | Ensemble | - | - | - | - | - |
| [Zhang et al. (2019)](https://openreview.net/forum?id=rkeS1RVtPS)<sup>6</sup> | Deterministic (ResNet-50) | - / 0.960 | - / 76.046% | - /  92.78% | 25.6M |
| | cSGHMC | - / 0.888 | - / 77.11% | - / 93.524% | 307.2M |

1. Each ensemble member achieves roughly 75.6 test top-1 accuracy.
2. See documentation for differences from original paper, e.g., preprocessing.
3. Modifies architecture. Cyclical learning rate.
4. Uses ResNet-152. Training uses pre-trained SGD solutions. SWAG uses rank 20 which requires 20 + 2 copies of the model parameters, and 30 samples at test time.
5. Uses ResNet-18. Scales KL by an additional factor of 5.
6. cSGHMC uses a total of 9 copies of the full size of weights for prediction. The authors use a T=1/200 temperature scaling on the log-posterior (see the newly added appendix I at https://openreview.net/forum?id=rkeS1RVtPS)

TODO(trandustin): Add column for Checkpoints.
