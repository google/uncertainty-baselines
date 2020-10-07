# Wide ResNet 28-10 on CIFAR

## CIFAR-10

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | cNLL/cA/cCE | Train Runtime (hours) | Wall-clock/Compute Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- |
| Deterministic | 1e-3 / 0.159 | 99.9% / 96.0% | 1e-3 / 0.0231 | 1.05 / 76.1% / 0.153 | 1.2 (8 TPUv2 cores) | 5.17 / 0.079 (8 TPUv2 cores) | 36.5M |
| BatchEnsemble (size=4) | 0.08 / 0.136 | 99.9% / 96.3% |  5e-5 / 0.0177 | 0.97 / 77.8% / 0.124 | 5.4 (8 TPUv2 cores) | 11.9 / 0.319 (8 TPUv2 cores) | 36.6M |
| Hyper-BatchEnsemble (size=4) | 0.001 / 0.126 | 100% / 96.3% | 0.001 / 0.009  |- |  - |  - | 73.1M |
| MIMO | - / 0.123 | - / 96.4% | - / 0.010 | 0.927 / 76.6% / 0.112 | - | - / 0.080 | 36.5M |
| Rank-1 BNN (Gaussian, size=4) | - / 0.128 | - / 96.3% |  - / 0.008 | 0.84 / 76.7% / 0.080 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| Rank-1 BNN (Cauchy, size=4, 4 samples) | - / 0.120 | - / 96.5% |  - / 0.009 | 0.74 / 80.5% / 0.090 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| SNGP | 1e-3 / 0.134 | 99.9% / 96.0% |  5e-4 / 0.008 | 0.75 / 78.2% / 0.076 | 2.6 (8 TPUv2 cores) | 6.2 / 0.378 (8 TPUv2 cores) | 36.5M |
| SNGP, with AugMix| 1e-3 / 0.103 | 94.9% / 96.9% |  0.0038 / 0.0045 | 0.33 / 89.1% / 0.015 | 2.6 (8 TPUv2 cores) | 6.2 / 0.378 (8 TPUv2 cores) | 36.5M |
| SNGP, with MC Dropout (size=10) | 1e-3 / 0.131 | 99.9% / 95.9% |  7e-4 / 0.008 | 0.76 / 77.7% / 0.082 | 4.5 (8 TPUv2 cores) | 7.6 / 2.846 (8 TPUv2 cores) | 36.5M |
| SNGP, with BatchEnsemble (size=4) | - / 0.127 | -% / 96.2% |  - / 0.006 | 0.75 / 78.1% / 0.080 | - (8 TPUv2 cores) | - / - (8 TPUv2 cores) | -M |
| SNGP Ensemble (size=4) | 1e-3 / 0.109 | 99.9% / 96.7% |  5e-4 / 0.005 | 0.72 / 79.2% / 0.074 | 2.6 (32 TPUv2 cores) | 24.5 / 1.199 (8 TPUv2 cores) | 146M |
| Monte Carlo Dropout (size=1) | 2e-3 / 0.160 | 99.9% / 95.9% | 2e-3 / 0.0241 | 1.27 / 68.8% / 0.166 | 1.2 (8 TPUv2 cores) | 4.7 / 0.082 (8 TPUv2 cores) | 36.5M |
| Monte Carlo Dropout (size=30) | 1e-3 / 0.145 | 99.9% / 96.1% | 1.5e-3 / 0.019 | 1.27 / 70.0% / 0.167 | 1.2 (8 TPUv2 cores) | 19.1 / 2.457  (8 TPUv2 cores) | 36.5M |
| Monte Carlo Dropout, improved (size=30)<sup>11</sup> | 1e-3 / 0.115 | 99.9% / 96.4% | 1e-3 / 0.006 | 0.75 / 79.3% / 0.075 | 4.8 (8 TPUv2 cores) | 19.6 / 2.387  (8 TPUv2 cores)  | 36.5M |
| Ensemble (size=4) | 2e-3 / 0.114 | 99.9% / 96.6% | - / 0.010 | 0.81 / 77.9% / 0.087 | 1.2 (32 TPUv2 cores) | 20.7 / 0.317  (8 TPUv2 cores) | 146M |
| Variational inference (sample=1) | 1e-3 / 0.211 | 99.9% / 94.7% | 1e-3 / 0.029 | 1.46 / 71.3% / 0.181 | 5.5 (8 TPUv2 cores) | 4.5 / 0.220 (8 TPUv2 cores) | 73M |

## CIFAR-100

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | cNLL/cA/cCE | Train Runtime (hours) | Wall-clock/Compute Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- |
| Deterministic<sup>10</sup> | 1e-3 / 0.875 | 99.9% / 79.8% | 2e-3 / 0.0857 | 2.70 / 51.37% / 0.239 | 1.1 (8 TPUv2 cores) | 4.6 / 0.079 (8 TPUv2 cores) | 36.5M |
| BatchEnsemble (size=4) | 3e-3 / 0.690 | 99.7% / 81.9% | 2e-3 / 0.0265 | 2.56 / 53.1% / 0.149 | 5.5 (8 TPUv2 cores) | 13.7 / 0.319 (8 TPUv2 cores) | 36.6M |
| Hyper-BatchEnsemble (size=4) | 0.005 / 0.685 | 99.9% / 81.9% | 0.005 / 0.022  | - | - | - |  | 73.2M |
| MIMO | - / 0.690 | - / 82.0% | - / 0.022 | 2.28 / 53.7% / 0.129 | - | - / 0.080 | 36.5M |
| Rank-1 BNN (Gaussian, size=4) | - / 0.692 | - / 81.3% |  - / 0.018 | 2.24 / 53.8% / 0.117 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| Rank-1 BNN (Cauchy, size=4, 4 samples) | - / 0.689 | - / 82.4% |  - / 0.012 | 2.04 / 57.8% / 0.142 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| SNGP | 7e-3 / 0.805 | 99.9% / 79.6% |  5e-4 / 0.024 | 2.13 / 53.8% / 0.098 | 2.6 (8 TPUv2 cores) | 5.8 / 0.378 (8 TPUv2 cores) | 36.5M |
| SNGP, with AugMix | 0.68 / 0.755 | 83.8% / 80.6% |  0.07 / 0.024 | 1.44 / 65.9% / 0.054 | 2.6 (8 TPUv2 cores) | 5.8 / 0.378 (8 TPUv2 cores) | 36.5M |
| SNGP, with MC Dropout (size=10) | 2e-2 / 0.750 | 99.9% / 79.6% | 1e-2 / 0.017 | 2.06 / 53.8% / 0.087 | 4.5 (8 TPUv2 cores) | 6.7 / 2.841 (8 TPUv2 cores) | 36.5M |
| SNGP, with BatchEnsemble (size=4) | - / 0.755 | -% / 81.4% |  - / 0.032 | 2.03 / 55.2% / 0.112 | - (8 TPUv2 cores) | - / - (8 TPUv2 cores) | -M |
| SNGP Ensemble (size=4) | 7e-3 / 0.665 | 99.9% / 81.9% | 5e-4 / 0.011 | 1.95 / 56.8% / 0.091 | 2.6 (32 TPUv2 cores) | 23.32 / 1.198 (8 TPUv2 cores) | 146M |
| Monte Carlo Dropout (size=1) | 1e-2 / 0.830 | 99.9% / 79.6% | 9e-3 / 0.0501 | 2.90 / 42.63% / 0.202 | 1.1 (8 TPUv2 cores) | 4.8 / 0.082 (8 TPUv2 cores) | 36.5M |
| Monte Carlo Dropout (size=30) | 6e-3 / 0.785 | 99.9% / 80.7% | 5e-3 / 0.0487 | 2.73 / 46.2 / 0.207 | 1.1 (8 TPUv2 cores) | 19.5 / 2.457 (8 TPUv2 cores) | 36.5M |
| Monte Carlo Dropout, improved (size=30)<sup>11</sup> | 1e-2 / 0.649 | 99.9% / 82.2% | 8e-3 / 0.0165 | 2.09 / 56.2% / 0.110 | 5.17 (8 TPUv2 cores) | 19.6 / 2.393 (8 TPUv2 cores) |  36.5M |
| Ensemble (size=4) | 0.003 / 0.666 | 99.9% / 82.7% | - / 0.021 | 2.27 / 54.1% / 0.138 | 1.1 (32 TPUv2 cores) | 20.1 / 0.317 (8 TPUv2 cores) |  146M |
| Variational inference (sample=1) | 3e-3 / 0.944 | 99.9% / 77.8% | 2e-3 / 0.097 | 3.18 / 48.2% / 0.271 | 5.5 (8 TPUv2 cores) | 4.69 / 0.210 (8 TPUv2 cores) | 73M |

## Metrics

We define metrics specific to CIFAR below. For general metrics, see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __cNLL/cA/cCE__. Negative-log-likelihood, accuracy, and calibration error on [CIFAR-10-C](https://arxiv.org/abs/1903.12261); we apply the same corruptions to produce a CIFAR-100-C. `c` stands for corrupted. Results take the mean across corruption intensities and corruption types.

## CIFAR-10 Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`meliketoy/wide-resnet.pytorch`](https://github.com/meliketoy/wide-resnet.pytorch) | Deterministic | - | - / 96.21% | 6.8 (1 GPU) | 36.5M |
| | Non-MC Dropout | - | - / 96.27% | 6.8 (1 GPU)  | 36.5M |
| [He et al. (2015)](https://arxiv.org/abs/1512.03385)<sup>1</sup> | Deterministic (ResNet-56) | - | - / 93.03% | - | 850K |
| | Deterministic (ResNet-110) | - | - / 93.39% | - | 1.7M |
| [Zagoruyko and Komodakis (2016)](https://github.com/szagoruyko/wide-residual-networks) | Deterministic | - | - / 96.00% | - | 36.5M |
| | Non-MC Dropout | - | - / 96.11% | - | 36.5M |
| [Louizos et al. (2017)](https://arxiv.org/abs/1705.08665)<sup>2</sup> | Group-normal Jeffreys | - | - / 91.2% | - | 998K |
| | Group-Horseshoe | - | - / 91.0% | - | 820K |
| [Molchanov et al. (2017)](https://arxiv.org/abs/1701.05369)<sup>2</sup> | Variational dropout | - | - / 92.7% | - | 304K |
| [Louizos et al. (2018)](https://arxiv.org/abs/1712.01312) | L0 regularization | - | - / 96.17% | 200 epochs | - |
| [Havasi et al. (2019)](https://openreview.net/forum?id=rkglZyHtvH)<sup>3</sup> | Refined VI (no batchnorm) | - / 0.696 | - / 75.5% | 5.5 (1 P100 GPU) | - |
| | Refined VI (batchnorm) | - / 0.593 | - / 79.7% | 5.5 (1 P100 GPU) | - |
| | Refined VI hybrid (no batchnorm) | - / 0.432 | - / 85.8% | 4.5 (1 P100 GPU) | - |
| | Refined VI hybrid (batchnorm) | - / 0.423 | - / 85.6% | 4.5 (1 P100 GPU) | - |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>5</sup> | Deterministic (ResNet-56) | - / 0.243 | - / 94.4% | 1000 epochs (1 V100 GPU) | 850K |
| | Adaptive Thermostat Monte Carlo (single sample) | - / 0.303 | - / 92.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | - / 0.194 | - / 93.9% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | - / 0.343 | - / 91.7% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | - / 0.211 | - / 93.5% | 1000 epochs (1 V100 GPU) | - |
| [Izmailov et al. (2019)](https://arxiv.org/abs/1907.07504)<sup>6</sup> | Deterministic | - / 0.1294 | - / 96.41% | 300 epochs | 36.5M |
| | SWA | - / 0.1075 | - / 96.46% | 300 epochs | 36.5M |
| | SWAG | - / 0.1122 | - / 96.41% | 300 epochs | 803M |
| | Subspace Inference (PCA+VI) | - / 0.1081 | - / 96.32% | >300 epochs | 219M |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>7</sup>  | Variational Online Gauss-Newton | - / 0.48 | 91.6% / 84.3% | 2.38 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530)<sup>8</sup> | Deterministic (ResNet-20) | - / 1.120 | - / 91% | - | 274K |
| | Dropout | - / 0.771 | - / 91% | - | 274K |
| | Ensemble | - / 0.653 | - | - / 93.5% | - |
| | Variational inference | - / 0.823 | 88% | - | 630K |
| [Wen et al. (2019)](https://openreview.net/forum?id=Sklf1yrYDr)<sup>4</sup> | Deterministic (ResNet-32x4) | - | - / 95.31% | 250 epochs | 7.43M |
| | BatchEnsemble | - | - / 95.94% | 375 epochs | 7.47M |
| | Ensemble | - | - / 96.30% | 250 epochs each | 29.7M |
| | Monte Carlo Dropout | - | - / 95.72% | 375 epochs | 7.43M |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>9</sup> | Deterministic (ResNet-18) | - | - / 94.71% | 200 epochs | 11.7M |
| | cSGHMC | - | - / 95.73% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | - / 96.05% | 800 epochs | 561.6M |

## CIFAR-100 Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Train/Test NLL | Train/Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`meliketoy/wide-resnet.pytorch`](https://github.com/meliketoy/wide-resnet.pytorch) | Deterministic | - | - / 81.02% | 6.8 (1 GPU) | 36.5M |
| | Non-MC Dropout | - | - / 81.49% | 6.8 (1 GPU)  | 36.5M |
| [Zagoruyko and Komodakis (2016)](https://github.com/szagoruyko/wide-residual-networks) | Deterministic | - | - / 80.75% | - | 36.5M |
| | Non-MC Dropout | - | - / 81.15% | - | 36.5M |
| [Izmailov et al. (2019)](https://arxiv.org/abs/1907.07504)<sup>7</sup> | Deterministic | - / 0.7958 | - / 80.76% | 300 epochs | 36.5M |
| | SWA | - / 0.6684 | - / 82.40% | 300 epochs | 36.5M |
| | SWAG | - / 0.6078 | - / 82.23% | 300 epochs | 803M |
| | Subspace Inference (PCA+VI) | - / 0.6052 | - / 82.63% | >300 epochs | 36.5M |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>10</sup> | Deterministic (ResNet-18) | - | - / 77.40% | 200 epochs | 11.7M |
| | cSGHMC | - | - / 79.50% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | - / 80.81% | 800 epochs | 561.6M |

1. Trains on 45k examples.
2. Not a ResNet (VGG). Parameter count is guestimated from counting number of parameters in [original model](http://torch.ch/blog/2015/07/30/cifar.html) to be 14.9M multiplied by the compression rate.
3. Uses ResNet-20. Does not use data augmentation.
4. Uses ResNet-32 with 4x number of typical filters. Ensembles uses 4 members.
5. Uses ResNet-56 and modifies architecture. Cyclical learning rate.
6. SWAG with rank 20 requires 20 + 2 copies of the model parameters and uses 30 samples at test time. Deterministic baseline is reported in [Maddox et al. (2019)](https://arxiv.org/abs/1902.02476). Subspace inference with a rank 5 projection requires a total of 5 + 1 copies of the model parameters at test time (the posterior inference parameters are negligible). Subspace inference also uses a fixed temperature.
7. ResNet-20. Scales KL by an additional factor of 10.
8. ResNet-20. Trains on 40k examples. Performs variational inference over only first convolutional layer of every residual block and final output layer. Has free parameter on normal prior's location. Uses scale hyperprior (and with a fixed scale parameter). NLL results are medians, not means; accuracies are guestimated from Figure 2's plot.
9. Uses ResNet-18. cSGHMC uses a total of 12 copies of the full size of weights for prediction. Ensembles use 4 times cSGHMC's number. The authors use a T=1/200 temperature scaling on the log-posterior (see the newly added appendix I at https://openreview.net/forum?id=rkeS1RVtPS).
10. Results are slightly worse than open-source implementations such as the [original paper](https://github.com/szagoruyko/wide-residual-networks)'s 80.75%. Our experiments only tuned over l2, so there may be more work to be done.
11. MC dropout can be significantly improved using three techniques. a, Using dropout after every layer instead of only in the residual connections. b, Using filterwise dropout in the convolutional layers instead of dropout for each individual value in the feature map. c, Repeating examples in the training batch, although this comes at an increased training cost.

TODO(trandustin): Add column for Checkpoints.
