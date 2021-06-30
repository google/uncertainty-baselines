# Wide ResNet 28-10 on CIFAR

## CIFAR-10

| Method | Test NLL | Test Accuracy | Test Cal. Error | cNLL/cA/cCE | Train Runtime (hours) | Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- |
| [Deterministic](deterministic.py) | 0.159 | 96.0% | 0.0231 | 1.05 / 76.1% / 0.153 | 1.2 (8 TPUv2 cores) | 0.079 (8 TPUv2 cores) | 36.5M |
| [BatchEnsemble (size=4)](batchensemble.py) | 0.136 | 96.3% | 0.0177 | 0.97 / 77.8% / 0.124 | 5.4 (8 TPUv2 cores) | 0.319 (8 TPUv2 cores) | 36.6M |
| [Hyper-BatchEnsemble (size=4)](hyperbatchensemble.py) | 0.126 | 96.3% | 0.009  |- |  - |  - | 73.1M |
| [MIMO](mimo.py) | 0.123 | 96.4% | 0.010 | 0.927 / 76.6% / 0.112 | - | 0.080 | 36.5M |
| [Rank-1 BNN (Gaussian, size=4)](rank1_bnn.py) | 0.128 | 96.3% |  0.008 | 0.84 / 76.7% / 0.080 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| [Rank-1 BNN (Cauchy, size=4, 4 samples)](rank1_bnn.py) | [0.120 | 96.5% |  0.009 | 0.74 / 80.5% / 0.090 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| [SNGP](sngp.py) | 0.134 | 96.0% | 0.007 | 0.74 / 78.5% / 0.078 | 2.6 (8 TPUv2 cores) | 0.378 (8 TPUv2 cores) | 36.5M |
| [SNGP, with AugMix](sngp.py)| 0.103 | 96.9% | 0.0045 | 0.33 / 89.1% / 0.015 | 2.6 (8 TPUv2 cores) | 0.378 (8 TPUv2 cores) | 36.5M |
| [SNGP, with MC Dropout (size=10)](sngp.py) | 0.131 | 95.9% | 0.008 | 0.76 / 77.7% / 0.082 | 4.5 (8 TPUv2 cores) | 2.846 (8 TPUv2 cores) | 36.5M |
| [SNGP, with BatchEnsemble (size=4)](sngp_batchensemble.py) | 0.127 | 96.2% | 0.006 | 0.75 / 78.1% / 0.080 | - (8 TPUv2 cores) | - | -M |
| [SNGP Ensemble (size=4)](sngp_ensemble.py) | 0.109 | 96.7% | 0.005 | 0.72 / 79.2% / 0.074 | 2.6 (32 TPUv2 cores) | 1.199 (8 TPUv2 cores) | 146M |
| [Monte Carlo Dropout (size=1)](dropout.py) |0.160 | 95.9% | 0.0241 | 1.27 / 68.8% / 0.166 | 1.2 (8 TPUv2 cores) | 0.082 (8 TPUv2 cores) | 36.5M |
| [Monte Carlo Dropout (size=30)](dropout.py) | 0.145 | 96.1% | 0.019 | 1.27 / 70.0% / 0.167 | 1.2 (8 TPUv2 cores) | 2.457  (8 TPUv2 cores) | 36.5M |
| [Monte Carlo Dropout, improved (size=30)](dropout.py)<sup>11</sup> | 0.116 | 96.2% | 0.005 | 0.69 / 79.6% / 0.068 | 4.8 (8 TPUv2 cores) | 2.387  (8 TPUv2 cores)  | 36.5M |
| [Ensemble (size=4)](ensemble.py) | 0.114 | 96.6% | 0.010 | 0.81 / 77.9% / 0.087 | 1.2 (32 TPUv2 cores) | 0.317  (8 TPUv2 cores) | 146M |
| [Hyper-deep ensemble (size=4)](hyperdeepensemble.py)<sup>12</sup> | 0.118 | 96.4% | 0.008 | 0.83 / 76.8% / 0.079 | 1.2 (32 TPUv2 cores) | 20.7 / 0.317  (8 TPUv2 cores) | 146M |
| [Variational inference (sample=1)](variational_inference.py) | 0.211 | 94.7% | 0.029 | 1.46 / 71.3% / 0.181 | 5.5 (8 TPUv2 cores) | 0.220 (8 TPUv2 cores) | 73M |

## CIFAR-100

| Method | Test NLL | Test Accuracy | Test Cal. Error | cNLL/cA/cCE | Train Runtime (hours) | Test Runtime (ms / example)| # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | -----------| ----------- | ----------- |
| [Deterministic](deterministic.py)<sup>10</sup> | 0.875 | 79.8% | 0.0857 | 2.70 / 51.37% / 0.239 | 1.1 (8 TPUv2 cores) | 0.079 (8 TPUv2 cores) | 36.5M |
| [BatchEnsemble (size=4)](batchensemble.py) | 0.690 | 81.9% | 0.0265 | 2.56 / 53.1% / 0.149 | 5.5 (8 TPUv2 cores) | 0.319 (8 TPUv2 cores) | 36.6M |
| [Hyper-BatchEnsemble (size=4)](hyperbatchensemble.py) | 0.678 | 81.9% | 0.020  | - | - | - |  | 73.2M |
| [MIMO](mimo.py) | 0.690 | 82.0% | 0.022 | 2.28 / 53.7% / 0.129 | - | 0.080 | 36.5M |
| [Rank-1 BNN (Gaussian, size=4)](rank1_bnn.py) | 0.692 | 81.3% |  0.018 | 2.24 / 53.8% / 0.117 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| [Rank-1 BNN (Cauchy, size=4, 4 samples)](rank1_bnn.py) | 0.689 | 82.4% |  0.012 | 2.04 / 57.8% / 0.142 | 7.3 (8 TPUv2 cores) | - | 36.6M |
| [SNGP](sngp.py) | 0.805 | 80.2% | 0.020 | 2.02 / 54.6% / 0.092 | 2.6 (8 TPUv2 cores) | 0.378 (8 TPUv2 cores) | 36.5M |
| [SNGP, with AugMix](sngp.py) | 0.755 | 80.6% | 0.024 | 1.44 / 65.9% / 0.054 | 2.6 (8 TPUv2 cores) | 0.378 (8 TPUv2 cores) | 36.5M |
| [SNGP, with MC Dropout (size=10)](sngp.py) | 0.750 | 79.6% | 0.017 | 2.06 / 53.8% / 0.087 | 4.5 (8 TPUv2 cores) | 2.841 (8 TPUv2 cores) | 36.5M |
| [SNGP, with BatchEnsemble (size=4)](sngp_batchensemble.py) | 0.755 | 81.4% |  0.032 | 2.03 / 55.2% / 0.112 | - (8 TPUv2 cores) | - | -M |
| [SNGP Ensemble (size=4)](sngp_ensemble.py) | 0.665 | 81.9% | 0.011 | 1.95 / 56.8% / 0.091 | 2.6 (32 TPUv2 cores) | 1.198 (8 TPUv2 cores) | 146M |
| [Monte Carlo Dropout (size=1)](dropout.py) | 0.830 | 79.6% | 0.0501 | 2.90 / 42.63% / 0.202 | 1.1 (8 TPUv2 cores) | 0.082 (8 TPUv2 cores) | 36.5M |
| [Monte Carlo Dropout (size=30)](dropout.py) | 0.785 | 80.7% | 0.0487 | 2.73 / 46.2 / 0.207 | 1.1 (8 TPUv2 cores) | 2.457 (8 TPUv2 cores) | 36.5M |
| [Monte Carlo Dropout, improved (size=30)](dropout.py)<sup>11</sup> | 0.637 | 82.1% | 0.028 | 1.93 / 57.2% / 0.098 | 5.17 (8 TPUv2 cores) | 2.393 (8 TPUv2 cores) |  36.5M |
| [Ensemble (size=4)](ensemble.py) | 0.666 | 82.7% | 0.021 | 2.27 / 54.1% / 0.138 | 1.1 (32 TPUv2 cores) | 0.317 (8 TPUv2 cores) |  146M |
| [Hyper-deep ensemble (size=4)](hyperdeepensemble.py)<sup>12</sup> | 0.654 | 83.0% | 0.022 | 2.26 / 53.2% / 0.128 | 1.2 (32 TPUv2 cores) | 0.317  (8 TPUv2 cores) | 146M |
| [Variational inference (sample=1)](variational_inference.py) | 0.944 | 77.8% | 0.097 | 3.18 / 48.2% / 0.271 | 5.5 (8 TPUv2 cores) | 0.210 (8 TPUv2 cores) | 73M |
| [Heteroscedastic](heteroscedastic.py) | 0.833 | 80.2% | 0.059 | 2.40 / 52.1% / 0.177 | 5 (8 TPUv2 cores) | 4 (8 TPUv2 cores) | 37M |
| [Heteroscedastic Ensemble (size=4)](het_ensemble.py) | 0.671 | 82.7% | 0.026 | 2.07 / 55.2% / 0.105 | - | - | 148M |

## Metrics

We define metrics specific to CIFAR below. For general metrics, see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/main/baselines).

1. __cNLL/cA/cCE__. Negative-log-likelihood, accuracy, and calibration error on [CIFAR-10-C](https://arxiv.org/abs/1903.12261); we apply the same corruptions to produce a CIFAR-100-C. `c` stands for corrupted. Results take the mean across corruption intensities and corruption types.

## CIFAR-10 Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Test NLL | Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`meliketoy/wide-resnet.pytorch`](https://github.com/meliketoy/wide-resnet.pytorch) | Deterministic | - | 96.21% | 6.8 (1 GPU) | 36.5M |
| | Non-MC Dropout | - | 96.27% | 6.8 (1 GPU)  | 36.5M |
| [He et al. (2015)](https://arxiv.org/abs/1512.03385)<sup>1</sup> | Deterministic (ResNet-56) | - | 93.03% | - | 850K |
| | Deterministic (ResNet-110) | - | 93.39% | - | 1.7M |
| [Zagoruyko and Komodakis (2016)](https://github.com/szagoruyko/wide-residual-networks) | Deterministic | - | 96.00% | - | 36.5M |
| | Non-MC Dropout | - | 96.11% | - | 36.5M |
| [Louizos et al. (2017)](https://arxiv.org/abs/1705.08665)<sup>2</sup> | Group-normal Jeffreys | - | 91.2% | - | 998K |
| | Group-Horseshoe | - | 91.0% | - | 820K |
| [Molchanov et al. (2017)](https://arxiv.org/abs/1701.05369)<sup>2</sup> | Variational dropout | - | 92.7% | - | 304K |
| [Louizos et al. (2018)](https://arxiv.org/abs/1712.01312) | L0 regularization | - | 96.17% | 200 epochs | - |
| [Havasi et al. (2019)](https://openreview.net/forum?id=rkglZyHtvH)<sup>3</sup> | Refined VI (no batchnorm) | 0.696 | 75.5% | 5.5 (1 P100 GPU) | - |
| | Refined VI (batchnorm) | 0.593 | 79.7% | 5.5 (1 P100 GPU) | - |
| | Refined VI hybrid (no batchnorm) | 0.432 | 85.8% | 4.5 (1 P100 GPU) | - |
| | Refined VI hybrid (batchnorm) | 0.423 | 85.6% | 4.5 (1 P100 GPU) | - |
| [Heek and Kalchbrenner (2019)](https://arxiv.org/abs/1908.03491)<sup>5</sup> | Deterministic (ResNet-56) | 0.243 | 94.4% | 1000 epochs (1 V100 GPU) | 850K |
| | Adaptive Thermostat Monte Carlo (single sample) | 0.303 | 92.4% | 1000 epochs (1 V100 GPU) | - |
| | Adaptive Thermostat Monte Carlo (multi-sample) | 0.194 | 93.9% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (single sample) | 0.343 | 91.7% | 1000 epochs (1 V100 GPU) | - |
| | Sampler-based Nose-Hoover Thermostat (multi-sample) | 0.211 | 93.5% | 1000 epochs (1 V100 GPU) | - |
| [Izmailov et al. (2019)](https://arxiv.org/abs/1907.07504)<sup>6</sup> | Deterministic | 0.1294 | 96.41% | 300 epochs | 36.5M |
| | SWA | 0.1075 | 96.46% | 300 epochs | 36.5M |
| | SWAG | 0.1122 | 96.41% | 300 epochs | 803M |
| | Subspace Inference (PCA+VI) | 0.1081 | 96.32% | >300 epochs | 219M |
| [Osawa et al. (2019)](https://arxiv.org/abs/1906.02506)<sup>7</sup>  | Variational Online Gauss-Newton | 0.48 | 91.6% / 84.3% | 2.38 (128 P100 GPUs) | - |
| [Ovadia et al. (2019)](https://arxiv.org/abs/1906.02530)<sup>8</sup> | Deterministic (ResNet-20) | 1.120 | 91% | - | 274K |
| | Dropout | 0.771 | 91% | - | 274K |
| | Ensemble | 0.653 | - | 93.5% | - |
| | Variational inference | 0.823 | 88% | - | 630K |
| [Wen et al. (2019)](https://openreview.net/forum?id=Sklf1yrYDr)<sup>4</sup> | Deterministic (ResNet-32x4) | - | 95.31% | 250 epochs | 7.43M |
| | BatchEnsemble | - | 95.94% | 375 epochs | 7.47M |
| | Ensemble | - | 96.30% | 250 epochs each | 29.7M |
| | Monte Carlo Dropout | - | 95.72% | 375 epochs | 7.43M |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>9</sup> | Deterministic (ResNet-18) | - | 94.71% | 200 epochs | 11.7M |
| | cSGHMC | - | 95.73% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | 96.05% | 800 epochs | 561.6M |

## CIFAR-100 Related Results

We note results in the literature below. Note there are differences in the setup
(sometimes major), so take any comparisons with a grain of salt.

| Source | Method | Test NLL | Test Accuracy | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [`meliketoy/wide-resnet.pytorch`](https://github.com/meliketoy/wide-resnet.pytorch) | Deterministic | - | 81.02% | 6.8 (1 GPU) | 36.5M |
| | Non-MC Dropout | - | 81.49% | 6.8 (1 GPU)  | 36.5M |
| [Zagoruyko and Komodakis (2016)](https://github.com/szagoruyko/wide-residual-networks) | Deterministic | - | 80.75% | - | 36.5M |
| | Non-MC Dropout | - | 81.15% | - | 36.5M |
| [Izmailov et al. (2019)](https://arxiv.org/abs/1907.07504)<sup>7</sup> | Deterministic | 0.7958 | 80.76% | 300 epochs | 36.5M |
| | SWA | 0.6684 | 82.40% | 300 epochs | 36.5M |
| | SWAG | 0.6078 | 82.23% | 300 epochs | 803M |
| | Subspace Inference (PCA+VI) | 0.6052 | 82.63% | >300 epochs | 36.5M |
| [Zhang et al. (2019)](https://arxiv.org/abs/1902.03932)<sup>10</sup> | Deterministic (ResNet-18) | - | 77.40% | 200 epochs | 11.7M |
| | cSGHMC | - | 79.50% | 200 epochs | 140.4M |
| | Ensemble of cSGHMC (size=4) | - | 80.81% | 800 epochs | 561.6M |

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
12. The results of hyper-deep ensemble are slightly worse than those reported in the [original paper](https://arxiv.org/pdf/2006.13570.pdf). This is due to a simplifcation of the implementation, namely the absence of the stratification phase (see appendix C.7.5 in the [original paper](https://arxiv.org/pdf/2006.13570.pdf) for a discussion about the impact of the stratification phase).

TODO(trandustin): Add column for Checkpoints.
