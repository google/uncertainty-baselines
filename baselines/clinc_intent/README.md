# CLINC Intent Detection

Task-oriented dialog systems need to know when a query falls outside their range of supported intents. The [CLINC Intent Detection dataset](https://www.tensorflow.org/datasets/catalog/clinc_oos) introduces a new intent detection benchmark that covers 150 intent classes over 10 domains, capturing the breadth that a production task-oriented agent must handle. It also contains out-of-scope queries to test the model's ability in out-of-domain (OOD) detection, since a reliable task-driven dialog system cannot assume that every query at inference time belongs to a system-supported intent class.


## BERT-base (12 layer, 768 unit)

| Method | Test NLL | Test Accuracy | Test Cal. Error | cNLL/cAcc/cECE | OOD AUROC/AUPRC | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| [Deterministic](deterministic.py) | 0.186 | 96.5% | 0.0218 | 1.803 / 77.6% / 0.142 | 0.942 / 0.853 | 0.8 (8 TPUv3 cores) | 110M |
| [Monte Carlo Dropout (Size=10)](dropout.py) | 0.171 | 96.5% | 0.0170 | 1.717 / 77.7% / 0.126 | 0.951 / 0.861 | 0.8 (80 TPUv3 cores) | 110M |
| [SNGP](sngp.py) | 0.139 | 96.9% | 0.0104 | 1.576 / 78.0% / 0.080 | 0.969 / 0.908 | 0.4 (80 TPUv3 cores) | 110M |
| [SNGP Ensemble (Size=10)](sngp_ensemble.py) | 0.118 | 97.4% | 0.0094 | 1.424 / 79.5% / 0.064 | 0.973 / 0.910 | 0.8 (80 TPUv3 cores) | 1100M |
| [Ensemble (Size=10)](ensemble.py) |  0.169 | 97.5% | 0.0128 | 1.600 / 79.1% / 0.098 | 0.958 / 0.862 | 0.8 (80 TPUv3 cores) | 1100M |

## Metrics

We define metrics specific to CLINC OOS below.

1. __cNLL/cA/cCE__. Testing negative-log-likelihood, accuracy, and calibration error on combined in-scope and out-of-scope queries.
2. __OOD AUROC/AUPRC__. Areas under the receiver-operating-characteristics (ROC) curve and the precision-recall (PR) curve for the model's ability in distinguish in-scope and out-of-scope queries.
