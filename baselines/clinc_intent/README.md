# CLINC Intent Detection

Task-oriented dialog systems need to know when a query falls outside their range of supported intents. The [CLINC Intent Detection dataset](https://www.tensorflow.org/datasets/catalog/clinc_oos) introduce a new intent detection benchmark that covers 150 intent classes over 10 domains, capturing the breadth that a production task-oriented agent must handle. It also contains out-of-scope queries to test the model's ability in out-of-domain (OOD) detection, since a reliable task-driven dialog system cannot assume that every query at inference time belongs to a system-supported intent class.


## BERT-base (12 layer, 768 unit)

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | cNLL/cAcc/cECE | OOD AUROC/AUPRC | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Determinisitic | 0.003 / 0.186 | 99.9% / 96.5% | 3e-4 / 0.0218 | 1.803 / 77.6% / 0.142 | 0.942 / 0.853 | 0.8 (8 TPUv3 cores) | 110M |
| Ensemble (Size=10) | 0.001 / 0.169 | 99.9% / 97.5% | 1e-4 / 0.0128 | - / - / - | - / - | 0.8 (80 TPUv3 cores) | 1100M |

## Metrics

We define metrics specific to CLINC OOS below.

1. __cNLL/cA/cCE__. Testing negative-log-likelihood, accuracy, and calibration error on combined in-scope and 
out-of-scope queries.
2. __OOD AUROC/AUPRC__. Areas under the receiver-operating-characteristics (ROC) curve and the precision-recall (PR) curve for the model's ability in distinguish in-scope and out-of-scope queries.
