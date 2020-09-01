# Toxic Comments Detection

## BERT-base (12 layer, 768 units)

| Method | Test AUROC/AUPRC/Accuracy | Test ECE/Brier Score | Transfer AUROC/AUPRC/Accuracy | Transfer ECE/Brier Score | Oracle Collaborative Accuracy (fraction=0.05/0.1/0.2) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.971/0.762/0.928 | 0.0218/0.0522 | 0.785/0.656/0.963 | 0.0137/0.0270 | 0.951/0.973/1.000 |

## Metrics
We define metrics specific to Toxic Comments below. For general metrics, 
see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __Test/Transfer__. The model was trained on Wikipedia Toxicity, and we 
evaluated the performance on Wikipedia Toxicity (Test), and transfer learning 
performance on CivilComments (Transfer).

2. __Oracle Collaborative Accuracy__. We apply fractions (the fraction of 
total examples to send to moderators) 0.001, 0.01 and 0.05 on the in-domain test set.