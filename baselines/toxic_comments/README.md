# Toxic Comments Detection

## BERT-base (12 layer, 768 units)

| Method | Test AUROC/AUPRC/Accuracy | Test ECE/Brier Score | Oracle Collaborative Accuracy (fraction=0.01/0.05/0.1) | Transfer AUROC/AUPRC/Accuracy | Transfer ECE/Brier Score | Transfer Oracle Collaborative Accuracy (fraction=0.01/0.05/0.1) |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic      | 0.971/0.771/0.921 | 0.0257/0.0549 | 0.927/0.947/0.965 | 0.790/0.666/0.956 | 0.0172/0.0265 | 0.962/0.976/0.985 |
| Ensemble (size=10) | 0.972/0.799/0.925 | 0.0193/0.0539 | 0.930/0.948/0.966 | 0.795/0.674/0.959 | 0.0173/0.0247 | 0.964/0.978/0.988 |

## Metrics
We define metrics specific to Toxic Comments below. For general metrics,
see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __Test/Transfer__. The model was trained on Wikipedia Toxicity, and we
evaluated the performance on Wikipedia Toxicity (Test), and transfer learning
performance on CivilComments (Transfer).

2. __Oracle Collaborative Accuracy__. Accuracy after sending a certain
fraction of examples that the model is not confident about to the human
moderators. Here we apply fractions `0.01`, `0.05` and `0.1`. Note that if we
randomly send sentences to human moderators, the final accuracy is equal to
`accuracy * (1 - fraction) + 1.0 * fraction`, and here for deterministic
model, the accuracies are `0.928` and `0.963` at fraction `0.05`, which is much
worse than Oracle Collaborative Accuracy (`0.952` and `0.980`).