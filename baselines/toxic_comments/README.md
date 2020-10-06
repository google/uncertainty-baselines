# Toxic Comments Detection

## BERT-base (12 layer, 768 units)

| Method | Test AUROC/AUPRC/Acc | Test ECE/Brier Score | Oracle Collaborative Acc | Transfer AUROC/AUPRC/Acc | Transfer ECE/Brier Score | Transfer Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.971/0.790/0.925 | 0.0195/0.0526 | 0.930/0.949/0.968/0.986/1.000 | 0.789/0.668/0.958 | 0.0162/0.0251 | 0.963/0.978/0.989/0.993/0.996 |
| SNGP                | 0.970/0.778/0.932 | 0.0199/0.0512 | 0.938/0.956/0.973/0.983/1.000 | 0.774/0.657/0.965 | 0.0139/0.0256 | 0.971/0.983/0.990/0.994/1.000 |
| SNGP + Focal Loss<sup>1</sup>   | 0.970/0.786/0.949 | 0.0099/0.0367 | 0.955/0.971/0.985/0.992/1.000 | 0.787/0.661/0.980 | 0.0237/0.0285 | 0.984/0.993/1.000/1.000/1.000 |
| Monte Carlo Dropout | 0.968/0.769/0.931 | 0.0191/0.0509 | 0.936/0.956/0.971/0.982/1.000 | 0.800/0.681/0.964 | 0.0190/0.0248 | 0.970/0.983/0.990/0.993/0.996 |
| Ensemble (size=10)  | 0.972/0.798/0.925 | 0.0191/0.0536 | 0.930/0.947/0.967/0.985/1.000 | 0.795/0.676/0.959 | 0.0160/0.0246 | 0.964/0.978/0.989/0.993/0.996 |

## Metrics
We define metrics specific to Toxic Comments below. For general metrics,
see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __Test/Transfer__. The model was trained on Wikipedia Toxicity, and we
evaluated the performance on Wikipedia Toxicity (Test), and transfer learning
performance on CivilComments (Transfer).

2. __Oracle Collaborative Acc__. Accuracy after sending a certain
fraction of examples that the model is not confident about to the human
moderators. Here we apply fractions `0.01`, `0.05`, `0.10`, `0.15` and
`0.20`. Note that if we randomly send sentences to human moderators,
the final accuracy is equal to `accuracy * (1 - fraction) + 1.0 * fraction`,
and here for deterministic model, the accuracies are `0.928` and `0.963` at
fraction `0.05`, which is much worse than Oracle Collaborative Accuracy
(`0.952` and `0.980`).


## Notes

1. Trained with [focal loss](https://openreview.net/forum?id=SJxTZeHFPH)
(with alpha = 0.1, gamma = 1) to handle class imbalance in the toxic comment 
datasets.
