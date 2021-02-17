# Toxic Comments Detection

## Test Performance on Wikipedia Toxicity

The table below shows the predictive and uncertainty performance on the held-out dataset of [Wikipedia Toxicity](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes). All models are based on BERT-base.

| Method | AUROC/AUPRC/Acc | ECE/Brier Score | Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.972/0.782/0.922 | 0.0259/0.0565 | 0.927/0.944/0.962/0.975/0.985 |
| SNGP                | 0.971/0.778/0.924 | 0.0303/0.0550 | 0.929/0.947/0.965/0.979/0.986 |
| Monte Carlo Dropout | 0.971/0.787/0.926 | 0.0196/0.0505 | 0.931/0.950/0.969/0.982/0.991 |
| Ensemble<sup>1</sup> (size=10)  | 0.974/0.803/0.922 | 0.0261/0.0559 | 0.927/0.945/0.963/0.977/0.986 |
| Deterministic + Focal Loss<sup>2</sup>  | 0.973/0.790/0.949 | 0.1489/0.0613 | 0.954/0.970/0.985/0.991/0.996 |
| SNGP + Focal Loss  | 0.974/0.802/0.947 | 0.0128/0.0370 | 0.953/0.970/0.983/0.991/0.997 |
| Monte Carlo Dropout + Focal Loss| 0.971/0.799/0.949 | 0.1479/0.0626 | 0.953/0.968/0.982/0.990/0.997 |
| Ensemble + Focal Loss (size=10)  | 0.973/0.806/0.948 | 0.1534/0.0638 | 0.952/0.968/0.982/0.991/0.996 |

## Transfer Learning Performance on CivilComments

In practice, it is common scenario where a trained toxic detection model is deployed in a noisy environment with greater topical diversity and distribution shift.
To approximate this, we evaluate the performance of a WikipediaToxicity-trained model on the [CivilComments](https://www.tensorflow.org/datasets/catalog/civil_comments) dataset.
Comparing to WikipediaToxicity (which contains conversation between Wikipedia editors), CivilComments is a much more diverse and noisy dataset that aggregates comments from approximately 50 English-language news sites across the world.

| Method | AUROC/AUPRC/Acc | ECE/Brier Score | Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.783/0.673/0.958 | 0.0129/0.0245 | 0.963/0.978/0.988/0.994/0.997 |
| SNGP                | 0.773/0.669/0.962 | 0.0084/0.0250 | 0.966/0.981/0.991/0.996/0.997 |
| Monte Carlo Dropout | 0.774/0.667/0.964 | 0.0125/0.0242 | 0.969/0.983/0.993/0.997/0.998 |
| Ensemble (size=10)  | 0.786/0.676/0.961 | 0.0126/0.0244 | 0.966/0.980/0.990/0.994/0.997 |
| Deterministic + Focal Loss  | 0.804/0.679/0.982 | 0.1992/0.0372 | 0.985/0.992/0.996/0.997/0.998 |
| SNGP + Focal Loss  | 0.802/0.684/0.981 | 0.0155/0.0278 | 0.984/0.993/0.996/0.998/0.998 |
| Monte Carlo Dropout + Focal Loss| 0.799/0.679/0.980 | 0.2423/0.0379 | 0.985/0.992/0.996/0.997/0.998 |
| Ensemble + Focal Loss (size=10)  | 0.806/0.682/0.979 | 0.2020/0.0388 | 0.984/0.992/0.996/0.997/0.998 |


## Metrics
We define metrics specific to Toxic Comments below. For general metrics,
see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __Oracle Collaborative Accuracy__. Accuracy after sending a certain
fraction of examples that the model is not confident about to the human
moderators. Here we apply fractions `0.01`, `0.05`, `0.10`, `0.15` and
`0.20`. Note that if we randomly send sentences to human moderators,
the final accuracy is equal to `accuracy * (1 - fraction) + 1.0 * fraction`,
and here for deterministic model, the accuracies are `0.926` and `0.930` at
fraction `0.05` and `0.10`, which is much worse than Oracle Collaborative Accuracy
(`0.944` and `0.962`).


## Notes

1. A simple ensemble that averages over individual model's predictive
probabilities.

2. Trained with [focal loss](https://openreview.net/forum?id=SJxTZeHFPH)
(with alpha = 0.1 and gamma fine-tuned according to the architecture) to handle
class imbalance in the toxic comment datasets.
