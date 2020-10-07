# Toxic Comments Detection

## Test Performance on Wikipedia Toxicity

The table below shows the predictive and uncertainty performance on the held-out dataset of [Wikipedia Toxicity](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes). All models are based on BERT-base.

| Method | AUROC/AUPRC/Acc | ECE/Brier Score | Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.971/0.790/0.925 | 0.0195/0.0526 | 0.930/0.949/0.968/0.986/1.000 |
| SNGP                | 0.970/0.778/0.932 | 0.0199/0.0512 | 0.938/0.956/0.973/0.983/1.000 |
| Monte Carlo Dropout | 0.968/0.769/0.931 | 0.0191/0.0509 | 0.936/0.956/0.971/0.982/1.000 |
| Ensemble<sup>1</sup> (size=10)  | 0.972/0.798/0.925 | 0.0191/0.0536 | 0.930/0.947/0.967/0.985/1.000 |
| Deterministic + Focal Loss<sup>2</sup>  | 0.972/0.787/0.949 | 0.0153/0.0633 | 0.954/0.963/0.983/0.985/1.000 |
| SNGP + Focal Loss  | 0.971/0.783/0.945 | 0.0099/0.0383 | 0.952/0.969/0.984/0.991/1.000 |
| Ensemble + Focal Loss (size=10)  | 0.973/0.806/0.948 | 0.0154/0.0640 | 0.952/0.961/0.981/0.983/0.991 |

## Transfer Learning Performance on CivilComments

In practice, it is common scenario where a trained toxic detection model is deployed in a noisy environment with greater topical diversity and distribution shift.
To approximate this, we evaluate the performance of a WikipediaToxicity-trained model on the [CivilComments](https://www.tensorflow.org/datasets/catalog/civil_comments) dataset.
Comparing to WikipediaToxicity (which contains conversation between Wikipedia editors), CivilComments is a much more diverse and noisy dataset that aggregates comments from approximately 50 English-language news sites across the world.

| Method | AUROC/AUPRC/Acc | ECE/Brier Score | Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.789/0.668/0.958 | 0.0162/0.0251 | 0.963/0.978/0.989/0.993/0.996 |
| SNGP                | 0.774/0.657/0.965 | 0.0139/0.0256 | 0.971/0.983/0.990/0.994/1.000 |
| Monte Carlo Dropout | 0.800/0.681/0.964 | 0.0190/0.0248 | 0.970/0.983/0.990/0.993/0.996 |
| Ensemble (size=10)  | 0.795/0.676/0.959 | 0.0160/0.0246 | 0.964/0.978/0.989/0.993/0.996 |
| Deterministic + Focal Loss  | 0.804/0.679/0.981 | 0.0208/0.0396 | 0.979/0.984/0.991/0.993/0.996 |
| SNGP + Focal Loss  | 0.805/0.689/0.980 | 0.0237/0.0288 | 0.983/0.992/0.995/1.000/1.000 |
| Ensemble + Focal Loss (size=10)  | 0.806/0.682/0.980 | 0.0204/0.0390 | 0.975/0.984/0.992/0.993/0.996 |


## Metrics
We define metrics specific to Toxic Comments below. For general metrics,
see [`baselines/`](https://github.com/google/uncertainty-baselines/tree/master/baselines).

1. __Oracle Collaborative Accuracy__. Accuracy after sending a certain
fraction of examples that the model is not confident about to the human
moderators. Here we apply fractions `0.01`, `0.05`, `0.10`, `0.15` and
`0.20`. Note that if we randomly send sentences to human moderators,
the final accuracy is equal to `accuracy * (1 - fraction) + 1.0 * fraction`,
and here for deterministic model, the accuracies are `0.928` and `0.963` at
fraction `0.05`, which is much worse than Oracle Collaborative Accuracy
(`0.952` and `0.980`).


## Notes

1. A simple ensemble that averages over individual model's predictive
probabilities.

2. Trained with [focal loss](https://openreview.net/forum?id=SJxTZeHFPH)
(with alpha = 0.1 and gamma fine-tuned according to the architecture) to handle
class imbalance in the toxic comment datasets.
