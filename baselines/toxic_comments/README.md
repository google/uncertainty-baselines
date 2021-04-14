# Toxic Comments Detection

## Test Performance on Wikipedia Toxicity

The table below shows the predictive and uncertainty performance on the held-out dataset of [Wikipedia Toxicity](https://www.tensorflow.org/datasets/catalog/wikipedia_toxicity_subtypes). All models are based on BERT-base.

| Method | AUROC/AUPRC/Acc | ECE/Brier Score | Oracle Collaborative Acc |
| ----------- | ----------- | ----------- | ----------- |
| Deterministic       | 0.972/0.782/0.922 | 0.0259/0.0565 | 0.927/0.944/0.962/0.975/0.985 |
| SNGP                | 0.971/0.778/0.924 | 0.0303/0.0550 | 0.929/0.947/0.965/0.979/0.986 |
| Monte Carlo Dropout | 0.971/0.787/0.926 | 0.0196/0.0505 | 0.931/0.950/0.969/0.982/0.991 |
| Ensemble<sup>1</sup> (size=10)  | 0.974/0.803/0.922 | 0.0261/0.0559 | 0.927/0.945/0.963/0.977/0.986 |
| SNGP Ensemble (size=10) | 0.975/0.806/0.922 | 0.0301/0.0556 | 0.928/0.945/0.965/0.984/1.000 |
| Deterministic + Focal Loss<sup>2</sup>  | 0.973/0.790/0.949 | 0.1489/0.0613 | 0.954/0.970/0.985/0.991/0.996 |
| SNGP + Focal Loss  | 0.974/0.802/0.947 | 0.0128/0.0370 | 0.953/0.970/0.983/0.991/0.997 |
| Monte Carlo Dropout + Focal Loss| 0.971/0.799/0.949 | 0.1479/0.0626 | 0.953/0.968/0.982/0.990/0.997 |
| Ensemble + Focal Loss (size=10)  | 0.973/0.806/0.948 | 0.1534/0.0638 | 0.952/0.968/0.982/0.991/0.996 |
| SNGP Ensemble + Focal Loss (size=10) | 0.975/0.816/0.946 | 0.0086/0.0382 | 0.952/0.969/0.984/0.992/1.000 |

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
| SNGP Ensemble (size=10) | 0.774/0.673/0.962 | 0.0066/0.0248 | 0.967/0.981/0.991/0.996/1.000 |
| Deterministic + Focal Loss | 0.804/0.679/0.982 | 0.1992/0.0372 | 0.985/0.992/0.996/0.997/0.998 |
| SNGP + Focal Loss  | 0.802/0.684/0.981 | 0.0155/0.0278 | 0.984/0.993/0.996/0.998/0.998 |
| Monte Carlo Dropout + Focal Loss| 0.799/0.679/0.980 | 0.2423/0.0379 | 0.985/0.992/0.996/0.997/0.998 |
| Ensemble + Focal Loss (size=10)  | 0.806/0.682/0.979 | 0.2020/0.0388 | 0.984/0.992/0.996/0.997/0.998 |
| SNGP Ensemble + Focal Loss (size=10) | 0.805/0.687/0.979 | 0.0184/0.0266 | 0.984/0.992/0.994/0.998/1.000 |

## Transfer Learning Performance on CivilComments Identity Dataset (SNGP + Focal Loss)

| Identity Type                  | AUROC/AUPRC/Acc      | ECE/Brier Score | Oracle Collaborative Acc           |
| ------------------------------ | -------------------- | --------------- | ---------------------------------- |
| gender                         | 0.7666/0.7826/0.9698 | 0.0269/0.0483   | 0.9741/0.9848/0.9908/0.9937/0.9959 |
| sexual_orientation             | 0.7709/0.9077/0.9570 | 0.0550/0.0876   | 0.9650/0.9793/0.9837/0.9854/0.9869 |
| religion                       | 0.7495/0.7765/0.9751 | 0.0130/0.0574   | 0.9797/0.9878/0.9922/0.9957/0.9959 |
| race                           | 0.7622/0.8934/0.9508 | 0.0245/0.0985   | 0.9566/0.9677/0.9760/0.9807/0.9846 |
| disability                     | 0.7412/0.8355/0.9688 | 0.0366/0.0595   | 0.9688/0.9779/0.9908/0.9944/0.9977 |

| Identity Type                  | AUROC/AUPRC/Acc      | ECE/Brier Score | Oracle Collaborative Acc           |
| ------------------------------ | -------------------- | --------------- | ---------------------------------- |
| male                           | 0.7599/0.7756/0.9747 | 0.0334/0.0478   | 0.9783/0.9860/0.9913/0.9940/0.9956 |
| female                         | 0.7664/0.7805/0.9733 | 0.0263/0.0465   | 0.9774/0.9889/0.9942/0.9960/0.9979 |
| transgender                    | 0.6357/0.8183/0.9688 | 0.0460/0.0528   | 0.9688/0.9844/0.9891/0.9937/0.9984 |
| homosexual_gay_or_lesbian      | 0.7664/0.9100/0.9531 | 0.0554/0.0881   | 0.9607/0.9780/0.9831/0.9854/0.9867 |
| christian                      | 0.7425/0.7296/0.9822 | 0.0163/0.0439   | 0.9853/0.9930/0.9965/0.9984/0.9985 |
| jewish                         | 0.7609/0.8345/0.9812 | 0.0347/0.0613   | 0.9812/0.9906/0.9970/0.9984/0.9997 |
| muslim                         | 0.7195/0.8435/0.9583 | 0.0116/0.0916   | 0.9653/0.9758/0.9817/0.9856/0.9887 |
| atheist                        | 0.7317/0.7530/0.9844 | 0.0266/0.0389   | 0.9922/0.9922/0.9955/0.9994/1.0000 |
| black                          | 0.7512/0.9193/0.9516 | 0.0157/0.1183   | 0.9552/0.9657/0.9732/0.9781/0.9829 |
| white                          | 0.7513/0.9073/0.9449 | 0.0254/0.1045   | 0.9504/0.9634/0.9710/0.9755/0.9799 |
| asian                          | 0.7186/0.6465/0.9766 | 0.0276/0.0358   | 0.9766/0.9891/0.9922/0.9922/0.9925 |
| psychiatric_or_mental_illness  | 0.7355/0.8362/0.9635 | 0.0363/0.0621   | 0.9635/0.9727/0.9856/0.9892/0.9951 |

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
