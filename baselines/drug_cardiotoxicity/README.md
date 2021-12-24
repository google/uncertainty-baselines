# Drug Cardiotoxicity

In this baseline, we develop Graph Neural Nets models that predict whether a drug candidate molecule causes cardiotoxicity (a binary classification task) using Tensorflow Dataset [cardiotox](https://www.tensorflow.org/datasets/catalog/cardiotox). Model performance can be found in our recent publication [Reliable Graph Neural Networks for Drug Discovery Under Distributional Shift](https://arxiv.org/abs/2111.12951). A list of available models:

- Base GNN model (`deterministic.py`)
- GNN-GP and GNN-SNGP (`sngp.py`)

## Cite
Please cite our paper if you use this code in your own work:

```
@ARTICLE{Han2021-tu,
  title         = "Reliable Graph Neural Networks for Drug Discovery Under
                   Distributional Shift",
  author        = "Han, Kehang and Lakshminarayanan, Balaji and Liu, Jeremiah",
  month         =  nov,
  year          =  2021,
  archivePrefix = "arXiv",
  primaryClass  = "cs.LG",
  eprint        = "2111.12951"
}
```
