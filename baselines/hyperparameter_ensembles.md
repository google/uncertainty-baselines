# Hyperparameter Ensembles for Robustness and Uncertainty Quantification

**Quick Links:**
[Paper](https://arxiv.org/abs/2006.13570) |
[Quick tutorial to hyper-deep ensembles (notebook)](https://github.com/google/uncertainty-baselines/blob/main/baselines/notebooks/Hyperparameter_Ensembles.ipynb) | 
[Paper Experiments: hyper-deep ensembles](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/hyperdeepensemble.py) | 
[Paper Experiments: hyper-batch ensembles](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/hyperbatchensemble.py)

Ensembles over neural network weights trained from different random initialization, known as deep ensembles, achieve state-of-the-art accuracy and calibration. The recently introduced batch ensembles provide a drop-in replacement that is more parameter efficient. We design ensembles not only over weights, but over hyperparameters to improve the state of the art in both settings. We propose two methods: *hyper-deep ensembles* and *hyper-batch ensembles*.

## Hyper-deep ensembles
Hyper-deep ensembles consist of a simple procedure that involves a random search over different hyperparameters, themselves stratified across multiple random initializations. It builds upon the approach of Caruana et al., 2004. The strong performance of hyper-deep ensembles highlights the benefit of combining models with both weight and hyperparameter diversity. Typically, training a *single* neural network involves some form of hyperparameter search. Instead of only taking the best model, we show that ensembling over those different variants of the model leads to great performance improvements. We hope the code we provide below is helpful as a strong baseline for research papers, but also an interesting choice in application scenarios when the compute budget is not too restrictive.

  <img src="http://florianwenzel.com/img/hyperens.png" alt="" width="400">
  
  *Hyper-deep ensembles vs. deep ensembles for a Wide ResNet 28-10
over CIFAR-100.*



### Code
**Quick intro:** Here is a [notebook](/notebooks/Hyperparameter_Ensembles.ipynb) that shows how easy hyper-deep ensembles can be implemented on top of an already existing deep neural network model. This is a great starting point if you want to play with hyper-batch ensembles yourself.

**Paper experiments:** To reproduce the results in our paper (ResNet-20 and Wide ResNet 28-10 architectures) the code can be found as part of *uncertainty-baselines* [here](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/hyperdeepensemble.py).

## Hyper-batch ensembles
Hyper-deep ensembles involve using multiple models at training and test time which is sometimes too expensive. We further propose a parameter-efficient version, hyper-batch ensembles, which builds on the layer structure of batch ensembles (Wen et al., 2019) and self-tuning networks (Mackay et al., 2018). Hyper-batch ensembles amortize the behavior of hyper-deep ensembles within a single model, resulting in computational and memory costs notably lower than typical ensembles.

### Code
**Edward2 Implementation:** We implement hyper-batch ensembles as new keras layer types that can be used as a drop-in replacement for your existing layers in a deep neural network model. An amortized version for convolutional layers `Conv2DHyperBatchEnsemble` can be found [here](https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/convolutional.py) and for dense layers `DenseHyperBatchEnsemble` can be found [here](https://github.com/google/edward2/blob/master/edward2/tensorflow/layers/dense.py).

**Paper experiments:** To reproduce the results in our paper (ResNet-20 and Wide ResNet 28-10 architectures) the code can be found as part of *uncertainty-baselines* [here](https://github.com/google/uncertainty-baselines/blob/main/baselines/cifar/hyperbatchensemble.py).

## For questions reach out to
Florian Wenzel ([florianwenzel@google.com](mailto:florianwenzel@google.com)) \
Rodolphe Jenatton ([rjenatton@google.com](mailto:rjenatton@google.com))


## Reference
> Florian Wenzel, Jasper Snoek, Dustin Tran and Rodolphe Jenatton (2020).
> [Hyperparameter Ensembles for Robustness and Uncertainty Quantification](https://arxiv.org/abs/2006.13570).
> In _Neural Information Processing Systems_.

```none
@inproceedings{wenzel2020good,
  author = {Florian Wenzel and Jasper Snoek and Dustin Tran and Rodolphe Jenatton},
  title = {Hyperparameter Ensembles for Robustness and Uncertainty Quantification},
  booktitle = {Neural Information Processing Systems)},
  year = {2020},
}
```
