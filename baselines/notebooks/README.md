# Notebooks

This directory contains tutorials and learning resources for state-of-the-art 
uncertainty methods that are used in the uncertainty baseline.


## Spectral-normalized Neural Gaussian Process (SNGP)

Spectral-normalized neural GP (SNGP) is a simple method to improve
a deterministic neural network's uncertainty by applying spectral
normalization to the hidden layers, and then replace the dense output layer
with a Gaussian process layer.

* [Paper](https://arxiv.org/abs/2006.10108)
* [Official TF.org Tutorial](https://www.tensorflow.org/tutorials/understanding/sngp)
* [TF.text Tutorial for BERT-SNGP](https://www.tensorflow.org/text/tutorials/uncertainty_quantification_with_sngp_bert)


## Hyperparameter Ensemble

Ensembles over neural network weights trained from different random initialization, known as deep ensembles, achieve state-of-the-art accuracy and calibration. The recently introduced batch ensembles provide a drop-in replacement that is more parameter efficient. We design ensembles not only over weights, but over hyperparameters to improve the state of the art in both settings. We propose two methods: *hyper-deep ensembles* and *hyper-batch ensembles*.

* [Paper](https://arxiv.org/abs/2006.13570) 
* [Notebook tutorial](https://github.com/google/uncertainty-baselines/blob/master/baselines/notebooks/Hyperparameter_Ensembles.ipynb)
