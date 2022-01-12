# Exploring the Limits of Out-of-Distribution Detection

This repository contains code used to run experiments in
[Exploring the Limits of Out-of-Distribution Detection](https://arxiv.org/abs/2106.03004)
where we demonstrate that large-scale pre-trained transformers can significantly improve the state-of-the-art (SOTA) on a range of near OOD tasks across different data modalities. For instance, on CIFAR-100 vs CIFAR-10 OOD detection, we improve the AUROC from 85% (current SOTA) to more than 96% using Vision Transformers pre-trained on ImageNet-21k.


# Experiments for finetuning ViT pre-trained models and evaluating OOD detection performance.

The code can be found at ./vit.

`deterministic.py` contains the code for training and evaluation. `ood_utils.py` contains the code for utility functions for OOD evaluaiton, including computing OOD scores such as Mahalanobis distance and Maximum of Softmax Probability, and computing OOD metrics AUROC, AUPRC, and FPR@TPR.

The model configs files for CIFAR-10 and CIFAR-100 can be found at ./vit/experiments.

The pre-trained model checkpoint files can be downloaded from the Google Cloud storage bucket maintained by the [ViT team](https://github.com/google-research/vision_transformer) at [https://console.cloud.google.com/storage/vit_models/](https://console.cloud.google.com/storage/vit_models/).
For example, the ViT-B_16 pre-trained model checkpoint file can be downloaded from [https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz).

Finetuning the CIFAR model roughly takes 3.5 hours using TPU. Instructions for running the experiments using TPU can be found at [https://github.com/google/uncertainty-baselines#usage](https://github.com/google/uncertainty-baselines#usage).







