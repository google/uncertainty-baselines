# Out-of-Distribution Detection and Selective Generation for Conditional Language Models

This repository contains code to compute the relative Mahalanobis distance (RMD) used in
[Out-of-Distribution Detection and Selective Generation for Conditional Language Models](https://arxiv.org/abs/2209.15558) for out-of-distributino detection for conditional language models. 
The RMD score is shown to be a highly accurate and lightweight OOD detection method for CLMs, as demonstrated on abstractive summarization and translation. 

For the summarization experiments, we use a [PEGASUS LARGE](https://proceedings.mlr.press/v119/zhang20ae.html) model fine-tuned on the [XSUM](https://aclanthology.org/D18-1206/) dataset. 
The model configuration can be found in the [Pegasus codebase](https://github.com/google-research/pegasus/blob/main/pegasus/params/ood_params.py) for reference. Please note that we have not tested whether the code is compatible with the latest version of the codebase.

