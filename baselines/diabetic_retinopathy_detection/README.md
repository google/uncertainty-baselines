# RETINA Benchmark

## Overview

Hi, good to see you here! 👋

Thanks for checking out the code for the RETINA Benchmark, part of the Uncertainty Baselines project.

See our 2021 NeurIPS Datasets and Benchmarks paper introducing this benchmark in detail [here](https://openreview.net/forum?id=jyd4Lyjr2iB).

This codebase will allow you to reproduce experiments from the paper (see citation [here](#cite)) as well as use the benchmarking utilities for predictive performance, robustness, and uncertainty quantification (evaluation and plotting) for your own Bayesian deep learning methods.

We would greatly appreciate a citation if you use this code in your own work.

## Prediction Task Overview

In this benchmark, models try to predict the presence or absence of diabetic retinopathy (a binary classification task) using data from the [Kaggle Diabetic Retinopathy Detection challenge](https://www.kaggle.com/c/diabetic-retinopathy-detection/data) and the [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection). Please see these pages for details on data collection, etc.

Models are trained with images of blood vessels in the eye, as seen in the [TensorFlow Datasets description](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection).

## Abstract

Bayesian deep learning seeks to equip deep neural networks with the ability to precisely quantify their predictive uncertainty, and has promised to make deep learning more reliable for safety-critical real-world applications. Yet, existing Bayesian deep learning methods fall short of this promise; new methods continue to be evaluated on unrealistic test beds that do not reflect the complexities of downstream real-world tasks that would benefit most from reliable uncertainty quantification. We propose a set of real-world tasks that accurately reflect such complexities and are designed to assess the reliability of predictive models in safety-critical scenarios. Specifically, we curate two publicly available datasets of high-resolution human retina images exhibiting varying degrees of diabetic retinopathy, a medical condition that can lead to blindness, and use them to design a suite of automated diagnosis tasks that require reliable predictive uncertainty quantification. We use these tasks to benchmark well-established and state-of-the-art Bayesian deep learning methods on task-specific evaluation metrics. We provide an easy-to-use codebase for fast and easy benchmarking following reproducibility and software design principles. We provide implementations of all methods included in the benchmark as well as results computed over 100 TPU days, 20 GPU days, 400 hyperparameter configurations, and evaluation on at least 6 random seeds each.

## Installation

Set up and activate the Python environment by executing

```
conda create -n ub python=3.8
conda activate ub
python3 -m pip install -e .[models,jax,tensorflow,torch,retinopathy]  # In uncertainty-baselines root directory
pip install "git+https://github.com/google-research/robustness_metrics.git#egg=robustness_metrics"
pip install 'git+https://github.com/google/edward2.git'
```

## Tuning Scripts

All hyperparameter tuning and fine-tuning (i.e., retraining with 6 different training seed) scripts are located in [baselines/diabetic_retinopathy_detection/experiments/tuning](baselines/diabetic_retinopathy_detection/experiments/tuning) and [baselines/diabetic_retinopathy_detection/experiments/top_config](baselines/diabetic_retinopathy_detection/experiments/top_config) respectively.

## Train a Model

Tuning scripts accept hyperparameters as simple Python arguments. We also implement logging using TensorBoard and Weights and Biases across all uncertainty quantification methods for the convenience of the user.

Execute a tuning script as follows (all tuning scripts are located in [baselines/diabetic_retinopathy_detection](baselines/diabetic_retinopathy_detection), and have by default had their arguments fixed to the configuration achieving the highest AUC on the in-domain validation set for the Country Shift task).

```
python baselines/diabetic_retinopathy_detection/deterministic.py --data_dir='gs://ub-data/retinopathy' --use_gpu=True --output_dir='gs://ub-data/retinopathy-out/deterministic'
``` 

## Select Top Performing Models

Model selection utilities are provided in [baselines/diabetic_retinopathy_detection/model_selection](baselines/diabetic_retinopathy_detection/model_selection).

First, follow the steps detailed in [parse_tensorboards.py](baselines/diabetic_retinopathy_detection/model_selection/parse_tensorboards.py) to convert TensorFlow event files to a public TensorBoard, and then parse this into a DataFrame containing results (per epoch metric logs, and hyperparameter details). The script expects that the TensorFlow event files are each in a folder corresponding to their identity, such as

```
dr_tuning/
   |--> 1/
        |--> tuning-run-seed-1.out.tfevents...
   |--> 2/
        |--> tuning-run-seed-2.out.tfevents...
   |--> 3/
        |--> tuning-run-seed-3.out.tfevents...
  ...
```

Following the steps in [parse_tensorboards.py](baselines/diabetic_retinopathy_detection/model_selection/parse_tensorboards.py) produces a file `results.tsv`. We can parse this file to obtain a ranking of the models based on our two tuning criteria: in-domain validation AUC, and area under the balanced accuracy referral curve (see [paper](https://openreview.net/pdf?id=jyd4Lyjr2iB)), by executing `python analyze_tensorboards.py` in the directory containing the `results.tsv` file. This ranking allows the user to select top performing checkpoints.

## Accessing Model Checkpoints

For each method, task (Country or Severity Shifts), and tuning method (see model selection details above) we release the six best-performing checkpoints [here](https://console.cloud.google.com/storage/browser/gresearch/reliable-deep-learning/checkpoints/baselines/diabetic_retinopathy_shifts).

For more details on the models, see the accompanying [Model Card](./model_card.md) along with method implementation and modification details provided in Section 5 of the benchmark whitepaper located [here](https://openreview.net/pdf?id=jyd4Lyjr2iB). 

## Evaluate a Model

### Evaluation Sweep Scripts

Scripts for the evaluation sweeps used for the paper are located in [baselines/diabetic_retinopathy_detection/experiments/eval](baselines/diabetic_retinopathy_detection/experiments/eval).

`.py` sweep files are used with [XManager](https://github.com/deepmind/xmanager), a framework for launching experiments on Google Cloud Platform. 

`.yaml` sweep files are tuning scripts used with [Weights & Biases](https://docs.wandb.ai/guides/sweeps).

### Selective Prediction and Referral Curves

In Selective Prediction, a model's predictive uncertainty is used to choose a subset of the test set for which predictions will be evaluated. In particular, the uncertainty per test input forms a ranking. The X% of test inputs with the highest uncertainty are referred to a specialist, and the model performance is evaluated on the (100 - X)% remaining inputs. Standard evaluation therefore uses a _referral fraction_ = 0, i.e., the full test set is retained.

We may wish to use a predictive model of diabetic retinopathy to ease the burden on clinical practitioners. Under Selective Prediction, the model refers the examples on which it is least confident to specialists. We can tune the _referral fraction_ parameter based on practitioner availability, and a model with well-calibrated uncertainty will have high performance on metrics such as AUC/accuracy on the retained (non-referred) evaluation data, because its uncertainty and predictive performance are (negatively) correlated.

### Using Evaluation Utilities

Once you have trained a few models and have placed the top performing checkpoints in a `checkpoint_bucket`, run an evaluation over the methods, and store both scalar results on predictive performance and uncertainty quantification metrics (e.g., accuracy, AUC, expected calibration error) as well as results needed for _selective prediction_ and _receiver operating characteristic_ plots, as follows.

Single model evaluation (e.g., X different training seeds for a deterministic model).
```
python baselines/diabetic_retinopathy_detection/eval_model.py --checkpoint_bucket='bucket-name' --output_bucket='results-bucket-name' --dr_decision_threshold='moderate' --model_type='deterministic' --single_model_multi_train_seeds=True
```

Ensemble evaluation, where each ensemble is formed by sampling without replacement from all available checkpoints in the directory, with sample size `k_ensemble_members` = 3 and number of sampling repetitions `ensemble_sampling_repetitions` = 6 (as in paper):
```
python baselines/diabetic_retinopathy_detection/eval_model.py --checkpoint_bucket='bucket-name' --output_bucket='results-bucket-name' --dr_decision_threshold='moderate' --model_type='deterministic' --k_ensemble_members=3 --ensemble_sampling_repetitions=6
```

## Plot ROC and Selective Prediction Curves

Now we can generate the same ROC and selective prediction plots as appear in the paper (e.g., if you have run the above training and evaluation for many different Bayesian deep learning methods).

Note the flag `distribution_shift` to specify for which distribution shift you aim to generate outputs. See [plot_results.py](baselines/diabetic_retinopathy_detection/plot_results.py) for info on expected directory structure.

```
python baselines/diabetic_retinopathy_detection/plot_results.py --results_dir='gs://results-bucket-name' --output_dir='gs://plot-outputs' --distribution_shift=aptos
```

## Previous Tuning Details

The below tuning was done for the initial Uncertainty Baselines release. See [baselines/diabetic_retinopathy_detection/experiments/initial_tuning](baselines/diabetic_retinopathy_detection/experiments/initial_tuning) for the corresponding tuning scripts and the trained model checkpoints [here](https://console.cloud.google.com/storage/browser/gresearch/reliable-deep-learning/checkpoints/baselines/diabetic_retinopathy_shifts).

### Model Checkpoints
For each method we release the best-performing checkpoints. These checkpoints were trained on the combined training and validation set, using hyperparameters selected from the best validation performance. Each checkpoint was selected to be from the step during training with the best test AUC (averaged across the 10 random seeds). This was epoch 63 for the deterministic model, epoch 72 for the MC-Dropout method, epoch 31 for the Variational Inference method, and epoch 61 for the Radial BNNs method. For more details on the models, see the accompanying [Model Card](./model_card.md), which covers all the models below, as the dataset is exactly the same across them all, and the only model differences are minor calibration improvements. The checkpoints can be browsed [here](https://console.cloud.google.com/storage/browser/gresearch/reliable-deep-learning/checkpoints/baselines/diabetic_retinopathy_detection).

### Tuning
For this baseline, two rounds of quasirandom search were conducted on the hyperparameters listed below, where the first round was a heuristically-picked larger search space and the second round was a hand-tuned smaller range around the better performing values. Each round was for 50 trials, and the final hyperparemeters were selected using the final validation AUC from the second tuning round. These best hyperparameters were used to retrain combined train and validation sets over 10 seeds. **We note that the learning rate schedules could likely be tuned for improved performance, but leave this to future work.** All our intermediate and final tuning results are available below hosted on [tensorboard.dev](tensorboard.dev).

Below are links to [tensorboard.dev](tensorboard.dev) TensorBoards for each baseline method that contain the metric values of the various tuning runs as well as the hyperparameter points sampled in the `HPARAMS` tab at the top of the page.

#### Deterministic
[[First Tuning Round]](https://tensorboard.dev/experiment/nAygVvdjSWWAEQRDD8Z0Aw/) [[Final Tuning Round]](https://tensorboard.dev/experiment/GLxGQR8pQhypBr9jGdBMUQ/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/lh5yXcwzRc2ZNmId34ujPw/)

---

#### Monte-Carlo Dropout
[[First Tuning Round]](https://tensorboard.dev/experiment/xDVLkDAgR1uJqyxIqkdPIQ/) [[Final Tuning Round]](https://tensorboard.dev/experiment/1qy7JJfYQYqQ1lanieSYew/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/aMr4glcES6qg43P4HvckTg/)

---

#### Radial Bayesian Neural Networks
[[First Tuning Round]](https://tensorboard.dev/experiment/5CzJYikVTvKQLdqSnmUrpg/) [[Final Tuning Round]](https://tensorboard.dev/experiment/RDf1PKZkSZ2PGo1H8wnWBw/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/040rBdKBQPir8cDhReyk3A/)

---

#### Variational Inference
[[First Tuning Round]](https://tensorboard.dev/experiment/gVwRJIRoQoyRrfG1boJVPA/) [[Final Tuning Round]](https://tensorboard.dev/experiment/n9NYA7ryRG6jCYdpyQYoOQ/)  [[Best Hyperparamters 10 seeds]](https://tensorboard.dev/experiment/mPZt9k0lQ1yF2TAuE2cxqw/)

---


### Search spaces
Search space for the initial and final rounds of tuning on the deterministic method. We used a stepwise decay for the initial round but switched to a linear decay for the final round to alleviate overfitting, where we tuned the linear decay factor on the grid `[1e-3, 1e-2, 0.1]`.

| | Learning Rate | 1 - momentum | L2 |
|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] |
| Final | [0.03, 0.5] | [5e-3, 0.05] | [1e-6, 2e-4] |

Search space for the initial and final rounds of tuning on the Monte Carlo Dropout method.

| | Learning Rate | 1 - momentum | L2 | dropout |
|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [0.01, 0.25] |
| Final | [1e-2,0.5] | [1e-2, 0.04] | [1e-5, 1e-3] | [0.01, 0.2]  |

Search space for the initial and final rounds of tuning on the Radial BNN method.

| | Learning Rate | 1 - momentum | L2 | stddev_mean_init | stddev_stddev_init |
|---|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [1e-5,1e-1] | [1e-2,1] |
| Final | [0.15,1] | [1e-2, 0.05] | [1e-4, 1e-3] | [1e-5, 2e-2] | [1e-2, 0.2] |

Search space for the initial and final rounds of tuning on the Variational Inference method.

| | Learning Rate | 1 - momentum | L2 | stddev_mean_init | stddev_stddev_init |
|---|---|---|---|---|---|
| Initial | [1e-3,0.1] | [1e-2,0.1] | [1e-5,1e-3] | [1e-5,1e-1] | [1e-2,1] |
| Final | [0.02,5] | [0.02, 0.1] | [1e-5, 2e-4] | [1e-5, 2e-3] | [1e-2, 1] |

## Cite

Please cite our paper if you use this code in your own work:

```
@inproceedings{
    band2021benchmarking,
    title={Benchmarking Bayesian Deep Learning on Diabetic Retinopathy Detection Tasks},
    author={Neil Band and Tim G. J. Rudner and Qixuan Feng and Angelos Filos and Zachary Nado and Michael W Dusenberry and Ghassen Jerfel and Dustin Tran and Yarin Gal},
    booktitle={Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
    year={2021},
    url={https://openreview.net/forum?id=jyd4Lyjr2iB}
}
```

## Acknowledgements

The Diabetic Retinopathy Detection baseline was contributed through collaboration with the [Oxford Applied and Theoretical Machine Learning](http://oatml.cs.ox.ac.uk/) (OATML) group, with sponsorship from:

<table align="center">
    <tr>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/intel.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oatml.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/oxcs.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
        <td><img src="https://github.com/OATML/bdl-benchmarks/blob/alpha/assets/turing.png" style="float: left; width: 200px; margin-right: 1%; margin-bottom: 0.5em; margin-top: 0.0em"></td>
    </tr>
</table>
