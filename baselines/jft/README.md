# JFT-300M (and ImageNet 21K)

## Getting Started

There are three stages: pretraining, finetuning, and evaluation.

1. __Pretraining.__  A pretraining job trains a specific model on a specific
upstream dataset. The job's config determines options such as the choice of
model and upstream dataset.

    The job reports upstream train/validation performance. It also includes few-shot
    performance across certain downstream datasets (see
    [`experiments/common_fewshot.py`](experiments/common_fewshot.py)).

    Example:
    [`experiments/jft300m_vit_l32.py`](experiments/jft300m_vit_l32.py) trains a
    ViT-L/32 on JFT-300M.

2. __Finetuning.__ A finetuning job trains a specific pretrained model on a
specific downstream dataset. The job's config determines options such as the
pretrained model checkpoint, choice of finetuning model, and choice of
downstream dataset. It can mix & match models: for example, load weights from a
pretrained deterministic ViT and finetune a BatchEnsemble ViT.

    The job reports downstream train/validation peformance. It also includes
    certain out-of-distribution evaluations: CIFAR/ImageNet OOD detection (both
    few-shot and full dataset), CIFAR-10H, and ImageNet ReaL. To launch a
    finetuning job in order to only perform evaluation, set `only_eval=True` and
    potentially training step/log values to 1
    ([example](experiments/imagenet21k_vit_base16_eval_imagenet.py)).

    Example:
    [`experiments/vit_base16_finetune_cifar10_and_100.py`](experiments/vit_base16_finetune_cifar10_and_100.py)
    finetunes a ViT on CIFAR-10. This config is set
    to use a ViT-B/16 checkpoint obtained from the previous step.

    TODO(trandustin,dusenberrymw): Add finetuning configs for L/32.

    Example: For active learning, see [`active_learning.py`](active_learning.py)
    which is a slight adaptation to an existing finetuning config.

    Example: Diabetic retinopathy is currently implemented in a branch of the
    codebase.
    [`aptos-deterministic.yaml`](https://github.com/google/uncertainty-baselines/blob/drd-vit-i21k/baselines/diabetic_retinopathy_detection/experiments/vit16_finetune/aptos-deterministic.yaml)
    is a wandb config sweep for finetuning and reports train, validation, and
    out-of-distribution performance. For OOD, you can specify the specific shift
    with `distribution_shift`.

    Example: Segmentation is currently implemented in a fork of the codebase and
    uses two steps.
    [`run_deterministic_splits_vm.sh`](https://github.com/ekellbuch/uncertainty-baselines/blob/add_umetrics/experimental/cityscapes/run_deterministic_splits_vm.sh)
    finetunes and
    [`run_deterministic_eval.sh`](https://github.com/ekellbuch/uncertainty-baselines/blob/add_umetrics/experimental/cityscapes/run_deterministic_eval.sh)
    loads checkpoints for all seeds/train splits and computes metrics.

3. __Evaluation.__ Most evaluation is covered from the two previous jobs. The
following are not covered, and which we use Robustness Metrics: ImageNet OOD
(i.e., for non-detection tasks).

    TODO(trandustin): Provide example command using this codebase's models.
