# Robust segvit

**Robust_segvit** is a codebase to evaluate the robustness of semantic segmentation models.  The code is built on top of [uncertainty_baselines](https://github.com/google/uncertainty-baselines) and [Scenic](https://github.com/google-research/scenic). 

## Installation
Robust_segvit is developed in [JAX](https://github.com/google/jax)/[Flax](https://github.com/google/flax).

To run the code: <br>
1. Install [uncertainty_baselines](https://github.com/google/uncertainty-baselines). <br>
2. Install [Scenic](https://github.com/google-research/scenic). <br>
3. Follow the instructions for a toy run in [./run_deterministic_mac.sh]().

## Datasets
The experiment configurations for the different datasets are in:

- configs/cityscapes: Cityscapes dataset. <br>
- configs/ade20k_ind: ADE20k_ind dataset. <br>
- configs/street_hazards: Street Hazards dataset. <br>

## Comments:
- The checkpoint used for finetuning is the same the original segmenter model: [vit_large_patch16_384](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

## Citing work:

If you reference this code, please cite [our paper](https://github.com/google/uncertainty-baselines). <br>