# Robust segvit

*Robust_segvit* is a codebase to evaluate the robustness of semantic segmentation models.

Robust_segvit is developed in [JAX](https://github.com/google/jax) and uses [Flax](https://github.com/google/flax), [uncertainty_baselines](https://github.com/google/uncertainty-baselines) and [Scenic](https://github.com/google-research/scenic).

## Code structure
See uncertainty_baselines/google/experimental/cityscapes.


## Cityscapes

We investigate the performance of different reliability methods on image segmentation tasks. <br>

[x] configs/cityscapes: contains experiment configurations for the cityscapes dataset. <br>


## Debugging:

To run the code on cpu, install the dependencies as in:
[x] Copy ananconda environment
[x] Install jaxlib, jax, flax from source
[x] Install scenic from source
[x] Install uncertainty_baselines from source

## Issues
[] Fails to read segmenter_be model.
