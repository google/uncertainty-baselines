
# Log

[x] add cityscapes dataset loader. <br />
[x] add deterministic training for segmenter.  <br />
[x] include transfer learning option: init from pretrained backbone. <br />
[x] include option to train vit+ model using different train split. <br />
[x] add pavpu metric. <br />
[x] calculate uncertainty metrics. <br />

[x] add run with vit l-32 backbone: run_l32_splits_vm.sh <br /> 
[x] add eval for vit l-32 models: run_deterministic_eval_l32.sh <br />

Under development (no tpu compatibility)
[x] add run to train ensemble models: run_ensemble.sh <br />
[x] add early stopping flag <br />
[] Eval ensemble models: run_ensemble_eval <br />

[Wandb integration ](https://docs.wandb.ai/guides/sweeps/configuration) <br />
[x] Visualize results in wandb: run_ensemble.sh <br />
[x] Hyperparameter sweep: experiments/toy/toy_sweep  <br />

```
wandb sweep experiments/toy/toy_sweep.yaml
wandb agent ${SWEEPID}
```

Code to run: <br />
[] Vanilla deterministic upstream + deterministic downstream.  <br />
```
wandb sweep experiments/sweep_vit32/imagenet21k_segmenter_cityscapes_deterministic.yaml
wandb agent ${SWEEPID}
```
[] Ensemble (ensemble upstream + ensemble downstream).  <br />
[] Ensemble (ensemble upstream + deterministic downstream).  <br />
[] Ensemble (BatchEnsemble upstream + deterministic downstream). <br />


