
# Log

[x] add cityscapes dataset loader. <br />
[x] add deterministic training for segmenter.  <br />
[x] include transfer learning option: init from pretrained backbone. <br />
[x] include option to train vit+ model using different train split. <br />
[x] add pavpu metric. <br />
[x] calculate uncertainty metrics. <br />

[x] add run with vit l-32 backbone: run_l32_splits_vm.sh <br /> 
[x] add eval for vit l-32 models: run_deterministic_eval_l32.sh <br />

## [Wandb integration ](https://docs.wandb.ai/guides/sweeps/configuration) <br />
[x] Visualize results in wandb: run_ensemble.sh <br />
[x] Hyperparameter sweep: experiments/toy/toy_sweep  <br />

```
wandb sweep experiments/toy/toy_sweep.yaml
wandb agent ${SWEEPID}
```

## Experiments

Fully implemented: <br />

[x] Vanilla deterministic upstream + deterministic downstream.  <br />
Given a deterministic model trained on imagenet21k, 
replace the decoder by a segmentation decoder and finetune the model on cityscapes.
```
wandb sweep experiments/sweep_vit32/imagenet21k_segmenter_cityscapes_deterministic.yaml
wandb agent <USERNAME/PROJECTNAME/SWEEPID>
```
Once the models have trained independently, we can evaluate the results by running:   <br />
```
./run_deterministic_eval_l32.sh
```

Missing wandb configuration: <br />

[x] Ensemble (ensemble upstream + ensemble downstream).  <br />
Given E deterministic models trained on imagenet21k, 
replace the E decoders in each model by E new segmentations encoders 
(This step is achieved by calling get_pretrained_backbone_path)
Finetune each model separately on cityscapes
Then, aggregate the results.

```
./run_ensemble2.sh
```
Once the models have trained independently, we can aggregate the results by running:   <br />
```
./run_ensemble_eval.sh
```
[] Ensemble (ensemble upstream + deterministic downstream).  <br />
Given E deterministic models trained on imagenet21k, 
replace the E decoders in each model by 1 new segmentations encoder.
Finetune the new model on cityscapes.
```

```
[] Batch Ensemble (batch ensemble upstream + deterministic downstream).  <br />
Given a BE deterministic model trained on imagenet21k,
replace the MLP blocks in the encoder by 
replace the E decoders in each model by rank-1 decoder which to get outputs [N, E, K].
Finetune the new model on cityscapes.
Average over E to get the results.
```

```

To compare parameter between vit and vit_be model run:
```
python -m unittest -v uncertainty_baselines/models/vit_batchensemble_test.py

```




