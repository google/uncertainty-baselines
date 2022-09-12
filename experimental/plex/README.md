Plex: Towards Reliability Using Pretrained Large Model Extensions
---

This README and directory provides more information for the *Plex: Towards
Reliability using Pretrained Large Model Extensions* paper (Tran et al., 2022),
which can be found at https://goo.gle/plex-paper.

### Demo notebooks

A demo notebook for ViT-Plex can be found
[here](https://github.com/google/uncertainty-baselines/blob/main/experimental/plex/plex_vit_demo.ipynb).
The demo showcases basic mechanics for loading and using the pretrained and
finetuned ViT-Plex model checkpoints, as well as more advanced usecases
including zero-shot out-of-distribution detection.

A demo notebook for ViT-T5 is to be announced!

### Training scripts

All training scripts and configuration files for ViT-Plex can be found
[here](https://github.com/google/uncertainty-baselines/tree/main/baselines/jft).
Of note, the pretrained ViT-Plex Large model that was pretrained on ImageNet-21K
used [this
script](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/batchensemble.py)
and [this
config](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/experiments/vit_be/imagenet21k_be_vit_large_32.py),
and the finetuned ViT-Plex Large model that was finetuned on ImageNet2012 used
[this
script](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/plex.py)
and [this
config](https://github.com/google/uncertainty-baselines/blob/main/baselines/jft/experiments/vit_l32_plex_finetune.py).

Training scripts and configuration files for T5-Plex is to be announced soon!

All layer definitions are located
[here](https://github.com/google/edward2/tree/main/edward2/jax/nn) in Edward2
(Tran et al., 2018).

### Model checkpoints

| Model | Description | Location |
| ----- | ----------- | -------- |
| ViT-Plex L ImageNet21K | A ViT Large model pretrained on ImageNet21K with batch ensemble. | `gs://plex-paper/plex_vit_large_imagenet21k.npz` |
| ViT-Plex L ImageNet21K -> ImageNet | The above `ViT-Plex L ImageNet21K` model (which uses batch ensemble) finetuned on ImageNet with a heteroscedastic output layer. | `gs://plex-paper/plex_vit_large_imagenet21k_to_imagenet.npz` |
| T5-Plex S C4 -> MNLI | A pretrained T5 Small model (pretrained on C4) finetuned on MNLI with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_small_c4_to_mnli` |
| T5-Plex S C4 -> NaLUE | A pretrained T5 Small model (pretrained on C4) finetuned on NaLUE with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_small_c4_to_nalue` |
| T5-Plex S C4 -> Wiki Toxic Comments | A pretrained T5 Small model (pretrained on C4) finetuned on Wikipedia Toxicity Subtypes with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_small_c4_to_wiki_toxic_comments` |
| T5-Plex L C4 -> MNLI | A pretrained T5 Large model (pretrained on C4) finetuned on MNLI with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_large_c4_to_mnli` |
| T5-Plex L C4 -> NaLUE | A pretrained T5 Large model (pretrained on C4) finetuned on NaLUE with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_large_c4_to_nalue` |
| T5-Plex L C4 -> Wiki Toxic Comments | A pretrained T5 Large model (pretrained on C4) finetuned on Wikipedia Toxicity Subtypes with batch ensemble and an SNGP output layer. | `gs://plex-paper/plex_t5_large_c4_to_wiki_toxic_comments` |

### Paper plots

A notebook used to generate the figures from the paper can be found
[here](https://github.com/google/uncertainty-baselines/blob/main/experimental/plex/plots.ipynb).

### How to cite

A bibtex entry will be announced.
