A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models
___

This README and directory provides more information for the [*A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models*](https://arxiv.org/abs/2302.06235) ICML 2023 paper (Allingham et al., 2023).

### Results notebook

All of the CLIP results in the paper (and more) can be reproduced using the `Zero-shot prompt ensembling for text-image models results.ipynb` notebook.

### Classnames and Prompts

All of the classnames and prompts used in the paper can be found [here](https://github.com/google/uncertainty-baselines/blob/main/experimental/multimodal/multimodal_utils.py).

### CLIP implementation

The CLIP implementation is adapted from Scenic, and can be found [here](https://github.com/google/uncertainty-baselines/blob/main/uncertainty_baselines/models/clip.py).

### CLIP checkpoints

The CLIP checkpoints used in the paper are those published by OpenAI. Code for downloading and converting them from PyTorch to JAX can be found [here](https://github.com/google-research/scenic/blob/main/scenic/projects/baselines/clip/model.py).

### LAION-400m dataset

We make use of (the first 20k) images in the LAION-400m dataset. Instructions for accessing the dataset using `tensorflow_datasets` are [here](https://www.tensorflow.org/datasets/catalog/laion400m). Instructions for downloading the dataset are [here](https://laion.ai/blog/laion-400-open-dataset/).

### How to cite

If you use this code in your work, please cite the paper:

```
@InProceedings{allingham2023simple,
  title =    {A Simple Zero-shot Prompt Weighting Technique to Improve Prompt Ensembling in Text-Image Models},
  author =       {Allingham, James Urquhart and Ren, Jie and Dusenberry, Michael W and Gu, Xiuye and Cui, Yin and Tran, Dustin and Liu, Jeremiah Zhe and Lakshminarayanan, Balaji},
  booktitle =    {Proceedings of the 40th International Conference on Machine Learning},
  pages =    {547--568},
  year =     {2023},
  editor =   {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume =   {202},
  series =   {Proceedings of Machine Learning Research},
  month =    {23--29 Jul},
  publisher =    {PMLR},
}
```