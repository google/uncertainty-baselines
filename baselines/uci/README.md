# Multilayer Perceptron on UCI Datasets

| Method | Boston Housing | Concrete Strength | Energy Efficiency | kin8nm | Naval Propulsion | Power Plant | Protein Structure | Wine | Yacht Hydrodynamics | Train Runtime (mins) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic<sup>1</sup> | 16.11 | 3.94 | 1.29 | -1.06 | -5.06 | 3.28 | 3.16 | 1.05 | 1.89 | - (1 P100 GPU) | 2K |
| Ensemble (size=10)<sup>1</sup> | 6.11 | 3.20 | 0.61 | -1.17 | -5.17 | 3.18 | 3.12 | 0.97 | 0.73 | - (1 P100 GPU) | 20K |
| Refined VI<sup>2</sup> | 2.94 | 3.08 | 0.75 | -1.06 | -6.33 | 2.83 | 2.92 | 0.97 | 1.68 | - (1 P100 GPU) | - |
| Variational Inference<sup>2</sup> | 3.12 | 3.22 | 0.93 | -1.03 | -6.12 | 2.85 | 2.93 | 1.00 | 2.01 | - (1 P100 GPU) | - |
| MC Dropout<sup>3</sup> | 2.40 | 2.93 | 1.21 | -1.14 | -4.45 | 2.80 | 2.87 | 0.93 | 1.25 | - | - |
| Variational Matrix Gaussian<sup>4</sup> | 2.46 | 3.01 | 1.06 | -1.10 | -2.46 | 2.82 | 2.84 | 0.95 | 1.30 | - | - |
| HS-BNN<sup>5</sup> | 2.54 | 3.09 | 2.66 | -1.12 | -5.52 | 2.81 | 2.89 | 0.95 | 2.33 | - | - |
| PBP-MV<sup>6</sup> | 2.54 | 3.04 | 1.01 | -1.28 | -4.85 | 2.78 | 2.77 | 0.97 | 1.64 | - | - |

## Metrics

We define metrics specific to UCI datasets below. For general metrics, see [`baselines/`](https://github.com/google/edward2/tree/master/baselines).

1. __UCI Dataset__. Negative-log-likelihood on the full test set and not averaged per data point.
2. __Train Runtime.__ Training runtime is the total wall-clock time to train the model, including any intermediate test set evaluations. It is averaged across UCI datasets.

Footnotes.

1. Ensemble's binary is the same as deterministic's (`deterministic.py`). To reproduce results, use the following flags: `--ensemble_size=10`. For both methods, use
`--training_steps=2500` for `--dataset=boston_housing`,
`--training_steps=7500` for `--dataset=concrete_strength`,
`--training_steps=7500` for `--dataset=energy_efficiency`,
`--training_steps=7500` for `--dataset=kin8nm`,
`--training_steps=7500` for `--dataset=naval_propulsion`,
`--training_steps=7500` for `--dataset=power_plant`,
`--training_steps=7500` for `--dataset=protein_structure`,
`--training_steps=2500` for `--dataset=wine`,
`--training_steps=7500` for `--dataset=yacht_hydrodynamics`.
2. Refined VI's binary is the same as variational inference's (`variational_inference.py`). See the result `auxiliary_sampling_test_probabilistic_log_likelihood` for refined VI and `base_model_*` for typical VI.
3. Monte Carlo Dropout (Gal and Ghahramani 2016) results from Mukhoti et al. (2018). Models are trained to convergence (4,000 epochs on each dataset), with hyperparameter values obtained by performing grid search over (softmax temperature, dropout probability); the best pair is chosen by performance on a validation set formed by randomly selecting 20% of the training set. Train/test split and model size follow Hernandez-Lobato and Adams (2015).
4. Variational Matrix Gaussian (VMG) results from Louizos and Welling (2016). Models are trained to convergence of training loss. Train/test split and model size follow Hernandez-Lobato and Adams (2015). 
5. Bayesian Neural Networks with Horseshoe priors (HS-BNN) results from Ghosh and Doshi-Velez (2017). Train/test split and model size follow Hernandez-Lobato and Adams (2015).
6. Probabilistic Backpropagation with a Matrix Variate Gaussian prior results from Sun et al. (2016). Models are trained to convergence. Train/test split follows Hernandez-Lobato and Adams (2015). Model size is notably non-standard; networks have two hidden layers instead of one layer as in the above three setups and Hernandez-Lobato and Adams (2015).