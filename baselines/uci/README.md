# Multilayer Perceptron on UCI Datasets

| Method | Boston Housing | Concrete Strength | Energy Efficiency | kin8nm | Naval Propulsion | Power Plant | Protein Structure | Wine | Yacht Hydrodynamics | Train Runtime (mins) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic<sup>1</sup> | 16.11 | 3.94 | 1.29 | -1.06 | -5.06 | 3.28 | 3.16 | 1.05 | 1.89 | - (1 P100 GPU) | 2K |
| Ensemble (size=10)<sup>1</sup> | 6.11 | 3.20 | 0.61 | -1.17 | -5.17 | 3.18 | 3.12 | 0.97 | 0.73 | - (1 P100 GPU) | 20K |
| Refined VI<sup>2</sup> | 2.94 | 3.08 | 0.75 | -1.06 | -6.33 | 2.83 | 2.92 | 0.97 | 1.68 | - (1 P100 GPU) | - |
| Variational Inference<sup>2</sup> | 3.12 | 3.22 | 0.93 | -1.03 | -6.12 | 2.85 | 2.93 | 1.00 | 2.01 | - (1 P100 GPU) | - |

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
