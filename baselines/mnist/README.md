# LeNet5 on (Fashion) MNIST

## MNIST

| Method | Test NLL | Test Accuracy | Test Cal. Error | Train Runtime (mins) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.037 | 99.1% | - | 10 (1 P100 GPU) | 60K |
| Ensemble (size=10)<sup>1</sup> | 0.016 | 99.4% | 0.005 | 10 (1 P100 GPU) | 600K |
| Refined VI<sup>2</sup> | 0.034 | 99.0% | 0.010 | - (1 P100 GPU) | - |
| Variational Inference | 0.048 | 98.5% | 0.010 | - (1 P100 GPU) | - |

## Fashion MNIST

| Method | Test NLL | Test Accuracy | Test Cal. Error | Train Runtime (mins) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| Deterministic | 0.321 | 91.0% | - | 10 (1 P100 GPU) | 60K |
| Ensemble (size=10)<sup>1</sup> | 0.200 | 93.2% | 0.010 | 10 (1 P100 GPU) | 600K |
| Refined VI<sup>2</sup> | 0.259 | 90.6% | 0.011 | - (1 P100 GPU) | - |
| Variational Inference | 0.292 | 89.5% | 0.011 | - (1 P100 GPU) | - |

1. Ensemble's binary is the same as deterministic's (`deterministic.py`). To reproduce results, use the following flags: `--ensemble_size=10`.
2. Refined VI's binary is the same as variational inference's (`variational_inference.py`). See the results `auxiliary_sampling_*` for refined VI and `base_model_*` for typical VI.
