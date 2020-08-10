# Clinc Intent Detection

## Deterministic

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| TextCNN | 0.0145 / 0.2626 | 99.8% / 94.4% | 0.0035 / 0.0270 | 2.0 (8 TPUv2 cores) | 2.45M |
| BERT-tiny (2 layer, 128 unit) | 0.0001 / 0.4517 | 99.9% / 94.5% | 0.0001 / 0.0381 | 1.0 (8 TPUv3 cores) | 4M |
| BERT-small (4 layer, 256 unit) | 0.0001 / 0.2796 | 99.9% / 96.8% | 0.0001 / 0.0286 | 1.0 (8 TPUv3 cores) | 11M |
| BERT-medium (8 layer, 512 unit) | 0.0002 / 0.3617 | 99.9% / 97.6% | 0.0001 / 0.0256 | 1.0 (8 TPUv3 cores) | 41M |
| BERT-base (12 layer, 768 unit) | 0.0002 / 0.1854 | 99.9% / 97.7% | 0.0001 / 0.0187 | 1.0 (8 TPUv3 cores) | 110M |
| BERT-large (24 layer, 1024 unit) | 0.0001 / 0.1402 | 99.9% / 98.1% | 0.0001 / 0.0236 | 2.0 (8 TPUv3 cores) | 340M |

## Ensemble (10 Models, Parallel Training)

| Method | Train/Test NLL | Train/Test Accuracy | Train/Test Cal. Error | Train Runtime (hours) | # Parameters |
| ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| TextCNN | - / - | - / - | - / - | - (8 TPUv2 cores) | 25M |
| BERT-tiny (2 layer, 128 unit) | 0.0001 / 0.3326 | 99.9% / 94.7%  | 0.0001 / 0.0226 | 1.0 (8 TPUv3 cores) | 40M |
| BERT-small (4 layer, 256 unit) | 0.0001 / 0.2130 | 99.9% / 96.1% | 0.0001 / 0.0148 | 1.0 (8 TPUv3 cores) | 110M |
| BERT-medium (8 layer, 512 unit) | 0.0002 / 0.1902 | 99.9% / 96.2% | 0.0001 / 0.0111 | 1.1 (8 TPUv3 cores) | 410M |
| BERT-base (12 layer, 768 unit) | 0.0001 / 0.1691 | 99.9% / 97.5% | 0.0001 / 0.0128 | 1.2 (8 TPUv3 cores) | 1,100M |
| BERT-large (24 layer, 1024 unit) | 0.0001 / 0.1605 | 99.9% / 97.8% | 0.0001 / 0.0110 | 2.4 (8 TPUv3 cores) | 3,400M |
