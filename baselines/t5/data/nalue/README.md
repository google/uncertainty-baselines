# NaLUE: Natural Language Understanding Evaluation task.

This folder implements data utilities for the Natural Language Understanding
Evaluation (NaLUE) task. Specifically, the NaLUE task combines data source
from three multi-domain intent understanding datasets ([CLINC150](https://aclanthology.org/D19-1131.pdf), [Banking77](https://arxiv.org/pdf/2003.04807v1.pdf), [HWU64](https://link.springer.com/chapter/10.1007/978-981-15-9323-9_15)) to create a
large compositional intent understanding datasets with ~30K utterances and ~260 
intents, where each utterance corresponds to a compositional label of three tokens
(vertical_name, domain_name, intent_name).
