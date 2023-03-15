#!/bin/bash

# evaluate model using wandb
#wandb sweep run_toy_mac.yaml
# before make sure we can run code vanilla version:

DATASET='ade20k_ind'  # or cityscapes
DATASET='street_hazards'

# Parameters
DATASET='cityscapes'
model='deterministic'

base_output_dir="gs://ub-ekb/segmenter/${DATASET}/${model}_eval"

# Debug on Mac OS X platform
use_gpu=False
if [ "$(uname)" = "Darwin" ] ; then
tpu=False
num_cores=1
batch_size=1
elif [ "$(uname)" = "Linux" ]; then
tpu='local'
num_cores=8
batch_size=8
fi

config_file="configs/${DATASET}/${model}_eval.py:runlocal"
run_name="${model}_eval"
output_dir="${base_output_dir}/${run_name}"
python deterministic.py \
--output_dir=${output_dir} \
--num_cores=$num_cores \
--use_gpu=$use_gpu \
--config=${config_file} \
--config.batch_size=${batch_size} \
--tpu=${tpu} \
