#!/bin/bash

# train toy model using wandb
#wandb sweep run_toy_mac.yaml
# before make sure we can run code vanilla version:

DATASET='ade20k_ind'  # or cityscapes
DATASET='cityscapes'

base_output_dir="gs://ub-ekb/segmenter/${DATASET}/toy_model"

# Debug on Mac OS X platform
use_gpu=False
if [ "$(uname)" = "Darwin" ] ; then
tpu=False
num_cores=1
batch_size=5
elif [ "$(uname)" = "Linux" ]; then
tpu='local'
num_cores=8
batch_size=8
fi

config_file="configs/${DATASET}/toy_model.py:runlocal"
run_name="toy_model"
output_dir="${base_output_dir}/${run_name}"
python deterministic.py \
--output_dir=${output_dir} \
--num_cores=$num_cores \
--use_gpu=$use_gpu \
--config=${config_file} \
--config.batch_size=${batch_size} \
--tpu=${tpu} \
