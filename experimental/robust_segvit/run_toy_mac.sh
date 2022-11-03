#!/bin/bash

# ----------------------------------------------------
# train toy model on a DATASET:
# ----------------------------------------------------

# to train toy model and track performance using wandb:
# wandb sweep run_toy_mac.yaml

DATASET='ade20k_ind'
DATASET='cityscapes'
DATASET='street_hazards'

# ----------------------------------------------------
# Set directory where outputs should be installed:
# ----------------------------------------------------
# can write results directly to gcp bucket
# base_output_dir="gs://ub-ekb/segmenter/${DATASET}/toy_model"
dt=$(date +"%Y-%m-%d-%H-%M-%S")

base_output_dir="results/${DATASET}"

run_name="toy_model"
output_dir="${base_output_dir}/${run_name}/${dt}"
# ----------------------------------------------------
# Set device configuration for Mac OS X platform
# or TPU v2-8/v3-8 frameworks.
# ----------------------------------------------------
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

# ----------------------------------------------------
# Set configuration file
# ----------------------------------------------------
config_file="configs/${DATASET}/toy_model.py:runlocal"

# ----------------------------------------------------
# Call model trainer.
# ----------------------------------------------------
python deterministic.py \
--output_dir=${output_dir} \
--num_cores=$num_cores \
--use_gpu=$use_gpu \
--config=${config_file} \
--config.batch_size=${batch_size} \
--tpu=${tpu} \
