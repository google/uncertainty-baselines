#!/bin/bash

# call eval model using wandb

# Debug on Mac OS X platform
use_gpu=False
if [ "$(uname)" = "Darwin" ] ; then
tpu=False
num_cores=1
batch_size=5
elif [ "$(uname)" = "Linux" ]; then
tpu='local'
num_cores=8
batch_size=16
fi

# default config for eval
use_wandb=True

for dataset  in "cityscapes"  #"ade20k_ind" "street_hazards"
do
for model in "deterministic" "gp" "het" "be" 
do
for rng_seed in 1
do
base_output_dir="gs://ub-ekb/segmenter/${dataset}"
config_file="configs/${dataset}/${model}.py"
run_name="${model}_eval"
output_dir="${base_output_dir}/${run_name}"
python deterministic.py \
--output_dir=${output_dir} \
--num_cores=$num_cores \
--use_gpu=$use_gpu \
--config=${config_file} \
--config.batch_size=${batch_size} \
--config.use_wandb=${use_wandb} \
--config.rng_seed=${rng_seed} \
--tpu=${tpu} \

done
done
done
