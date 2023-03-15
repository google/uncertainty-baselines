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
batch_size=8
fi

# default config for eval
eval_covariate_shift=False
method_name='msp'
use_wandb=True

for dataset  in "ade20k_ind" "street_hazards" "cityscapes"
do
for model in "gp" "be" "deterministic" "het"
do
base_output_dir="gs://ub-ekb/segmenter/${dataset}"
config_file="configs/${dataset}/${model}_eval.py"
run_name="${model}_eval"
output_dir="${base_output_dir}/${run_name}"
python deterministic.py \
--output_dir=${output_dir} \
--num_cores=$num_cores \
--use_gpu=$use_gpu \
--config=${config_file} \
--config.batch_size=${batch_size} \
--config.eval_robustness_configs.method_name=${method_name} \
--config.eval_covariate_shift=${eval_covariate_shift} \
--config.use_wandb=${use_wandb} \
--tpu=${tpu} \

done
done
