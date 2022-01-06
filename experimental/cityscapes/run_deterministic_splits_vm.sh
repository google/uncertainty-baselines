#!/bin/bash

# train segmenter model on cityscapes using different pretrained backbones for different splits

function get_config()
{
  local config_file_name="experiments/splits/imagenet21k_segmenter_cityscapes_$1_$2.py"
  echo "$config_file_name"
}

num_cores=8
tpu='local'
use_gpu=False


for rng_seed in 4 
do
for train_mode in "deterministic" "gp" "scratch"
do
for train_split in 100 75 50 25 10
do
config_file=$(get_config $train_mode $train_split)
run_name="${train_mode}_split${train_split}_seed${rng_seed}"
output_dir_ckpt="gs://ub-ekb/segmenter/cityscapes/run_splits1/checkpoints/${run_name}"
echo "Running experiment ${output_dir_ckpt}"
python3 deterministic.py --output_dir=${output_dir_ckpt} \
	--num_cores=$num_cores \
	--use_gpu=$use_gpu \
	--config=${config_file} \
	--config.rng_seed=${rng_seed} \
	--tpu=$tpu
done
done
done
