#!/bin/bash

# train cityscapes using segmenter with pretrained backbone
# deterministic splits

#declare -A configfiles=( [75]="experiments/splits/imagenet21k_segmenter_cityscapes75.py" ["sngp"]="experiments/imagenet21k_segmenter_cityscapes_sngp.py" ["scratch"]="experiments/segmenter_cityscapes.py")

function get_config()
{
  local config_file_name="experiments/splits/imagenet21k_segmenter_cityscapes_$1_$2.py"
  echo "$config_file_name"
}

num_cores=8
tpu='local'
use_gpu=False


for rng_seed in 0
do
for train_mode in "deterministic" "gp"
do
for train_split in 10 100 75 50 25
do
config_file=$(get_config $train_mode $train_split)   # or result=`myfunc`
run_name="${train_mode}_split${train_split}_seed${rng_seed}"
output_dir="gs://ub-ekb/segmenter/cityscapes/run_splits/${run_name}"
echo "${output_dir} ${config_file}"
python3 deterministic.py --output_dir=${output_dir} \
	--num_cores=$num_cores \
	--use_gpu=$use_gpu \
	--config=${config_file} \
	--config.rng_seed=${rng_seed} \
	--tpu=$tpu
done
done
done
exit
