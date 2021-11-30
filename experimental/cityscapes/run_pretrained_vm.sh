#!/bin/bash

# train cityscapes using segmenter with pretrained backbone
# supports 2 options to

declare -A configfiles=( ["deterministic"]="experiments/imagenet21k_segmenter_cityscapes.py" ["sngp"]="experiments/imagenet21k_segmenter_cityscapes_sngp.py" ["scratch"]="experiments/segmenter_cityscapes.py")

num_cores=8
tpu='local'
use_gpu=False

for config_mode in "sngp" "deterministic" "scratch"
do
config_file="${configfiles[$config_mode]}"
output_dir="gs://ub-ekb/segmenter/cityscapes/run0/$config_mode"
echo "${output_dir} ${config_file}"
python3 deterministic.py --output_dir=${output_dir} \
	--num_cores=$num_cores \
	--use_gpu=$use_gpu \
	--config=${config_file} \
	--tpu=$tpu

done
