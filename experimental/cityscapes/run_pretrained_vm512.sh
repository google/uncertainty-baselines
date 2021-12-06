#!/bin/bash

# train cityscapes using segmenter with pretrained backbone
# supports options to try 

declare -A configfiles=( ["deterministic"]="experiments/imagenet21k_segmenter_cityscapes512.py" ["sngp"]="experiments/imagenet21k_segmenter_cityscapes_sngp512.py" ["scratch"]="experiments/segmenter_cityscapes512.py")

num_cores=8
tpu='local'
use_gpu=False

for config_mode in "sngp" "scratch" #"deterministic" # "scratch"  "sngp" 
do
config_file="${configfiles[$config_mode]}"
output_dir="gs://ub-ekb/segmenter/cityscapes/run3/$config_mode"
echo "${output_dir} ${config_file}"
python3 deterministic.py --output_dir=${output_dir} \
	--num_cores=$num_cores \
	--use_gpu=$use_gpu \
	--config=${config_file} \
	--tpu=$tpu

done
