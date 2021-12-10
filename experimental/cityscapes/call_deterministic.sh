#!/bin/bash

echo
if [ "$(uname)" == "Darwin" ]; then
  echo "Debug On mac"
  # Do something under Mac OS X platform
  output_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/experimental/cityscapes/outputs"
  config_file="experiments/toy/segmenter_cityscapes.py"
  num_cores=0
  tpu='None'
  use_gpu=False
  rng_seed=2
  python3 deterministic.py --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
	--config=${config_file} \
	--config.rng_seed=${rng_seed} \


elif [ "$(uname)" == "Linux" ]; then
  echo "run run_pretrained_vm.sh instead"
fi

#python deterministic.py  "--output_dir=$output_dir --num_cores=$num_cores --use_gpu=$use_gpu --tpu=$tpu --config=$config"

