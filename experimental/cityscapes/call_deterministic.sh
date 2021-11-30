#!/bin/sh

config='experiments/imagenet21k_segmenter_cityscapes.py'
use_gpu=False

if [ "$(uname)" == "Darwin" ]; then
  # Do something under Mac OS X platform
  output_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/experimental/cityscapes/outputs"
  num_cores=0
  tpu='None'
python deterministic.py -- --output_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/experimental/cityscapes/outputs" --num_cores=0 --use_gpu=False --tpu=False --config='experiments/imagenet21k_segmenter_cityscapes.py'

elif [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
    # Do something under GNU/Linux platform
  output_dir="/home/ekellbuch/ub_ekb/experimental/cityscapes/outputs"
  tpu='local'
  num_cores=8
  python3 deterministic.py  -- --output_dir="/home/ekellbuch/ub_ekb/experimental/cityscapes/outputs" --num_cores=8 --tpu='local' --config='experiments/imagenet21k_segmenter_cityscapes.py'

fi

#python deterministic.py  "--output_dir=$output_dir --num_cores=$num_cores --use_gpu=$use_gpu --tpu=$tpu --config=$config"

