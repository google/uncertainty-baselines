#!/bin/sh

# train cityscapes using segmenter with pretrained backbone
# supports 2 options to


if [ "$(uname)" = "Darwin" ] ; then
  # Do something under Mac OS X platform
  config_file='experiments/imagenet21k_segmenter_cityscapes12.py'
  output_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/experimental/cityscapes/outputs"
  num_cores=0
  tpu=False
  use_gpu=False
  python deterministic.py --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  # --tpu=$tpu
elif [ "$(uname)" = "Linux" ]; then
  echo "in Linux"
  config_file='experiments/imagenet21k_segmenter_cityscapes13.py'
  output_dir="/home/ekellbuch/ub_ekb/experimental/cityscapes/outputs13"
  num_cores=8
  tpu='local'
  use_gpu=False
  python3 deterministic.py --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  --tpu=$tpu
fi
