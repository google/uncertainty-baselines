#!/bin/bash

# train cityscapes using segmenter with pretrained backbone
# supports 2 options to

function get_config()
{
  local config_file_name="experiments/splits_l32/imagenet21k_segmenter_cityscapes_$1_$2.py"
  echo "$config_file_name"
}

if [ "$(uname)" = "Darwin" ] ; then
  # Do something under Mac OS X platform
  config_file='experiments/imagenet21k_segmenter_cityscapes1.py'
  output_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/experimental/cityscapes/outputs"
  num_cores=0
  tpu=False
  use_gpu=False
  python deterministic_eval.py --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  # --tpu=$tpu
elif [ "$(uname)" = "Linux" ]; then
  echo "in Linux"
  train_mode="deterministic"
  train_split=100
  rng_seed=0
  config_file=$(get_config $train_mode $train_split)
  run_name="${train_mode}_split${train_split}_seed${rng_seed}"
  #config_file='experiments/imagenet21k_segmenter_cityscapes13.py' 
  #output_dir="/home/ekellbuch/ub_ekb/experimental/cityscapes/outputs13"
  output_dir="gs://ub-ekb/segmenter/cityscapes/run_splits_l32/checkpoints/${run_name}"
  num_cores=8
  tpu='local'
  use_gpu=False
  python3 deterministic_eval_l32.py --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  --tpu=$tpu
#  --config.batch_size=${batch_size} \

fi

#%%
