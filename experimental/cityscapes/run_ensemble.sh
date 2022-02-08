#!/bin/sh

# Run deterministic

base_output_dir="outputs/ensemble"

declare CITYSCAPES_TRAIN_SIZE=(
  ["1"]="29"
  ["10"]="298"
  ["25"]="744"
  ["50"]="1488"
  ["75"]="2231"
  ["100"]="2975"
  )

if [ "$(uname)" = "Darwin" ] ; then
  # Do something under Mac OS X platform
for split in 1
do
for model_type in "scratch"
#for model_type in "deterministic"
do
for rng_seed in 0 1 2
do
  config_file="experiments/imagenet21k_segmenter_cityscapes2.py"
  output_dir="${base_output_dir}/${model_type}_split${split}_seed${rng_seed}"
  num_cores=0
  tpu=False
  use_gpu=False
  train_split="train[:${split}%]"
  num_train_examples=${CITYSCAPES_TRAIN_SIZE[$split]}
  python deterministic.py \
  --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  --config.rng_seed=${rng_seed}  \
  --config.dataset_configs.train_split=${train_split} \
  --config.dataset_configs.number_train_examples_debug=${num_train_examples} \
  #--config.batch_size=8 \
  #--config.upstream_model=${model_type} \
  #

done
done
done
elif [ "$(uname)" = "Linux" ]; then
  echo "in Linux"
  config_file='experiments/imagenet21k_segmenter_cityscapes1.py'
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
