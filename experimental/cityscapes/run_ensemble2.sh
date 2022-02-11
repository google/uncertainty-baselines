#!/bin/bash

# train segmenter model on cityscapes using different pretrained backbones for different splits

function get_config()
{
  local config_file_name="experiments/sweep_vit32/imagenet21k_segmenter_cityscapes_deterministic.py"
  echo "$config_file_name"
}

function get_pretrained_backbone_path()
{
  local checkpoint_path="gs://ub-checkpoints/ImageNet21k_ViT-L32/$1/checkpoint.npz"
  echo "$checkpoint_path"
}

#base_output_dir="outputs/ensemble"
base_output_dir="gs://ub-ekb/segmenter/cityscapes/run_splits_vitl32/checkpoints"

declare CITYSCAPES_TRAIN_SIZE=(
  ["1"]="29"
  ["10"]="298"
  ["25"]="744"
  ["50"]="1488"
  ["75"]="2231"
  ["100"]="2975"
  )

# Debug on Mac OS X platform
use_gpu=False
if [ "$(uname)" = "Darwin" ] ; then
tpu=False
num_cores=0
batch_size=1
elif [ "$(uname)" = "Linux" ]; then
tpu='local'
num_cores=8
batch_size=8
fi
for num_training_epochs in 50 #30 50 150
do
for lr in "0.0001" # "0.03" "0.01" "0.003" "0.001"
do
for rng_seed in 0 1 2
do
for model_type in "deterministic"
do
for split in 100
do
  config_file=$(get_config $model_type)
  learning_rate=$( echo "$lr" | bc )
  run_name="${model_type}_split${split}_seed${rng_seed}_lr${learning_rate}_step${num_training_epochs}"
  output_dir="${base_output_dir}/${run_name}"
  train_split="train[:${split}%]"
  num_train_examples=${CITYSCAPES_TRAIN_SIZE[$split]}
  pretrained_backbone=$(get_pretrained_backbone_path $rng_seed)
  python deterministic.py \
  --output_dir=${output_dir} \
  --num_cores=$num_cores \
  --use_gpu=$use_gpu \
  --config=${config_file} \
  --config.rng_seed=${rng_seed}  \
  --config.dataset_configs.train_split=${train_split} \
  --config.dataset_configs.number_train_examples_debug=${num_train_examples} \
  --config.batch_size=${batch_size} \
  --tpu=${tpu} \
	--config.lr_configs.base_learning_rate=${learning_rate} \
	--config.num_training_epochs=${num_training_epochs} \
  --config.upstream_model=${model_type} \
  --config.pretrained_backbone_configs.checkpoint_path=${pretrained_backbone} \

done
done
done
done
done