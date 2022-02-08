#!/bin/bash

: '
train segmenter model on cityscapes using different pretrained backbones for different splits

Other parameters: number of training epochs, learning_rate, train_mode.


To List checkpoints run:
gsutil ls gs://ub-ekb/segmenter/cityscapes/run_splits_l32



'

function get_config()
{
  local config_file_name="experiments/splits_l32/imagenet21k_segmenter_cityscapes_$1_$2.py"
  echo "$config_file_name"
}

function get_pretrained_backbone_path()
{
  local checkpoint_path="gs://ub-checkpoints/ImageNet21k_ViT-L32/$1/checkpoint.npz"
  echo "$checkpoint_path"
}
num_cores=8
tpu='local'
use_gpu=False

for num_training_epochs in 100 #30 50 150
do
for lr in "0.0001" # "0.03" "0.01" "0.003" "0.001"
do
for rng_seed in 1 2 3 4
do
for train_mode in "deterministic" #"scratch"
do
for train_split in 100 #  75 50 25 10
do
learning_rate=$( echo "$lr" | bc )
config_file=$(get_config $train_mode $train_split)
run_name="${train_mode}_split${train_split}_seed${rng_seed}_lr${learning_rate}_step${num_training_epochs}"
output_dir_ckpt="gs://ub-ekb/segmenter/cityscapes/run_splits_vitl32/checkpoints/${run_name}"
pretrained_backbone=$(get_pretrained_backbone_path $rng_seed)
echo "${pretrained_backbone}"
echo "Running experiment ${output_dir_ckpt}"
#: '
python3 deterministic.py --output_dir=${output_dir_ckpt} \
	--num_cores=$num_cores \
	--use_gpu=$use_gpu \
	--config=${config_file} \
	--config.rng_seed=${rng_seed} \
	--config.lr_configs.base_learning_rate=${learning_rate} \
	--config.num_training_epochs=${num_training_epochs} \
	--tpu=$tpu
#	--config.pretrained_backbone_configs.checkpoint_path=${pretrained_backbone} \
#'
done
done
done
done
done
