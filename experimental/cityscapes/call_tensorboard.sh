#!/bin/sh

# map gsbucket to local
# ~/go/bin/gcsfuse --only-dir segmenter/cityscapes/run0 ub-ekb run0

# read local directory:
results_dir="/Users/ekellbuch/Projects/ood_segmentation/ub_ekb/gsbucket_out/run0"
for d in results_dir ; do
    echo "$d"
done

tensorboard --logdir ${results_dir} --reload_multifile True

