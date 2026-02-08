#!/usr/bin/env bash
set -e

####
# Based on:
# https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
###

cd data

## ImageNet validation set
mkdir -p imagenet
tar -xvf  ILSVRC2012_img_val.tar -C imagenet

# Extract the validation data and move images to subfolders:
cd imagenet
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# This results in a validation directory like so:
#
#  imagenet/
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......

