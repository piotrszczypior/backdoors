#!/usr/bin/env bash
set -e

####
# Based on:
# https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh
###

#########################
# ImageNet training set
####

mkdir -p data/train
mv ILSVRC2012_img_train.tar data/train/ 
cd data/train

tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar

# For each .tar file: 
#   1. create directory with same name as .tar file
#   2. extract and copy contents of .tar file into directory
#   3. remove .tar file
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done

# This results in a training directory like so:
#
#  data/train/
#  ├── n01440764
#  │   ├── n01440764_10026.JPEG
#  │   ├── n01440764_10027.JPEG
#  │   ├── ......
#  ├── ......

cd ../..

#########################
# ImageNet validation set
####

mkdir -p data/val
mv ILSVRC2012_img_val.tar data/val/ 
cd data/val

# Extract the validation data and move images to subfolders:
tar -xvf  ILSVRC2012_img_val.tar -C .
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

# This results in a validation directory like so:
#
#  data/val
#  ├── n01440764
#  │   ├── ILSVRC2012_val_00000293.JPEG
#  │   ├── ILSVRC2012_val_00002138.JPEG
#  │   ├── ......
#  ├── ......

