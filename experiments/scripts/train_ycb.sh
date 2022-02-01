#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset ycb\
  --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --workers 16\
  --batch_size 16
