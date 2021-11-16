#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset ycb\
  --dataset_root C:/Users/OpenARK/Desktop/datasets/YCB_Video_Dataset --workers 8\
  --resume_posenet pose_model_current.pth --start_epoch 12