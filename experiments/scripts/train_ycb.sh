#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset ycb\
  --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --resume_posenet pose_model_current.pth --start_epoch 20