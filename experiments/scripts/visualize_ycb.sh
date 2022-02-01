#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_ycb.py --dataset_root C:/Users/OpenARK/Desktop/datasets/YCB_Video_Dataset\
  --model pose_model_current2.pth
