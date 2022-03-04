#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_ycb_points.py --dataset_root ./datasets/ycb/YCB_Video_Dataset\
  --model trained_models/ycb/pose_model_current.pth --refine_model trained_models/ycb/pose_refine_model_current.pth
