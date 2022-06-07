#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/visualize_saliency.py --model "C:/Users/OpenARK/Desktop/adam/ycb trained 6d/6-7 resnet34/pose_model_current.pth"\
 --config "C:/Users/OpenARK/Desktop/adam/ycb trained 6d/6-7 resnet34/config_current.yaml"