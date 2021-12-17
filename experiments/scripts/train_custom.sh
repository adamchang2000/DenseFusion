#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python ./tools/train.py --dataset custom\
  --dataset_root ./datasets/custom/custom_preprocessed\
  --batch_size 32 --workers 8
