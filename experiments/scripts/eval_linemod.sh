#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_linemod.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model archived_trained_models/linemod/9-27/trained_models/pose_model_current.pth\
  --refine_model archived_trained_models/linemod/9-27/trained_models/pose_refine_model_current.pth