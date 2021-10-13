#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"

python3 ./tools/train.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed