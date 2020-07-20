#!/bin/bash

source /mnt/sdb/jingxu1/framework/pytorch/venvs/venv_pytorch_jing_src_py37/bin/activate
cd /mnt/sdb/jingxu1/framework/pytorch/exps/pytorch_benchmark/PeleeNet
OMP_NUM_THREADS=28 numactl -N 0 -m 0 python main.py /mnt/sdb/jingxu1/datasets/imagenet/imagenet -e -b 128 --pretrained
