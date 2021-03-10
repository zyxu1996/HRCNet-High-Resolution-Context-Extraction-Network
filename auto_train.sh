#!/bin/sh

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port 29504 train.py
echo 'train_end'


