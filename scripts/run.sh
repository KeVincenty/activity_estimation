#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

if [ $2 ]; then
  gpu=$2
else
  gpu="0"
fi

now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=${gpu} python main.py --config ${config} --exp_time ${now}