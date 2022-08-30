#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

now=$(date +"%Y%m%d_%H%M%S")

CUDA_VISIBLE_DEVICES=$2 python main.py  --config ${config} --exp_time ${now}