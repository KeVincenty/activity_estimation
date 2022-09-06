#!/usr/bin/env bash
if [ -f $1 ]; then
  config=$1
else
  echo "need a config file"
  exit
fi

if [ -f $2 ]; then
  model=$2
else
  echo "need a model file"
  exit
fi

if [ $3 ]; then
  gpu=$3
else
  gpu="0"
fi

CUDA_VISIBLE_DEVICES=${gpu} python inference.py --config ${config} --model_path ${model}