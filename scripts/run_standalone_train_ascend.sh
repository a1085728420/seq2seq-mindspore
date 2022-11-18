#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_standalone_train_ascend.sh PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_standalone_train_ascend.sh \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord"
echo "It is better to use absolute path."
echo "=============================================================================================================="

PRE_TRAIN_DATASET=$1

export GLOG_v=2

current_exec_path=$(pwd)
echo ${current_exec_path}
if [ -d "train" ];
then
    rm -rf ./train
fi
mkdir ./train
cp ../*.py ./train
cp -r ../src ./train
cp -r ../config ./train
cd ./train || exit
echo "start for training"
env > env.log
python train.py \
  --is_modelarts=False \
  --config=${current_exec_path}/train/config/config.json \
  --pre_train_dataset=$PRE_TRAIN_DATASET > log_seq2seq_network.log 2>&1 &
cd ..
