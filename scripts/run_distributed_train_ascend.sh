#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_distributed_train_ascend.sh RANK_TABLE_ADDR PRE_TRAIN_DATASET"
echo "for example:"
echo "sh run_distributed_train_ascend.sh \
  /home/workspace/rank_table_8p.json \
  /home/workspace/dataset_menu/train.tok.clean.bpe.32000.en.mindrecord"
echo "It is better to use absolute path."
echo "=============================================================================================================="

RANK_TABLE_ADDR=$1
PRE_TRAIN_DATASET=$2

current_exec_path=$(pwd)
echo ${current_exec_path}

export RANK_TABLE_FILE=$RANK_TABLE_ADDR
export MINDSPORE_HCCL_CONFIG_PATH=$RANK_TABLE_ADDR

echo $RANK_TABLE_FILE
export RANK_SIZE=8
export GLOG_v=2

for((i=0;i<=7;i++));
do
    rm -rf ${current_exec_path}/device$i
    mkdir ${current_exec_path}/device$i
    cd ${current_exec_path}/device$i || exit
    cp ../../*.py .
    cp -r ../../src .
    cp -r ../../config .
    export RANK_ID=$i
    export DEVICE_ID=$i
  python ../../train.py \
    --is_modelarts=False \
    --config=${current_exec_path}/device${i}/config/config.json \
    --pre_train_dataset=$PRE_TRAIN_DATASET > log_seq2seq_network${i}.log 2>&1 &
    cd ${current_exec_path} || exit
done
cd ${current_exec_path} || exit
