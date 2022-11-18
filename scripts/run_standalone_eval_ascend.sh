#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "sh run_standalone_eval_ascend.sh TEST_DATASET EXISTED_CKPT_PATH \
  VOCAB_ADDR BPE_CODE_ADDR TEST_TARGET"
echo "for example:"
echo "sh run_standalone_eval_ascend.sh \
  /home/workspace/dataset_menu/newstest2014.en.mindrecord \
  /home/workspace/seq2seq/seq2seq-8_3452.ckpt \
  /home/workspace/wmt14_fr_en/vocab.bpe.32000 \
  /home/workspace/wmt14_fr_en/bpe.32000 \
  /home/workspace/wmt14_fr_en/newstest2014.fr"
echo "It is better to use absolute path."
echo "=============================================================================================================="

TEST_DATASET=$1
EXISTED_CKPT_PATH=$2
VOCAB_ADDR=$3
BPE_CODE_ADDR=$4
TEST_TARGET=$5

current_exec_path=$(pwd)
echo ${current_exec_path}


export GLOG_v=2

if [ -d "eval" ];
then
    rm -rf ./eval
fi
mkdir ./eval
cp ../*.py ./eval
cp -r ../src ./eval
cp -r ../config ./eval
cd ./eval || exit
echo "start for evaluation"
env > env.log
python3 eval.py \
  --config=${current_exec_path}/eval/config/config_test.json \
  --test_dataset=$TEST_DATASET \
  --existed_ckpt=$EXISTED_CKPT_PATH \
  --vocab=$VOCAB_ADDR \
  --bpe_codes=$BPE_CODE_ADDR \
  --test_tgt=$TEST_TARGET >log_infer.log 2>&1 &
cd ..
