#!/usr/bin/env bash

SERVER_NUM=$1
GPU_NUM_PER_SERVER=$2
MODEL=$3
DISTRIBUTION=$4
ROUND=$5
EPOCH=$6
BATCH_SIZE=$7
CLIENT_NUM=$8
WORKER_NUM=$9
STAGE=${10}
LR=${11}
LOCAL_LR=${12}
FL_TYPE=${13}
PROJ_START=${14}
PROJ_RECOV=${15}
DEBUG=${16}
REG_LAMBDA=${17}
DESIGN=${18}
DATASET=${19}
DATA_DIR=${20}
GPU_STARTING=${21}
RUN_ID=${22}
SEED=${23}
hostname > mpi_host_file

PROCESS_NUM=`expr $WORKER_NUM + 1`
echo $PROCESS_NUM
# # sh run_spider_search.sh 1 8 darts hetero 1000 1 32 8 8 search 0.01 0.01 True 30 20 0 0.01 3 cifar100 './../../../data/cifar100' 0 900 0
mpirun -np $PROCESS_NUM -hostfile ./mpi_host_file python3 ./main_spider.py \
  --gpu_server_num $SERVER_NUM \
  --gpu_num_per_server $GPU_NUM_PER_SERVER \
  --model $MODEL \
  --dataset $DATASET \
  --partition_method $DISTRIBUTION  \
  --client_number $CLIENT_NUM \
  --client_num_in_total $CLIENT_NUM\
  --client_num_per_round $WORKER_NUM\
  --comm_round $ROUND \
  --epochs $EPOCH \
  --batch_size $BATCH_SIZE \
  --stage $STAGE \
  --lr $LR \
  --local_lr $LOCAL_LR\
  --FL $FL_TYPE \
  --proj_start $PROJ_START \
  --proj_recovery $PROJ_RECOV \
  --is_debug_mode $DEBUG \
  --pssl_lambda $REG_LAMBDA \
  --fednas_design $DESIGN \
  --data_dir $DATA_DIR \
  --gpu_starting $GPU_STARTING \
  --run_id $RUN_ID \
  --seed $SEED
