#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# Command: bash run_scripts/muge_finetune_vit-b-16_rbt-base.sh ${DATAPATH}

# Number of GPUs per GPU worker
GPUS_PER_NODE=1
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=1
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=localhost
# The port for communication
export MASTER_PORT=8514
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0

export PYTHONPATH=${PYTHONPATH}:`pwd`/cn_clip/

DATAPATH='/kaggle/input/fork-of-data-preprocess2'

# data options
train_data=${DATAPATH}/fj1/lmdb/train
val_data=${DATAPATH}/fj1/lmdb/valid # if val_data is not specif  ied, the validation will be automatically disabled

# restore options
resume=/kaggle/working/pretrained_weights/clip_cn_vit-h-14.pt # or specify your customed ckpt path to resume
#resume=/kaggle/input/layer12-epoch6/epoch_latest.pt
reset_data_offset="--reset-data-offset"
reset_optimizer="--reset-optimizer"
# reset_optimizer=""

# output options
output_base_dir=/kaggle/working/experiments/
name=fj1-cn_finetune_vit-h-14_roberta-large_bs1024_1gpu
save_step_frequency=999999 # disable it
save_epoch_frequency=1
log_interval=1
report_training_batch_acc="--report-training-batch-acc"
# report_training_batch_acc=""

# training hyper-params
context_length=52
warmup=6
#128
batch_size=128
valid_batch_size=128
#32
accum_freq=32
lr=3e-5
wd=0.001
max_epochs=3
valid_step_interval=999999
valid_epoch_interval=1
vision_model=ViT-H-14
text_model=RoBERTa-wwm-ext-large-chinese
use_augment="--use-augment"
freeze_vision="--freeze-vision"
# use_augment=""

python3 -m torch.distributed.launch --use_env --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} /kaggle/working/Chinese-CLIP/cn_clip/training/main2.py \
          --train-data=${train_data} \
          --val-data=${val_data} \
          --resume=${resume} \
          ${reset_data_offset} \
          ${reset_optimizer} \
          --logs=${output_base_dir} \
          --name=${name} \
          --save-step-frequency=${save_step_frequency} \
          --save-epoch-frequency=${save_epoch_frequency} \
          --log-interval=${log_interval} \
          ${report_training_batch_acc} \
          --context-length=${context_length} \
          --warmup=${warmup} \
          --batch-size=${batch_size} \
          --valid-batch-size=${valid_batch_size} \
          --valid-step-interval=${valid_step_interval} \
          --valid-epoch-interval=${valid_epoch_interval} \
          --accum-freq=${accum_freq} \
          --lr=${lr} \
          --wd=${wd} \
          --max-epochs=${max_epochs} \
          --vision-model=${vision_model} \
          ${use_augment} \
          ${freeze_vision}\
          --text-model=${text_model} \
          --grad-checkpointing
  
