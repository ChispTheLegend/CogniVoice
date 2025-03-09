#!/bin/bash

echo "run_cls.sh started"

# All of the Huggingface TrainingArguments and arguments in ./framework/training_args.py apply here
METHOD="${1:-whisper-tiny}"

LR=1e-5
WEIGHT_DECAY=1e-2
EVAL_STEP=10
BATCH_SIZE=1
EVAL_STRATEGY='steps'

export TF_CPP_MIN_LOG_LEVEL=3
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/lib/cuda

#export NEPTUNE_API_TOKEN="your_api_token"

python train.py \
        --do_train --do_eval --dataloader_num_workers 2 --save_total_limit 1 --per_device_eval_batch_size 4 --load_best_model_at_end \
        --overwrite_output_dir \
        --evaluation_strategy ${EVAL_STRATEGY} --save_strategy ${EVAL_STRATEGY} \
        --logging_steps ${EVAL_STEP} --eval_steps ${EVAL_STEP} --save_steps ${EVAL_STEP} \
        --per_device_train_batch_size ${BATCH_SIZE} \
        --num_train_epochs 1 \
        --learning_rate ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --method ${METHOD} \
        --output_dir outputs \
        --use_disvoice \
        --use_text \
        --poe_alpha 0.1 \
        --use_poe \
        --task cls \
        # --use_llama2 \
        trust_remote_code=True
        2>&1 | tee train_log.txt
