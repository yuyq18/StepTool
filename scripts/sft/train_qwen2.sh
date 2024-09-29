export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_PATH="data_train"
export TRAIN_SET="gpt4_dfs_G123_for_sft"
export MODEL_TYPE="qwen2"
export OUTPUT_DIR="sft_ckpts"
export WANDB_PROJECT="SFT-Qwen2"
export WANDB_RUN_NAME="sft_with_gpt4_paths"

torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6601 \
    src/sft/qwen2.py \
   --model_name_or_path ${MODEL_PATH} \
    --data_path ${TRAIN_PATH}/${MODEL_TYPE}$/${TRAIN_SET}.csv \
    --bf16 True \
    --output_dir ${OUTPUT_DIR}/${MODEL_TYPE} \
    --report_to "wandb" \
    --run_name ${WANDB_RUN_NAME} \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --eval_strategy "steps" \
    --eval_steps 400 \
    --save_strategy "steps" \
    --save_steps 400 \
    --save_total_limit 10 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --deepspeed ds_configs/stage3-cosine.json
