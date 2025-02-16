export TRAIN_PATH="data_train/rft"
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export WANDB_PROJECT="baselines"
torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6601 \
    src/baseline-rft/rft.py \
    --model_name_or_path ToolBench/ToolLLaMA-2-7b-v2 \
    --data_path ${TRAIN_PATH}/rft_data_example.json \
    --bf16 True \
    --output_dir "output/rft_baseline-3epoch" \
    --report_to "wandb" \
    --run_name "rft_baseline-3epoch" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --seed 2024 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.04 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --lazy_preprocess False \
    --deepspeed config/ds_configs/stage3-cosine.json
