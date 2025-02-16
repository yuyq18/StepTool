export TRAIN_PATH="data_train/eto"
export CUDA_VISIBLE_DEVICES=0,1
export WANDB_PROJECT="baselines"

python src/baseline-eto/dpo_train.py \
    --model_name_or_path ToolBench/ToolLLaMA-2-7b-v2 \
    --data_path ${TRAIN_PATH}/dpo_data_example.csv \
    --bf16 True \
    --output_dir "output/eto_baseline-3epoch" \
    --report_to "wandb" \
    --run_name "eto_baseline-3epoch" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --save_total_limit 10 \
    --seed 2024 \
    --learning_rate 1e-4 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --max_prompt_length 7000 \
    --beta 0.1