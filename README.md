# StepTool

## Environment Setup

1. Create a new Conda environment:

    ```bash
    conda create -n steptool python=3.10
    ```

2. Activate the environment:

    ```bash
    conda activate steptool
    ```

3. Install [Pytorch](https://pytorch.org/get-started/locally/) and other required dependencies via `pip`:

```bash
    pip install torch torchvision torchaudio
    pip install -r requirements.txt
```
**Note**: Ensure that the version of GCC/G++ is >= 9.0.0.

## Data Download

1. Download the compressed dataset from 

2. Uncompress the downloaded `data_train.zip` and put it into the .data_train/ directory

```bash
unzip data_train.zip
```


## SFT Training for Base Models

To train base models using supervised fine-tuning (SFT), run the provided scripts:

```bash
bash scripts/sft/train_qwen2.sh
bash scripts/sft/train_llama3-1.sh
```

Example command from `scripts/sft/train_llama3-1.sh`

```bash
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
export TRAIN_PATH="data_train"
export TRAIN_SET="gpt4_dfs_G123_for_sft"

export MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
export MODEL_TYPE="llama3-1"
export OUTPUT_DIR="sft_ckpts"
export WANDB_PROJECT="SFT-Llama3-1"
export WANDB_RUN_NAME="sft_with_gpt4_paths"

torchrun \
    --nproc_per_node 4 \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr localhost \
    --master_port 6601 \
    src/sft/llama3-1.py \
    --model_name_or_path ${MODEL_PATH} \
    --data_path ${TRAIN_PATH}/${MODEL_TYPE}/${TRAIN_SET}.json \
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
    --deepspeed config/ds_configs/stage3-cosine.json
```



## Step-grained Training with PPO

1. Configuration

Modify the configuration file `config/${MODEL_TYPE}/StepTool_ppo.json` as needed. The `MODEL_TYPE` can be one of `toollama`, `qwen2`, or `llama3-1`. Here’s an example configuration:

```json
{
    "peft_kwargs": {
        "r": 8,
        "lora_alpha": 16,
        "bias": "none",
        "task_type": "CAUSAL_LM"
    },
    "ppo_kwargs": {
        "learning_rate": 1e-5,
        "log_with": "wandb",
        "remove_unused_columns": false,
        "batch_size": 8,
        "mini_batch_size": 2,
        "gradient_accumulation_steps": 4,
        "kl_penalty": "kl",
        "init_kl_coef": 0.3,
        "target_kl": 6,
        "target": 6,
        "horizon": 10000,
        "gamma": 0.99
    }
}
```

2. Run the scripts

```bash
bash scripts/steptool_train/train_toolllama.sh
bash scripts/steptool_train/train_qwen2.sh
bash scripts/steptool_train/train_llama3-1.sh
```

Example command from `scripts/steptool_train/train_toolllama.sh`:

```bash
export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MODEL_TYPE="toolllama"
# load the base model after sft pretrain
export MODEL_PATH="ToolBench/ToolLLaMA-2-7b-v2"

python src/steptool/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/${MODEL_TYPE}/StepTool_ppo.json \
    --data_file ${TRAIN_PATH}/${MODEL_TYPE}/${TRAIN_SET}.csv \
    --max_context_len 4096 \
    --max_response_len 1024 \
    --epochs 5
```

## Evaluation on StableToolBench

TODO

## Experimental Results in Paper

You can download all the experimental results from this link

| **BaseModel** | **Strategy** | **Method** | **I1 Ins.** | **I1 Cat.** | **I1 Tool** | **I2 Cat.** | **I2 Ins.** | **I3 Ins.** | **Average** |
|---------------|--------------|------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|
| gpt-3.5       | COT          | /          | _53.8±1.2_    | _48.0±0.7_    | _51.4±1.2_    | _55.5±1.2_    | _43.4±1.3_    | _53.8±0.4_    | _51.0±1.0_    |
| gpt-3.5       | DFSDT        | /          | _60.0±0.5_    | _53.5±1.3_    | _65.7±0.5_    | _61.6±1.2_    | _50.5±0.7_    | _65.6±2.7_    | _59.5±1.2_    |
| ToolLlama     | COT          | /          | 54.2±0.5    | 50.3±0.8    | 56.5±1.5    | 52.0±0.6    | 45.4±0.6    | 37.2±1.0    | 49.3±0.8    |
| ToolLlama     | COT          | PPO        | 55.0±1.9    | 50.5±0.9    | 42.3±0.7    | 46.4±0.7    | 42.1±1.6    | 35.2±1.2    | 45.3±1.2    |
| ToolLlama     | COT          | StepTool   | **58.7±1.8**    | **57.8±1.7**    | **57.2±0.7**    | **52.7±0.8**    | **52.7±1.0**    | **42.1±1.5**    | **53.5±1.3**    |
| ToolLlama     | DFSDT        | /          | 57.0±1.0    | 52.3±1.5    | 57.5±1.2    | 52.4±0.7    | 49.7±1.7    | 53.8±1.9    | 53.8±1.3    |
| ToolLlama     | DFSDT        | PPO        | 57.5±1.5    | 54.2±0.5    | 53.5±2.0    | 50.8±1.2    | 48.1±0.8    | 43.2±0.4    | 51.2±1.1    |
| ToolLlama     | DFSDT        | StepTool   | **59.7±0.5**    | **55.9±0.0**    | **58.4±1.2**    | **52.8±1.2**    | **51.3±0.2**    | **66.7±0.4**    | **57.5±0.6**    |
| Llama3.1      | COT          | SFT        | 53.9±1.2    | 52.6±1.4    | 51.9±0.9    | 52.2±1.7    | 44.7±0.4    | 36.3±0.8    | 48.6±1.1    |
| Llama3.1      | COT          | PPO        | 50.2±0.9    | **57.8±0.8**    | 53.0±0.6    | 52.3±1.6    | 49.2±1.5    | 38.0±1.5    | 50.1±1.2    |
| Llama3.1      | COT          | StepTool   | **54.3±1.0**   | 56.4±0.3    | **53.2±0.9**    | **53.9±1.7**    | **49.7±0.8**    | **42.6±2.4**    | **51.7±1.2**    |
| Llama3.1      | DFSDT        | SFT        | 58.8±1.2    | 58.0±1.6    | 59.8±0.9    | 53.9±1.9    | 53.5±0.9    | 45.9±1.3    | 55.0±1.3    |
| Llama3.1      | DFSDT        | PPO        | 58.9±0.7    | **61.4±0.7**    | 59.9±1.0    | 55.9±1.0    | 49.5±0.0    | 44.8±0.4    | 55.1±0.9    |
| Llama3.1      | DFSDT        | StepTool   | **59.3±0.8**    | 60.9±1.3    | **60.2±1.3**    | **56.2±1.6**    | **59.3±1.4**    | **50.5±1.0**    | **57.7±1.2**    |
| Qwen2         | COT          | SFT        | 53.0±0.6    | 54.5±0.7    | 59.9±1.2    | 54.0±0.3    | **45.6±1.4**    | 40.7±0.8    | 51.3±0.8    |
| Qwen2         | COT          | PPO        | 58.8±0.9    | 54.9±0.7    | 57.0±0.5    | 54.3±1.0    | 45.1±1.0    | 48.4±3.1    | 53.1±1.2    |
| Qwen2         | COT          | StepTool   | **59.6±1.1**    | **56.1±0.8**    | **61.8±0.8**    | **54.8±0.6**    | 44.5±2.6    | **48.6±1.9**    | **54.2±1.3**    |
| Qwen2         | DFSDT        | SFT        | 63.7±1.3    | 59.3±1.3    | 64.8±1.0    | 56.7±1.1    | 49.1±2.1    | 57.7±1.0    | 58.6±1.3    |
| Qwen2         | DFSDT        | PPO        | 64.1±0.3    | 58.9±2.4    | 66.9±2.2    | 59.8±0.8    | 49.8±1.2    | 54.4±1.7    | 59.0±1.4    |
| Qwen2         | DFSDT        | StepTool   | **65.6±1.8**    | **60.8±0.3**    | **68.4±1.6**    | **60.9±0.9**    | **51.1±1.8**    | **65.3±1.7**    | **62.0±1.4**    |
