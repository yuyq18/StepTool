# Codes for StepTool

## 0. Environment Setup

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



## 1. Step-grained Data Construction

Step-grained rewards can be assigned using various methods, including automated rule-based systems, human annotations, or advanced models such as GPT-4.

Below is a reference prompt for GPT-4 to perform step-grained reward annotation:
```
Query:
{query}

Intermediate Steps:
{mid_steps}
Final Answer:
{final_answer}

Given the above query, all intermediate steps and the final answer, you need to evaluate the entire task-solving process by following rules:
(1) **Successful Tool Calling:** For each intermediate step, determine if a tool was called successfully and give a score of 0 (no) or 1 (yes).
(2) **Contribution to Final Answer:** For each intermediate step, rate its contribution to the final answer on a scale from 0 to 5.
(3) **Final Answer Status:** Determine if the final answer is 'Solved',  'Unsure', or 'Unsolved'.

Now provide your evaluation in JSON format with the parameters of 'succeed_tool_calling', 'contribution_to_final_answer' and 'final_answer_status'  to the function `evaluate_process_reward`.
```

We provide a sample training data file, data_train/${MODEL_TYPE}/step_grained_for_ppo_example.csv, for use in the subsequent training phase. 

The complete training dataset can be downloaded from [this Dropbox link](https://www.dropbox.com/scl/fo/faizqka89m4fbz0ukhwai/AEzVegNkK2sOfvUQQOj2uUQ?rlkey=425kosbbbeihewx61bujnop94&st=62zi4jea&dl=0).

## 2. Step-grained Training with PPO

The step-grained training is implemented in `src/steptool/step_ppo.py` and `src/steptool/step_ppotrainer.py`.

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

Example Command (from `scripts/steptool_train/train_toolllama.sh`):

```bash
export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo_example"
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


**Note**, for `qwen2` and `llama3.1`, these models must undergo supervised fine-tuning (SFT) beforehand:

```bash
bash scripts/sft/train_qwen2.sh
bash scripts/sft/train_llama3-1.sh
```

A sample training dataset for SFT is available in `data_train/${MODEL_TYPE}/gpt4_dfs_G123_for_sft_example.json`

## Train Baselines (RFT, PPO, ETO, ArCHer)

### RFT

```bash
bash scripts/baseline-rft/train_rft.sh
```

### PPO (Final Reward)

```bash
bash scripts/baseline-ppo/train_ppo.sh
```

### ETO (DPO)

```bash
bash scripts/baseline-eto/train_dpo.sh
```

### ArCHer

```bash
bash scripts/baseline-archer/build_data.sh
bash scripts/baseline-archer/train_archer.sh
```

## Evaluation on StableToolBench

### 1. Build the api server

To set up the API server, follow the [StableToolBench](https://github.com/THUNLP-MT/StableToolBench) instructions.

First, download a cache from [HuggingFace](https://huggingface.co/datasets/stabletoolbench/Cache) or [Tsinghua Cloud](https://cloud.tsinghua.edu.cn/f/07ee752ad20b43ed9b0d/?dl=1). 

After downloading, unzip the folder into the `stabletoolbench/server` folder and ensure the `server` folder contains `tool_response_cache` folder and `tools` folder. The resulting folder of `server` looks like:
```
├── /server/
│  ├── /tools/
│  │  └── ...
│  ├── /tool_response_cache/
│  │  └── ...
│  ├── config.yml
│  ├── main.py
│  ├── utils.py
```

Next, specify your configurations in `server/config.yml`

```
api_key: 
api_base: 
model: gpt-4-turbo-preview
temperature: 0
toolbench_url: http://8.130.32.149:8080/rapidapi
rapidapi_key: 
tools_folder: "./tools"
cache_folder: "./tool_response_cache"
is_save: true
port: 8081
```

To run the server:
```
cd server
python main.py
```
The server will be run at `http://localhost:{port}/virtual`. 
To use the server, you will further need a toolbench key. You can apply one from this [form](https://forms.gle/oCHHc8DQzhGfiT9r6).

### 2. Run the model using vLLM

We recommend setting up a new Conda environment for vLLM by following the [installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)

To build a vLLM server for the `ToolLLaMA-2-7b-v2` model, you can use the following command:
```
python -m vllm.entrypoints.openai.api_server --model ToolBench/ToolLLaMA-2-7b-v2 --served-model-name toolllama --max-model-len=8192 --dtype=bfloat16 --host 127.0.0.1 --port 8083 --rope-scaling '{"type": "linear", "factor": 2.0}'
```

**Note**: If you're using a LoRA version of the model, make sure to merge the LoRA weights with the base model before running it in vLLM.

### 3. Run the Evaluation Scripts

To evaluate the model on `StableToolBench`, first configure `stabletoolbench/config.yml`: 

```
api_key:
api_base:
toolbench_key:
tool_root_dir: server/tools
```

Then, infer the model on the `solvable_test_queries` by running:

```bash
bash scripts_eval/toolllama/inference_toolllama_vllm.sh
bash scripts_eval/qwen2/inference_qwen2_vllm.sh
bash scripts_eval/llama3-1/inference_llama3-1_vllm.sh
bash scripts_eval/baseline-rft/inference_rft_vllm.sh
bash scripts_eval/baseline-ppo/inference_ppo_vllm.sh
bash scripts_eval/baseline-eto/inference_eto_vllm.sh
bash scripts_eval/baseline-archer/inference_archer_vllm.sh
```

Finally, evaluate the `pass_rate` and `win_rate` metrics:

```bash
bash scripts_eval/toolllama/run_convert_answer.sh
bash scripts_eval/toolllama/run_pass_rate.sh
bash scripts_eval/toolllama/run_preference.sh

bash scripts_eval/qwen2/run_convert_answer.sh
bash scripts_eval/qwen2/run_pass_rate.sh
bash scripts_eval/qwen2/run_preference.sh

bash scripts_eval/llama3-1/run_convert_answer.sh
bash scripts_eval/llama3-1/run_pass_rate.sh
bash scripts_eval/llama3-1/run_preference.sh

bash scripts_eval/baseline-rft/run_convert_answer.sh
bash scripts_eval/baseline-rft/run_pass_rate.sh

bash scripts_eval/baseline-ppo/run_convert_answer.sh
bash scripts_eval/baseline-ppo/run_pass_rate.sh

bash scripts_eval/baseline-eto/run_convert_answer.sh
bash scripts_eval/baseline-eto/run_pass_rate.sh

bash scripts_eval/baseline-archer/run_convert_answer.sh
bash scripts_eval/baseline-archer/run_pass_rate.sh
```

## Main Experimental Results in the Paper

All results were re-evaluated in February 2025 to ensure the stability of the \texttt{gpt-4-turbo-2024-04-09} evaluator.

| **BaseModel** | **Strategy** | **Method** | **I1 Ins.** | **I1 Cat.** | **I1 Tool** | **I2 Cat.** | **I2 Ins.** | **I3 Ins.** | **Average** |
|-----|-----|-----|---------|---------|---------|---------|---------|---------|---------|
| gpt-3.5-turbo-0125 | CoT | / | _55.5±0.9_ | _49.5±0.2_ | _53.6±0.8_ | _55.9±1.8_ | _45.8±0.7_ | _56.0±2.7_ | _52.7±1.2_ |
| gpt-3.5-turbo-0125 | DFSDT | / | _60.7±0.8_ | _55.9±0.7_ | _68.6±0.7_ | _65.5±0.5_ | _56.3±1.7_ | _64.5±2.8_ | _61.9±1.2_ |
|-----|-----|-----|---------|---------|---------|---------|---------|---------|---------|
| ToolLLaMA-2-7b-v2 | CoT | SFT | 56.4±1.8 | 49.0±1.2 | 46.3±1.8 | 48.4±0.0 | 45.6±1.7 | 40.4±0.8 | 47.7±1.2 |
| ToolLLaMA-2-7b-v2 | CoT | RFT | 53.2±2.1 | 49.7±0.7 | 47.5±0.8 | 46.0±1.7 | 45.8±2.0 | 38.5±1.2 | 46.8±1.4 |
| ToolLLaMA-2-7b-v2 | CoT | PPO (Final Reward)      | 55.8±2.1 | 50.4±0.6 | 46.4±0.4 | 48.4±2.6 | 44.5±1.2 | 40.2±0.0 | 47.6±1.2 |
| ToolLLaMA-2-7b-v2 | CoT | ETO (DPO) | 51.4±0.1 | 50.6±0.8 | 48.9±1.7 | 43.8±0.4 | 48.4±0.2 | 38.8±1.0 | 47.0±0.7 |
| ToolLLaMA-2-7b-v2 | CoT | ArCHer | 57.3±1.0 | 49.5±1.0 | 48.3±0.4 | 43.8±0.4 | 46.5±0.6 | 35.5±2.8 | 47.6±1.1 |
| ToolLLaMA-2-7b-v2 | CoT | **StepTool**     | **62.8±0.1** | **58.8±0.9** | **61.6±1.2** | **56.9±2.2** | **56.3±2.7** | **42.6±0.7** | **56.5±1.3** |
|-----|-----|-----|---------|---------|---------|---------|---------|---------|---------|
| ToolLLaMA-2-7b-v2 | DFSDT | SFT | 62.8±1.8 | 55.2±0.5 | 57.8±0.7 | 56.0±0.3 | 52.0±1.2 | 54.1±1.3 | 56.3±1.0 |
| ToolLLaMA-2-7b-v2 | DFSDT | RFT | 58.3±2.9 | 51.3±0.5 | 55.1±1.3 | 48.7±0.8 | 50.5±0.8 | 58.5±2.4 | 53.7±1.5 |
| ToolLLaMA-2-7b-v2 | DFSDT | PPO (Final Reward) | 62.8±1.4 | 58.4±0.3 | 57.4±1.7 | 53.6±2.0 | 54.4±1.1 | 39.9±0.8 | 54.4±1.2 |
| ToolLLaMA-2-7b-v2 | DFSDT | ETO (DPO) | 60.1±0.8 | 54.8±1.5 | 56.1±1.4 | 54.4±3.0 | 54.6±1.0 | 44.0±2.8 | 54.0±1.8 |
| ToolLLaMA-2-7b-v2 | DFSDT | ArCHer | 63.5±1.0 | **59.5±1.5** | 59.2±1.6 | 55.4±0.5 | 54.7±0.7 | 53.3±2.0 | 57.6±1.2 |
| ToolLLaMA-2-7b-v2 | DFSDT | **StepTool** | **64.1±0.8** | 58.7±1.0 | **62.9±0.9** | **57.8±1.2** | **56.0±0.8** | **66.1±2.7** | **60.9±1.3** |