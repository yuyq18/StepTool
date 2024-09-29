export PYTHONPATH=./
export TRAIN_PATH="data_train"
export TRAIN_SET="step_grained_for_ppo_example"
export CUDA_VISIBLE_DEVICES="0,1,2,3"

export MODEL_TYPE="qwen2"
# load the base model after sft pretrain
export MODEL_PATH="sft-ckpts/qwen2/checkpoint-3639"

python src/steptool/step_ppo.py \
    --model_path ${MODEL_PATH} \
    --model_type ${MODEL_TYPE} \
    --config_path config/${MODEL_TYPE}/StepTool_ppo.json \
    --data_file ${TRAIN_PATH}/${MODEL_TYPE}/${TRAIN_SET}.csv \
    --epochs 5
    
