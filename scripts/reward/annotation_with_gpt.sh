# cd  ../../toolbench/tooleval
# export API_POOL_FILE=path/to/your/openai_key_json_file.json
export PYTHONPATH="./:./stabletoolbench/toolbench/tooleval"
export API_POOL_FILE=src/reward/openai_key.json
export CONVERTED_ANSWER_PATH=data/model_predictions_converted
export SAVE_PATH=data/reward_annotation/
mkdir -p ${SAVE_PATH}

# export CANDIDATE_MODEL="virtual_qwen2_sft_dfs_fix_epoch3"
export CANDIDATE_MODEL="qwen2"
export EVAL_MODEL="gpt-4-turbo-2024-04-09"
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}
# unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
# --evaluators_cfg_path \
python src/reward/annotation_with_gpt.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH}/${CANDIDATE_MODEL} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --evaluator ${EVAL_MODEL} \
    --max_eval_threads 1 \
    --task_num 5 \
    --evaluate_times 3 \
    --test_set G123_example \