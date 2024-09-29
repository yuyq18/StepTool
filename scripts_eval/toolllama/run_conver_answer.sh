cd stabletoolbench/toolbench/tooleval
export RAW_ANSWER_PATH=../../data_eval/answer
export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted
export MODEL_NAME=virtual_toolllama_dfs # change it accordingly
export STRATEGY="DFS_woFilter_w2"  # or CoT@1
export test_set=G1_instruction # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction

mkdir -p ${CONVERTED_ANSWER_PATH}/${MODEL_NAME}
answer_dir=${RAW_ANSWER_PATH}/${MODEL_NAME}/${test_set}
output_file=${CONVERTED_ANSWER_PATH}/${MODEL_NAME}/${test_set}.json

python convert_to_answer_format.py\
    --answer_dir ${answer_dir} \
    --method ${STRATEGY} \
    --output ${output_file}