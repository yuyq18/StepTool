cd  stabletoolbench/toolbench/tooleval
export API_POOL_FILE=../../openai_key.json
export CONVERTED_ANSWER_PATH=../../../data_eval/model_predictions_converted
export SAVE_PATH=../../../data_eval/pass_rate_results
mkdir -p ${SAVE_PATH}
export CANDIDATE_MODEL="baseline-archer_cot" # change it accordingly
export EVAL_MODEL=gpt-4-turbo-2024-04-09
mkdir -p ${SAVE_PATH}/${CANDIDATE_MODEL}

export test_set=G1_instruction # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction

python eval_pass_rate.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --save_path ${SAVE_PATH}/${CANDIDATE_MODEL} \
    --reference_model ${CANDIDATE_MODEL} \
    --test_ids ../../solvable_queries/test_query_ids \
    --max_eval_threads 1 \
    --evaluate_times 3 \
    --test_set ${test_set} \
    # --overwrite