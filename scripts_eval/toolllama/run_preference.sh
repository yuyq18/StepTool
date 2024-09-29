cd  toolbench/tooleval
export CONVERTED_ANSWER_PATH=../../data_eval/model_predictions_converted
export SAVE_PATH=../../data_eval/preference_results
export PASS_RATE_PATH=../../data_eval/pass_rate_results

export REFERENCE_MODEL=virtual_gpt3.5-0125_dfs # change it accordingly
export CANDIDATE_MODEL=virtual_toolllama_dfs # change it accordingly

export EVAL_MODEL=gpt-4-turbo-2024-04-09
mkdir -p ${SAVE_PATH}/${REFERENCE_MODEL}_${CANDIDATE_MODEL}

export test_set=G1_instruction # G1_category, G1_tool, G2_category, G2_instruction, G3_instruction

python eval_preference.py \
    --converted_answer_path ${CONVERTED_ANSWER_PATH} \
    --reference_model ${REFERENCE_MODEL} \
    --output_model ${CANDIDATE_MODEL} \
    --test_ids ../../solvable_queries/test_query_ids/ \
    --save_path ${SAVE_PATH}/${REFERENCE_MODEL}_${CANDIDATE_MODEL} \
    --pass_rate_result_path ${PASS_RATE_PATH} \
    --max_eval_threads 30 \
    --evaluate_times 3 \
    --test_set ${test_set} \
    # --overwrite