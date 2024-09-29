from evaluators import load_registered_automatic_evaluator
import os
import json
import csv
from evaluators.registered_cls.rtl import AnswerStatus, TaskStatus, AnswerPass
import random
from concurrent.futures import ThreadPoolExecutor,as_completed
import argparse
from tqdm import tqdm
import numpy as np
from utils import test_sets, get_steps
import backoff

abs_dir = os.path.split(__file__)[0]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--converted_answer_path', type=str, default="", required=True, help='converted answer path')
    parser.add_argument('--save_path', type=str, default="", required=False, help='result save path')
    parser.add_argument('--reference_model', type=str, default="", required=False, help='model predictions path')
    parser.add_argument('--reference_path', type=str, default=None, required=False, help='reference path')
    parser.add_argument('--test_ids', type=str, default="", required=True, help='model predictions path')
    parser.add_argument('--task_num', type=int, default=None, required=False, help='task num')
    parser.add_argument('--evaluator', type=str, default="tooleval_gpt-3.5-turbo_default", required=False, help='which evaluator to use.')
    parser.add_argument('--max_eval_threads', type=int, default=30, required=False, help='max threads nums')
    parser.add_argument('--evaluate_times', type=int, default=4, required=False, help='how many times to predict with the evaluator for each solution path.')
    parser.add_argument('--test_set', nargs='+', default=['G1_instruction'], help='test set name')
    parser.add_argument('--overwrite', action='store_true', help='whether to overwrite the existing result file')
    return parser.parse_args()

# def write_results(filename: str, reference_model: str, label_cnt: dict) -> None:
#     with open(filename, 'w', newline='') as file:
#         writer = csv.writer(file, delimiter="\t")
#         writer.writerow(["query", "available_tools", "model_intermediate_steps", "mid_steps_reward", "model_final_step", "model", "query_id"])
#         for query_id in label_cnt:
#             query = label_cnt[query_id]["query"]
#             tool_names = label_cnt[query_id]["tool_names"]
#             answer_steps = label_cnt[query_id]["answer_steps"]
#             mid_steps_reward = label_cnt[query_id]["mid_steps_reward"]
#             final_step = label_cnt[query_id]["final_step"]
#             # is_solved = label_cnt[query_id]["is_solved"]
#             writer.writerow([query, tool_names, answer_steps, mid_steps_reward, final_step, reference_model, query_id,])
            

if __name__ == "__main__":
    args = parse_args()
    evaluators = [load_registered_automatic_evaluator(evaluator_name=args.evaluator, evaluators_cfg_path=os.path.join(abs_dir,'evaluators')) for _ in range(args.max_eval_threads)]
    
    @backoff.on_exception(backoff.expo, Exception, max_time=15)
    def compute_process_reward(query_id, example, evaluate_time):
        global evaluators
        evaluator = random.choice(evaluators)
        # print(f"Query id: {query_id}. Evaluate time: {evaluate_time}")
        answer_steps, answer_steps_list, final_step = get_steps(example)
        
        # print("answer_steps_list", answer_steps_list[:3])
        # assert False

        # if "'name': 'Finish'" not in final_step:
        #     return query_id, {}, AnswerStatus.Unsolved, evaluate_time
        
        # for mid_step in answer_steps_list[:-1]:
        succeed_tool_calling_list, contributions, answer_status = evaluator.evaluate_process_reward(
            {
                'query':example['query'],
                'available_tools':example['available_tools'],
            },
            answer_steps_list[:-1],
            example['answer'],
        )
        process_reward = {
            "succeed_tool_calling": succeed_tool_calling_list,
            "contributions": contributions,
            # "answer_status": answer_status
        }
        # mid_steps_reward["succeed_tool_calling"].append(succeed_tool_calling)
        # mid_steps_reward["utility_score"].append(utility_score)
        
        # is_solved = "" 
        # is_solved, is_solved_reason = evaluator.check_is_solved(
        #     {
        #         'query':example['query'],
        #         'available_tools':example['available_tools'],
        #     },
        #     example['answer'],
        #     return_reason=True
        # )

        # return query_id, task_solvable, is_solved, label, reason, not_hallucinate, evaluate_time
        # print("process_reward: ", process_reward)
        return query_id, process_reward, answer_status, evaluate_time
        
    reference_model = args.reference_model
    output_list = []
    
    for test_set in args.test_set:
        
        save_file = f"{args.save_path}/{test_set}.json"
        if args.task_num:
            save_file = f"{args.save_path}/{test_set}_{args.task_num}.json"
        
        # reference_path = f"{args.converted_answer_path}/{reference_model}/{test_set}.json"
        # test_ids = list(json.load(open(os.path.join(args.test_ids, test_set+".json"), "r")).keys())
        reference_path = f"{args.converted_answer_path}/{test_set}.json"
        reference_examples = json.load(open(reference_path, "r"))
        if args.task_num:
            reference_examples = {k:reference_examples[k] for k in list(reference_examples.keys())[:args.task_num]}
        if os.path.exists(save_file) and not args.overwrite:
            old_existed_ids = list(json.load(open(save_file, "r")).keys())
            old_label_cnt = json.load(open(save_file, "r"))
            existed_ids = []
            label_cnt = {}
            for query_id in old_existed_ids:
                ans = old_label_cnt[query_id]
                if len(ans['process_reward'].keys()) == args.evaluate_times:
                    existed_ids.append(query_id)
                    label_cnt[query_id] = ans
        else:
            existed_ids = []
            label_cnt = {}
        
        # print("numbers of existed_ids: ", len(existed_ids))
        # print(len(label_cnt))
        # assert False
        with ThreadPoolExecutor(args.max_eval_threads) as pool:
            future = []
            
            for query_id in reference_examples:
                # if str(query_id) not in test_ids:
                #     continue
                if query_id in existed_ids:
                    continue
                for i in range(args.evaluate_times):
                    example = reference_examples[query_id]
                    future.append(pool.submit(
                        compute_process_reward,
                        query_id,
                        example,
                        evaluate_time=i
                    ))

            for thd in tqdm(as_completed(future),total=len(future),ncols=100):
                # if len(thd.result()) <= 3:
                #     print("thd Error: ", thd.result())
                #     assert False
                query_id, process_reward, is_solved, evaluate_time = thd.result()
                # process_reward = [1, 1, 1]
                example = reference_examples[query_id]
                query = example["query"]
                tool_names = []
                # breakpoint()
                for tool_dict in example["available_tools"]:
                    tool_name = tool_dict["function"]["name"]
                    tool_names.append(tool_name)
                answer_steps, answer_steps_list, final_step = get_steps(example)
                if query_id not in label_cnt:
                    label_cnt[query_id] = {}
                label_cnt[query_id]["query"] = query
                label_cnt[query_id]["tool_names"] = tool_names
                label_cnt[query_id]["answer_steps"] = answer_steps_list[:-1]
                # label_cnt[query_id]["mid_steps_reward"] = mid_steps_reward  # parsed
                if 'process_reward' not in label_cnt[query_id]:
                    label_cnt[query_id]["process_reward"] = {}
                label_cnt[query_id]["process_reward"][evaluate_time] = process_reward
                label_cnt[query_id]["final_step"] = final_step

                if 'is_solved' not in label_cnt[query_id]:
                    label_cnt[query_id]["is_solved"] = {}
                label_cnt[query_id]["is_solved"][evaluate_time] = str(is_solved)
                # print("========== Finish and Dump into json file===========", query_id, is_solved, evaluate_time)
                
                json.dump(label_cnt, open(save_file, "w"), ensure_ascii=False, indent=4)
                
        json.dump(label_cnt, open(save_file, "w"), ensure_ascii=False, indent=4)
        # json.dump(label_cnt, open(f"{args.save_path}/{test_set}.json", "w"), ensure_ascii=False, indent=4)
        
        # filename = f"{args.save_path}/{test_set}_{reference_model}.csv"
        # filename = f"{args.save_path}/{test_set}.csv"
        # write_results(filename, reference_model, label_cnt)
        # scores = []
        # for runtime in range(args.evaluate_times):
        #     score = 0
        #     # avg_mid_reward = 0 yyq TODO
        #     for query_id in label_cnt:
        #         solved_dict = {**label_cnt[query_id]['is_solved']}
        #         solved_dict = {int(k):v for k,v in solved_dict.items()}
        #         if runtime not in solved_dict:
        #             score += 0
        #             continue
        #         if solved_dict[runtime] == "AnswerStatus.Solved":
        #             score += 1
        #         elif solved_dict[runtime] == "AnswerStatus.Unsure":
        #             score += 0.5
        #     scores.append(score / len(label_cnt))

        # print(f"Test set: {test_set}. Model: {reference_model}.")
        # solve_rate = sum(scores) / len(scores) * 100
        # std_dev = np.std(scores).item() * 100
        # print(f"Solve rate: {solve_rate:.1f}% Std: {std_dev:.1f}%")

        
