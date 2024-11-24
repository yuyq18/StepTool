from typing import Dict, Any

from stabletoolbench.toolbench.tooleval.evaluators.registered_cls.utils import register_evaluator
from stabletoolbench.toolbench.tooleval.evaluators.registered_cls.rtl import ReinforceToolLearningEvaluator

from enum import Enum

class AnswerStatus(Enum):
    Unsure = "Unsure"
    Unsolved = "Unsolved"
    Solved = "Solved"
    
@register_evaluator
class ProcessRewardEvaluator(ReinforceToolLearningEvaluator):
    def evaluate_process_reward(self,
                        task_description:Dict,
                        mid_steps,
                        answer:Dict[Any,Any]):
        ret = self.function_call(
            'evaluate_process_reward',
            {
                'query': task_description['query'],
                'mid_steps': mid_steps,
                'final_answer':answer['final_answer'],
            }
        )
        answer_status = AnswerStatus(ret['final_answer_status'])
        return ret['succeed_tool_calling'], ret['contribution_to_final_answer'], answer_status
