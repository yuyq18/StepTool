# Ref: https://github.com/YifeiZhou02/ArCHer

# @misc{zhou2024archer,
#       title={ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL}, 
#       author={Yifei Zhou and Andrea Zanette and Jiayi Pan and Sergey Levine and Aviral Kumar},
#       year={2024},
#       eprint={2402.19446},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }

import torch
import transformers
from tqdm import tqdm
from archer_agent import ArcherAgent
from offpolicy_train_loop import offpolicy_train_loop

import torch.nn as nn
import numpy as np 
import wandb
from omegaconf import DictConfig, OmegaConf
import os
import hydra
from accelerate import Accelerator
from datetime import timedelta
from accelerate import DistributedDataParallelKwargs, InitProcessGroupKwargs
transformers.logging.set_verbosity_error()

CONFIG_NAME = os.environ.get("ARCHER_CONFIG_NAME", None)
@hydra.main(version_base=None, config_path="../../config/archer/", config_name=CONFIG_NAME)
def main(config: "DictConfig"):
    print(">>> Configuration file: "+CONFIG_NAME+"<<<")
    print(OmegaConf.to_yaml(config))
    try:
        from huggingface_hub import login
        login(token=config.huggingface_token)
    except:
        print(">>> Huggingface token not found.")

    accelerator = Accelerator(InitProcessGroupKwargs(timeout=timedelta(18000)))
    device = accelerator.device

    decode_f = lambda x:x
    # load decision model
    if config.agent_type.lower() == "archer_toolllama":
        print(">>> Using ArCHer agent with ToolLLAMA")
        agent = ArcherAgent(device=device, accelerator=accelerator, 
                            temperature=config.temperature, do_sample=config.do_sample, 
                            policy_lm=config.policy_lm, critic_lm=config.critic_lm,
                            cache_dir=config.cache_dir, max_new_tokens=config.max_new_tokens,
                            use_lora=config.use_lora,
                            eos_str=config.eos_str)
    else:
        raise NotImplementedError("Agent not implemented.")
    tokenizer = agent.tokenizer
    if config.checkpoint_path is not None:
        state_dict = torch.load(config.checkpoint_path, map_location=device)['model_state_dict']
        agent.model.load_state_dict(state_dict)

    if config.use_wandb and accelerator.is_main_process:
        wandb.login(key=config.wandb_key)
        wandb.init(project=config.project_name, name=config.run_name, config=dict(config))

    offpolicy_train_loop(env = None,
                agent = agent,
                tokenizer = tokenizer,
                eval_env = None,
                accelerator = accelerator,
                decode_f=decode_f,
                **config)


if __name__ == "__main__":
    main()
