from archer_environment import batch_interact_environment
from archer_data import DummyDataset, ReplayBuffer
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from archer_trainer import ArcherTrainer
import wandb
import threading
import os
import torch
import time
def offpolicy_train_loop(env,\
                eval_env,\
                agent,\
                tokenizer,\
                accelerator,\
                warmup_iter: int = 20,
                rollout_size: int = 50,\
                eval_size: int = 1,
                batch_size: int = 2,
                capacity: int = 500000,
                iterations: int = 10,\
                epochs:int = 3, \
                grad_accum_steps: int = 1,\
                env_idx:int = None,\
                do_sample: bool = False,\
                temperature: float = 2.0,\
                critic_lr: float= 1e-3,\
                lm_lr: float = 1e-5,\
                gamma: float = 0.9,
                tau: float = 0.1,
                use_wandb: bool = False,
                env_load_path: str = '',
                actor_epochs: int = 3,
                max_grad_norm: float = 0.01,
                save_path: str = None,
                save_freq: int = 25,
                eval_freq: int = 25,
                agent_type: str = "archer",
                decode_f: callable = lambda x: x,
                **kwargs):
    if agent_type.lower() == "archer_toolllama":
        trainer = ArcherTrainer(agent=agent,\
                            accelerator=accelerator,\
                                tokenizer=tokenizer,\
                                critic_lr = critic_lr,\
                                lm_lr = lm_lr,\
                                gamma = gamma,\
                                tau = tau,\
                                epochs = epochs,\
                                actor_epochs = actor_epochs,
                                grad_accum_steps=grad_accum_steps,
                                max_grad_norm=max_grad_norm)
    replay_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)

    os.makedirs(save_path, exist_ok=True)
    all_trajectories = torch.load(os.path.join(env_load_path, 'trajectories.pt'))
    info = {"rollout.mean": np.mean([d[0]["trajectory_reward"] for d in all_trajectories]),\
            "rollout.max": np.max([d[0]["trajectory_reward"] for d in all_trajectories]),\
            "rollout.min": np.min([d[0]["trajectory_reward"] for d in all_trajectories])}

    replay_buffer = torch.load(os.path.join(env_load_path, 'replay_buffer.pt'))
    agent.prepare()
    #main training loop
    print(">>>start iterations")
    for i in tqdm(range(iterations)):  # pre collected in replay_buffer.pt
        info = {}
        all_trajectories = torch.load(os.path.join(env_load_path, 'trajectories.pt'))
        replay_buffer = torch.load(os.path.join(env_load_path, 'replay_buffer.pt'))
        print("Training")
        if 'filtered' in agent_type.lower():
            filtered_buffer= ReplayBuffer(batch_size= batch_size, capacity=capacity)
            episode_rewards = [d[0]["trajectory_reward"] for d in all_trajectories]
            cutoff = np.quantile(episode_rewards, 1 - 0.1)
            print("Episode Reward Cutoff: ", cutoff)
            filtered_trajectories = list(filter(lambda x: x[0]["trajectory_reward"] >= cutoff, all_trajectories))
            data = sum(filtered_trajectories, [])
            for d in data:
                filtered_buffer.insert(**d)
            info.update(trainer.update(filtered_buffer, no_update_actor = (i < warmup_iter)))
        else:
            # data = list(filter(lambda x: x["reward"] >0, data))
            info.update(trainer.update(replay_buffer, no_update_actor = (i < warmup_iter)))
        if use_wandb and accelerator.is_main_process:
            wandb.log(info)
        if (i+1) % save_freq == 0 and save_path is not None and accelerator.is_main_process:
            print("Saving")
            trainer.save(os.path.join(save_path, 'trainer.pt'), save_dir=save_path)
            torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
    # return model