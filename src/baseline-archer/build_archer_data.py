from archer_data import ReplayBuffer
import pandas as pd
import numpy as np
import os
import torch
import json

model = "toolllama"
buffer_batch_size = 2
tool_data_file = os.environ.get("DATA_FILE", None)

# bsize = 4
df = pd.read_csv(tool_data_file, sep="\t")


# build origin trajectory
trajectories = [[] for _ in range(len(df))]

MAX_LEN = 1024

# TODO
for i in range(0, len(df)):
    prompt_list = eval(df.iloc[i]["prompt"])
    response_list = eval(df.iloc[i]["response"])
    reward_list = eval(df.iloc[i]["reward"])

    obs = prompt_list[0]
    next_obs = obs + response_list[0] + prompt_list[1]
    done = False
    if len(obs) > MAX_LEN:
        obs = obs[-MAX_LEN:]
    if len(next_obs) > MAX_LEN:
        
        next_obs = next_obs[-MAX_LEN:]
    trajectories[i].append({"observation": obs, \
                            "next_observation": next_obs, \
                            "reward": reward_list[0], \
                            "done": done, \
                            "action": response_list[0]})
    for j in range(1, len(response_list)):
        obs = next_obs
        next_obs = obs + response_list[j]
        if j+1 < len(response_list):
            next_obs += prompt_list[j+1]
        else:
            done = True
        
        if len(obs) > MAX_LEN:
            obs = obs[-MAX_LEN:]
        if len(next_obs) > MAX_LEN:
            next_obs = next_obs[-MAX_LEN:]
        trajectories[i].append({"observation": obs, \
                                "next_observation": next_obs, \
                                "reward": reward_list[j], \
                                "done": done, \
                                "action": response_list[j]})
        

def add_trajectory_reward(trajectory):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_reward = np.sum([d["reward"] for d in trajectory])
    for d in trajectory:
        d.update({"trajectory_reward": trajectory_reward})
    return trajectory

def add_mc_return(trajectory, gamma = 0.95):
    """
    add trajectory reward to the dict of each interaction
    """
    trajectory_rewards = np.array([d["reward"] for d in trajectory]).reshape(1, -1)
    gamma_row = np.cumprod(np.ones((1, trajectory_rewards.shape[1]))*gamma)
    gamma_matrix = np.triu(gamma_row.reshape(1, -1 )/ gamma_row.reshape(-1, 1))
    mc_returns = np.sum(trajectory_rewards*gamma_matrix, axis = 1)
    for d, mc in zip(trajectory, mc_returns):
        d.update({"mc_return": mc})

    return trajectory

all_trajectories = [add_mc_return(add_trajectory_reward(trajectory))\
                              for trajectory in trajectories]

# save to json
trajectory_json = {}
for i in range(len(all_trajectories)):
    trajectory_json[i] = all_trajectories[i]

with open("trajectories.json", "w") as f:
    json.dump(trajectory_json, f, indent=4, ensure_ascii=False)


# build replay_buffer
replay_buffer= ReplayBuffer(batch_size=buffer_batch_size)

data = sum(all_trajectories, [])
for t in data:
    replay_buffer.insert(**t)

print(">>> Saving Replay Buffer")
save_path = os.environ.get("SAVE_PATH", "save")
os.makedirs(save_path, exist_ok=True)
torch.save(replay_buffer, os.path.join(save_path, 'replay_buffer.pt'))
torch.save(all_trajectories, os.path.join(save_path, 'trajectories.pt'))
