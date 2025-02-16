# Ref: https://github.com/YifeiZhou02/ArCHer

# @misc{zhou2024archer,
#       title={ArCHer: Training Language Model Agents via Hierarchical Multi-Turn RL}, 
#       author={Yifei Zhou and Andrea Zanette and Jiayi Pan and Sergey Levine and Aviral Kumar},
#       year={2024},
#       eprint={2402.19446},
#       archivePrefix={arXiv},
#       primaryClass={cs.LG}
# }

from tqdm import tqdm
import numpy as np

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


def batch_interact_environment(agent, tokenizer, env, num_trajectories,\
        post_f = lambda x: x, use_tqdm = True, decode_f = lambda x: x,
        env_idx = None):
    """
    in a bacthed way, interact with the environments  to get a list of trajectories
    [[{"observation":, "next_observation":, "reward":, "done":},...],...]
    post_f: function to add additional attributes to the trajectory
    """
    bsize = env.bsize
    all_trajectories = []
    for num_t in tqdm(range(num_trajectories//bsize), disable = not use_tqdm):
        done = False
        trajectories = [[] for _ in range(bsize)]
        # obs = reset_to(env, 69)
        batch_obs = env.reset(idx=env_idx)
        batch_done = [False,]*bsize
        steps = 0
        while not all(batch_done):
            steps += 1
            # print(f"Environment stpes {str(steps)}")
            action = agent.get_action(batch_obs)
            batch_return = env.step(decode_f(action))
            for i,result in zip(range(bsize), batch_return):
                if result is None:
                    continue
                next_obs, r, done = result
                trajectories[i].append({"observation": batch_obs[i], \
                                "next_observation": next_obs, \
                                "reward": r, \
                                "done": done, \
                                "action": action[i]})
                batch_obs[i] = next_obs
                batch_done[i] = done
            # obs = next_obs
        print(trajectories[0][-1]["next_observation"])
        all_trajectories += [post_f(add_mc_return(add_trajectory_reward(trajectory)))\
                              for trajectory in trajectories]
        # breakpoint()
        # trajectories.append(post_f(add_trajectory_reward(trajectory)))
    return all_trajectories
