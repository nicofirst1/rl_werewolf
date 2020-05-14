import random

import numpy as np
from tqdm import tqdm

from utils import Params
from wrappers import EvaluationWrapper

if __name__ == '__main__':

    # initialize environment
    env_configs = {'num_players': 10}
    env = EvaluationWrapper(env_configs)
    # get agent ids
    agent_ids = env.reset().keys()

    metrics = {k: [] for k in env.custom_metrics.keys()}
    eps = 1000
    pbar = tqdm(total=eps)
    # perform 10000 episodes
    ep = 0
    while ep < eps:

        # filter out dead players
        valid_ids = np.where(np.array(env.status_map) == 1)[0]

        # if night filter out wolves
        if env.phase in [0, 1]:
            ww_idx = [k for k, v in env.role_map.items() if v == "werewolf"]
            for ww_id in ww_idx:
                valid_ids = np.delete(valid_ids, np.where(valid_ids == ww_id))

        # chose a valid action
        actions = {id: random.choice(valid_ids) for id in agent_ids}
        obs, rewards, dones, info = env.step(actions)
        # update agent ids
        agent_ids = obs.keys()

        if dones["__all__"]:
            for k, v in env.custom_metrics.items():
                metrics[k].append(v)
            env.reset()
            ep += 1
            pbar.update(1)

    for k, v in metrics.items():
        print(f"Mean value of {k} is : {np.mean(v)} +- {np.std(v)}")
