import random

import numpy as np
from tqdm import tqdm

from envs import CONFIGS
from policies.utils import random_non_wolf, revenge_target
from wrappers import EvaluationWrapper

if __name__ == '__main__':

    # initialize environment
    env = EvaluationWrapper(CONFIGS)
    # get agent ids
    obs = env.reset()
    obs = {k: v['dict_obs'] for k, v in obs.items()}

    # variables
    ww_unite = True  # false if ww can target each other during day
    metrics = {k: [] for k in env.custom_metrics.keys()}
    eps = 10000
    pbar = tqdm(total=eps)  # to get counter
    action_space=env.action_space
    to_kill_list=[]
    signal_conf=(CONFIGS['signal_length'],CONFIGS['signal_range'])

    ep = 0
    while ep < eps:

        actions = {}

        # filter out dead players
        valid_ids = np.where(np.array(env.status_map) == 1)[0]

        # get all other ids
        all_trg_ids = [env.shuffle_map[id_] for id_ in env.get_ids("all", alive=True)]

        # perform all ww actions
        #ww_actions,to_kill_list = revenge_target(action_space, list(obs.values()),signal_conf,unite=False)
        ww_actions = random_non_wolf(action_space, list(obs.values()),signal_conf,unite=False)

        # for every agent
        for idx, id_ in enumerate(obs.keys()):
            if "werewolf" in id_:
                # assign action to agent
                actions[id_] = ww_actions[idx]
            else:
                # perform vill actions
                actions[id_] = random.choice(all_trg_ids)

        # step
        obs, rewards, dones, info = env.step(actions)
        obs = {k: v['dict_obs'] for k, v in obs.items() if "werewolf" in k}

        if dones["__all__"]:
            for k, v in env.custom_metrics.items():
                metrics[k].append(v)

            obs = env.reset()
            obs = {k: v['dict_obs'] for k, v in obs.items()}

            ep += 1
            pbar.update(1)

    print()
    for k, v in metrics.items():
        print(f"Mean value of {k} is : {np.mean(v)} +- {np.std(v)}")
