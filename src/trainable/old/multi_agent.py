import random

import numpy as np
from tqdm import tqdm

from wrappers import EvaluationWrapper

if __name__ == '__main__':

    # initialize environment
    env_configs = {'num_players': 20}
    env = EvaluationWrapper(env_configs)
    # get agent ids
    agent_ids = env.reset().keys()

    # variables
    ww_unite=True # false if ww can target each other during day
    metrics = {k: [] for k in env.custom_metrics.keys()}
    eps = 10000
    pbar = tqdm(total=eps) # to get counter

    ep = 0
    while ep < eps:

        actions={}
        # filter out dead players
        valid_ids = np.where(np.array(env.status_map) == 1)[0]

        # get shuffled werewolf ids
        ww_trg_ids=[env.shuffle_map[id_] for id_ in env.get_ids("werewolf", alive=True)]
        vill_trg_ids=[env.shuffle_map[id_] for id_ in env.get_ids("villager", alive=True)]
        all_trg_ids= ww_trg_ids + vill_trg_ids

        # for every agent
        for id_ in agent_ids:
            if "werewolf" in id_:
                # perform ww actions
                if env.phase in [0, 1] or ww_unite:
                    actions[id_]=random.choice(vill_trg_ids)
                else:
                    actions[id_]=random.choice(all_trg_ids)
            else:
                # perform vill actions
                actions[id_] = random.choice(all_trg_ids)

        #step
        obs, rewards, dones, info = env.step(actions)
        # update agent ids
        agent_ids = obs.keys()

        if dones["__all__"]:
            for k, v in env.custom_metrics.items():
                metrics[k].append(v)
            env.reset()
            ep += 1
            pbar.update(1)

    print()
    for k, v in metrics.items():
        print(f"Mean value of {k} is : {np.mean(v)} +- {np.std(v)}")
