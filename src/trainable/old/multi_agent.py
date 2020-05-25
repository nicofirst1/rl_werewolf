import itertools
import operator as op
import random
from functools import reduce

import numpy as np
from tqdm import tqdm

from envs import CONFIGS
from policies.utils import random_non_wolf, revenge_target
from wrappers import EvaluationWrapper


def doublefactorial(n):
    if n <= 0:
        return 1
    else:
        return n * doublefactorial(n - 2)


def comb(n, r):
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def theo_win_wolf(env):
    m = env.num_wolves
    n = env.num_players

    teoretical_p_win = 1

    for i in range(m + 1):
        num = comb(m, i) * doublefactorial(n - i)
        num *= (-1) ** i
        den = doublefactorial(n) * doublefactorial((n % 2) - i)
        nd = num / den
        teoretical_p_win -= nd

    print(f"The theoretical ww win is : {teoretical_p_win}")


def experiment(eps, env, is_unite, is_random):
    # get agent ids
    obs = env.reset()
    obs = {k: v['dict_obs'] for k, v in obs.items()}

    ep = 0
    action_space = env.action_space
    to_kill_list = []
    metrics = {k: [] for k in env.custom_metrics.keys()}

    signal_conf = (CONFIGS['signal_length'], CONFIGS['signal_range'])

    pbar=tqdm(total=eps)

    while ep < eps:

        actions = {}

        # get all other ids
        all_trg_ids = [env.shuffle_map[id_] for id_ in env.get_ids("all", alive=True)]

        # perform all ww actions
        if is_random:
            ww_actions = random_non_wolf(action_space, list(obs.values()), signal_conf, unite=is_unite)
        else:
            ww_actions, to_kill_list = revenge_target(action_space, list(obs.values()), to_kill_list, signal_conf, unite=False)

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

        del pbar

    return metrics


def tune(combs, config, eps):
    metrics = {}

    def name(elem):
        return f"np.{elem[0]}_unite.{elem[1]}_rand.{elem[2]}"

    for elem in tqdm(combs):
        np, unite, rand = elem
        config['num_players'] = np
        env = EvaluationWrapper(config)
        metrics[name(elem)] = experiment(eps, env, unite, rand)

    return metrics


if __name__ == '__main__':
    config = CONFIGS
    config['max_days'] = 1000
    eps = 500
    #np, unite, rand
    combs = [[5, 10,], [ False], [False]]
    combs = list(itertools.product(*combs))
    metrics=tune(combs, config, eps)
