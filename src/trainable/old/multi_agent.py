import csv
import itertools
import operator as op
import random
import sys
from functools import reduce
from multiprocessing.pool import ThreadPool

import numpy as np
from tqdm import tqdm

from envs import CONFIGS
from policies.utils import random_non_wolf, revenge_target
from wrappers import EvaluationWrapper


def double_factorial(n):
    """
    Return the double factorial of a number
    """

    if n <= 0:
        return 1
    else:
        return n * double_factorial(n - 2)


def comb(n, r):
    """
    Return the combination of n elements in r sets
    """
    r = min(r, n - r)
    numer = reduce(op.mul, range(n, n - r, -1), 1)
    denom = reduce(op.mul, range(1, r + 1), 1)
    return numer / denom


def theo_win_wolf(env):
    """
    Return the theoretical win prob for wolves given the env
    """
    m = env.num_wolves
    n = env.num_players

    teoretical_p_win = 1

    for i in range(m + 1):
        num = comb(m, i) * double_factorial(n - i)
        num *= (-1) ** i
        den = double_factorial(n) * double_factorial((n % 2) - i)
        nd = num / den
        teoretical_p_win -= nd

    print(f"The theoretical ww win is : {teoretical_p_win}")
    return theo_win_wolf()


def experiment(args):
    """
    Run an experiment
    """
    # extract the args from the argument
    eps, env, is_unite, is_random = args
    # reset the env and get the observations
    obs = env.reset()
    obs = {k: v['dict_obs'] for k, v in obs.items()}

    # initializations
    ep = 0
    action_space = env.action_space
    to_kill_list = []
    metrics = []

    signal_conf = (CONFIGS['signal_length'], CONFIGS['signal_range'])

    # while loop on number of envs
    while ep < eps:

        actions = {}

        # get all other ids
        all_trg_ids = [env.shuffle_map[id_] for id_ in env.get_ids("all", alive=True)]

        # perform all ww actions
        if is_random:
            ww_actions = random_non_wolf(action_space, list(obs.values()), signal_conf, unite=is_unite)
        else:
            ww_actions, to_kill_list = revenge_target(action_space, list(obs.values()), to_kill_list, signal_conf,
                                                      unite=False)

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

        # if match is over
        if dones["__all__"]:
            # save the ww wins and reset
            metrics.append(env.custom_metrics['win_wolf'])
            obs = env.reset()
            obs = {k: v['dict_obs'] for k, v in obs.items()}

            ep += 1

    return [env.num_players, is_random, is_unite], metrics


def save_results(metrics, f_name):
    """
    Save the result of a tuning experiment in csv format
    """
    # the headers
    headers = ['num players', 'unite', 'random', 'mean', 'std']

    rows = [headers]
    # add a csv row
    for k, v in metrics.items():
        names = [elem.split('.')[1] for elem in k.split('_')]
        r = names + list(v)
        rows.append(r)

    with open(f_name, "w") as file:
        wr = csv.writer(file, dialect='excel')
        wr.writerows(rows)


def tune(combs, config, eps):
    """
    Runs multiple instances of the training with different parameters in parallel
    """

    def name(elem):
        return f"np.{elem[0]}_unite.{elem[1]}_rand.{elem[2]}"

    arg_list = []

    # for every combination of parameters
    for idx in tqdm(range(len(combs))):

        # change the number of players
        num_players = combs[idx][0]
        config['num_players'] = num_players

        # split the eps into multiple parts depending on the number of players
        rng = num_players // 10 + 1
        for _ in range(rng):
            # define new eps and add args to list
            ep = eps // rng
            env = EvaluationWrapper(config)
            arg_list.append([ep, env, combs[idx][1], combs[idx][2]])

    # define pool of threads
    res = []
    pool = ThreadPool()

    # start pool, append result and log percentage done
    for i, r in enumerate(pool.imap_unordered(experiment, arg_list), 1):
        sys.stderr.write('\rdone {0:%}'.format(i / len(arg_list)))
        res.append(r)

    metrics = {}
    # merge metrics coming from the same settings
    for k, v in res:
        k = name(k)
        if k not in metrics.keys():
            metrics[k] = []

        metrics[k] += v

    # reformat
    metrics = {name(elem[0]): elem[1] for elem in res}
    metrics = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}

    return metrics


if __name__ == '__main__':
    config = CONFIGS
    config['max_days'] = 1000000  # remove limit for max days
    eps = 500  # number of episodes
    # combination of the following parameters: number of players, use unite policy, use random policy
    combs = [[9, 21, 31, 51, 71, 101, 151, 201, 251, 301], [False], [True]]
    # generate a combination of the previous
    combs = list(itertools.product(*combs))
    # start the tuning
    metrics = tune(combs, config, eps)
    # save results to csv
    save_results(metrics, 'prov.csv')
