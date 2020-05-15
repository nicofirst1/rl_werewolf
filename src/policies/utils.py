import random

import numpy as np


def random_non_wolf(action_space, info_batch, unite=True):
    """
    Return a random id for ww filtering out other ww and dead agents
    @param action_space: the action space passed by the policy
    @param unite: bool, false to enable ww to target each other during day
    @param info_batch: a list of infos
    @return: list[int]
    """

    # if the batch is empty then return random stuff
    if not any(info_batch):
        return [action_space.sample() for _ in info_batch]

    # get all possible ids
    ids = set(range(0, action_space.n))

    # filter out dead players ids
    dead_ids = set(np.where(info_batch[0]['obs']['status_map'] == 0)[0])
    ids -= dead_ids

    # is use phase then ww are allowed to target each other during day
    if unite or info_batch[0]['obs']['phase'] in [0, 1]:
        # filter out other ww ids
        ww_ids = set([elem['obs']['own_id'] for elem in info_batch])
        ids -= ww_ids


    # convert back to list
    ids = list(ids)

    # return random choice
    return [random.choice(ids) for _ in info_batch]
