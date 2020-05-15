import random
from collections import Counter

import numpy as np


def random_non_wolf(action_space, obs, unite=False):
    """
    Return a random id for ww filtering out other ww and dead agents
    @param action_space: the action space passed by the policy
    @param unite: bool, false to enable ww to target each other during day
    @param obs: a list of observations
    @return: list[int]
    """

    # if the batch is empty then return random stuff
    if not any(obs):
        return [action_space.sample() for _ in obs]

    # get the roles from the ids
    all_ids, ww_ids, vil_ids, _ = roles_from_info(obs, alive=True)

    # is use phase then ww are allowed to target each other during day
    if unite or obs[0]['phase'] in [0, 1]:

        return [random.choice(vil_ids) for _ in obs]

    else:

        # return random choice
        return [random.choice(all_ids) for _ in obs]


def roles_from_info(obs, alive=True):
    """
    Split ids by role
    Parameters
    ----------
    obs : list[dict], info batch passed by the policy
    alive : bool, Optional, if true take only alive agent ids

    Returns
    -------
    all_ids: list, all the avaiable ids
    ww_ids: list, list of ww ids
    vill_ids: list, list of vil ids
    dead_ids: list, ids of dead players

    """
    # check if infobatch is non empty
    assert any(obs), "Info batch must be non empty"

    # get status map
    status_map = obs[0]['status_map']

    # get all ids from len od status map
    all_ids = set(range(len(status_map)))
    dead_ids = set(np.where(status_map == 0)[0])

    # filter out dead players ids
    if alive:
        all_ids -= dead_ids

    # split into ww and vil
    ww_ids = set([elem['own_id'] for elem in obs])
    vil_ids = all_ids - ww_ids

    # convert back to lists
    ww_ids = list(ww_ids)
    vil_ids = list(vil_ids)
    all_ids = list(all_ids)

    return all_ids, ww_ids, vil_ids, dead_ids


def revenge_target(action_space, obs, to_kill_list, unite=False):

    def chose_target(to_kill_lst):
        """
        Choose the most common out of the kill list
        Parameters
        ----------
        to_kill_lst : list[int], kill list

        Returns
        -------
            most common: int
        """
        return Counter(to_kill_lst).most_common(1)[0][1]

    if not any(obs):
        return random_non_wolf(action_space, obs, unite=unite), []

    # get infos
    phase = obs[0]['phase']
    targets = obs[0]['targets']

    # get the roles from the ids
    all_ids, ww_ids, vil_ids, dead_ids = roles_from_info(obs, alive=True)

    # remove dead players from the to kill list
    for dead in dead_ids:
        to_kill_list = list(filter(lambda a: a != dead, to_kill_list))

    # update kill list with vil that voted for ww
    if phase in [3, 4]:
        for ww in ww_ids:
            to_kill_list += np.where(targets == ww)[0].tolist()

        # remove ww ids
        to_kill_list = list(np.delete(to_kill_list, ww_ids))

    # if the list is empty return random
    if len(to_kill_list) == 0:
        return random_non_wolf(action_space, obs, unite=unite), to_kill_list

    # if unite then return non ww ids every time
    if unite:
        return [chose_target(to_kill_list) for _ in obs], to_kill_list

    # else return most common on kill list when eating
    elif phase == 1:
        return [chose_target(to_kill_list) for _ in obs], to_kill_list

    # else random ( we don't care about communication)
    else:
        return random_non_wolf(action_space, obs, unite=unite), to_kill_list
