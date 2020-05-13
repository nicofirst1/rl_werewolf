import logging
import random

import numpy as np

from utils import Params


def str_id_map(str_list):
    """
    Maps a list of string to ids
    :param str_list: list, of string
    :return:
    """
    str2id = {}
    id2str = {}

    for i in range(len(str_list)):
        k = str_list[i]
        str2id[k] = i
        id2str[i] = k

    return str2id, id2str


def most_frequent(choices):
    """
    Return most frequent elem in a object, used for votes
    :param choices: list, list of ints
    :param choices: dict, map agent_id:choice
    :return: int, most common
    """

    if isinstance(choices, dict):
        choices = [v for v in choices.values()]
        if any(isinstance(elem, np.ndarray) for elem in choices):
            choices = [item for sublist in choices for item in sublist]

    random.shuffle(choices)
    counter = 0
    num = choices[0]
    for i in choices:
        curr_frequency = choices.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return int(num)


def suicide_num(votes):
    """
    Return number of agents that vote for themself
    :param votes: dict, maps agent id to target
    :return: int
    """
    res = 0
    for id, trg in votes.items():

        if isinstance(trg, list):
            res += 1 if id in trg else 0
        else:
            res += 1 if id == trg else 0

    return res


def pprint(votes, signals, roles, logger, signal_length, level=logging.DEBUG, filtered_ids=None):
    """
    Print in a meaningful way the agent choices
    :param filtered_ids: list[str], optional, list of ids to consider
    :param votes: dict[int->list[int]], maps voter to targets
    :param roles: list[str], list of roles, ordered
    :param logger: logger
    :param level: str, level for logger, default DEBUG
    :return: None
    """

    # filter ids
    if filtered_ids is not None:
        votes = {k: v for k, v in votes.items() if k in filtered_ids}

    separator = "| {:<6} |" * (1 + signal_length)

    to_print = "\n|{:<15} |" + separator
    to_format = ["Role", "Vote"] + [f"Signal_{id}" for id in range(signal_length)]
    to_print = to_print.format(*to_format) + "\n"
    to_print += "-" * len(to_print) + "\n"

    for idx in votes.keys():
        targets = [f"Ag_{votes[idx]}"]
        if len(signals) > 0:
            targets += [f"{sign}" for sign in signals[idx]]
        name = f"{roles[idx]}_{idx}"
        fr = "|{:<15} |" + separator
        to_print += fr.format(name, *targets) + "\n"

    logger.log(level, to_print)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def downsample(vector, rate, maximum, minimum=0):
    """
    Downsample a vector in range [minimum, maximum] in rate parts
    :param vector: list/np.array, the vector to downsample
    :param rate: int, num of splits
    :param maximum: float, max value
    :param minimum: float, min value
    :return: np.array, array of 'rate' distinct elements,
    """

    def test():
        min_ = 5
        max_ = 15
        rate = 3
        prov = np.random.randint(min_, high=max_, size=10)
        res = downsample(prov, rate, max_, minimum=min_)
        a = 1

    # get the split rate
    split = (maximum - minimum) / rate

    # define 'rate' ranges of values shifted by the minimum
    ranges = [[split * i + minimum, split * (i + 1) + minimum] for i in range(rate)]
    # init empty res vector
    res = np.zeros((len(vector)))
    # for every range
    for i in range(len(ranges)):
        # get the current range
        cur_range = ranges[i]
        # find the args where the values of vector are in the current range
        indx = np.argwhere(np.logical_and(vector > cur_range[0], vector < cur_range[1]))
        # set those indices to the cluster index
        res[indx] = i

    return res


def trial_name_creator(something):
    name = str(something).rsplit("_", 1)[0]
    name = f"{name}_{Params.unique_id}"
    return name
