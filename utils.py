import logging

import numpy as np


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

    if isinstance(choices,dict):
            choices=[v for v in choices.values()]
            if any(isinstance(elem,np.ndarray) for elem in choices):
                choices=[item for sublist in choices for item in sublist]

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
    res=0
    for id,trg in votes.items():

        if isinstance(trg,list):
            res+=1 if id in trg else 0
        else:
            res+= 1 if id==trg else 0

    return res


def pprint(votes, roles, logger, level=logging.DEBUG):
    """
    Print in a meaningful way the agent choices
    :param votes: dict[int->list[int]], maps voter to targets
    :param roles: list[str], list of roles, ordered
    :param logger: logger
    :param level: str, level for logger, default DEBUG
    :return: None
    """


    separator="| {:<8} |"*len(votes)

    to_print="\n|{:<15} || {:<15} |"+separator
    to_format=["Role",'Voter/Voted' ]+[f"Vote_{id}" for id in votes.keys()]
    to_print=to_print.format(*to_format)+"\n"
    to_print+="-"*len(to_print)+"\n"

    for idx in votes.keys():
        targets=[f"Ag_{id}" for id in votes[idx]]
        role=roles[idx]
        name=f"Ag_{idx}"
        fr="|{:<15} || {:<15} |"+separator
        to_print+=fr.format(role,name,*targets)+"\n"

    logger.log(level,to_print)


