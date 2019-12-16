
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
    Return most frequent elem in a list, used for votes
    :param choices: list, list of ints
    :param choices: dict, map agent_id:choice
    :return: int, most common
    """

    if isinstance(choices,dict):
        choices=[v for v in choices.values()]

    counter = 0
    num = choices[0]
    for i in choices:
        curr_frequency = choices.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num
