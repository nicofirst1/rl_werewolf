from itertools import islice


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return dict(list(islice(iterable, n)))


def vocabulary2index(text):
    if not isinstance(text, list):
        text = text.split()

    uniq = set(text)

    index2text = {}
    text2index = {}
    indexed_text = []

    for elem in uniq:
        text2index[elem] = len(text2index)
        index2text[len(index2text)] = elem

    for word in text:
        indexed_text.append(text2index[word])

    return indexed_text, text2index, index2text


def slice_list(to_slice, indices):
    """
    Slice a list according to a list of indices
    :param to_slice: the list to be sliced
    :param indices: the list of indices
    :return: list of sliced elements
    """

    res = []
    for idx in indices:
        res.append(to_slice[idx])

    return res
