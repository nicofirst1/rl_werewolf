import numpy as np

from utils import Params
from utils.serialization import dump_pkl, load_pkl


class Prof:
    """
    This class is responsible to understand what the agents are learning, doing so by looking at the targets output
    in more episodes
    """

    def __init__(self, episode_file=None):

        # dictionary mapping episode instance to an episode class

        if episode_file is not None:
            self.episodes = load_pkl(episode_file)
        else:
            self.episodes = dict()
        self.log_step = Params.log_step

    def add_episode(self, episode_count, episode):
        """
        Adds an episode to the map
        :param episode_count: int, the episode num
        :param episode: Episode class, episode class to be added
        :return: None
        """
        # compute complete info for episodes
        self.episodes[episode_count] = episode
        dump_pkl(self.episodes, Params.episode_file)

    def compare_first_targets(self, episode_range: list):

        f, l = episode_range
        rg = list(self.episodes)[f:l]
        f = rg[0]
        l = rg[-1]

        eps = [v for k, v in self.episodes.items() if f <= k <= l]
        trg = [ep.targets[0] for ep in eps]

        return np.stack(trg)
