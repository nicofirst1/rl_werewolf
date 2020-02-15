from utils import Params

from utils.serialization import dump_pkl


class Prof:
    """
    This class is responsible to understand what the agents are learning, doing so by looking at the targets output
    in more episodes
    """

    def __init__(self):
        # dictionary mapping episode instance to an episode class
        self.episodes = dict()
        self.log_step=5

    def add_episode(self, episode_count, episode):
        """
        Adds an episode to the map
        :param episode_count: int, the episode num
        :param episode: Episode class, episode class to be added
        :return: None
        """
        # compute complete info for episodes
        self.episodes[episode_count] = episode
        dump_pkl(self.episodes,Params.episode_file)



