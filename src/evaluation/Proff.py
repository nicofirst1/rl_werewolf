from utils import Params
import pickle


class Proff:
    """
    This class is responsible to understand what the agents are learning, doing so by looking at the targets output
    in more episodes
    """

    def __init__(self):
        # dictionary mapping episode instance to an episode class
        self.episodes = dict()
        self.log_step=200

    def add_episode(self, episode_count, episode):
        """
        Adds an episode to the map
        :param episode_count: int, the episode num
        :param episode: Episode class, episode class to be added
        :return: None
        """
        self.episodes[episode_count*self.log_step] = episode
        self.dump_episodes()

    def dump_episodes(self):
        """
        Dump the episode dict to a pickle file
        :return: None
        """
        with open(Params.episode_file, "wb") as f:
            pickle.dump(self.episodes,f)

