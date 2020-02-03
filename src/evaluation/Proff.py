class Proff:
    """
    This class is responsible to understand what the agents are learning, doing so by looking at the targets output
    in more episodes
    """

    def __init__(self):
        # dictionary mapping episode instance to an episode class
        self.episodes = dict()

    def add_episode(self, episode_count, episode):
        """
        Adds an episode to the map
        :param episode_count: int, the episode num
        :param episode: Episode class, episode class to be added
        :return: None
        """
        self.episodes[episode_count] = episode


class Episode:
    """
    Class to hold info about a training episode, that is from the start of a game till one wins
    """

    def __init__(self):
        self.days = 0
        self.targets = []

    def add_target(self, target_mat):
        """
        Add a target to the target list and increment day
        :param target_mat: nxn matrix, the target output
        :return: None
        """
        self.targets.append(target_mat)
        self.days += 1
