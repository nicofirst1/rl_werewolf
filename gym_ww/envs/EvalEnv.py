from envs.PaEnv import ParametricActionWrapper
from evaluation.Proff import Proff, Episode


class EvaluationEnv(ParametricActionWrapper):
    """
    Wrapper around ParametricActionWrapper for implementing implementation
    """

    def step(self, action_dict):
        """
        Wrapper around original step function, add target output to episode class
        """

        self.episode.add_target(action_dict)

        return super().step(action_dict)

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """
        # add episode to proff
        self.proff.add_episode(self.episode_count, self.episode)

        # initialize episode and increase counter
        self.episode = Episode()
        self.episode_count += 1
        return super().reset()

    def __init__(self, configs, roles=None, flex=0):

        self.proff = Proff()
        self.episode = Episode()
        self.episode_count = 0

        super().__init__(configs, roles=roles, flex=flex)
