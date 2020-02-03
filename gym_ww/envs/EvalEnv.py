import numpy as np

from envs.PaEnv import ParametricActionWrapper
from evaluation import Proff, Episode
import collections


class EvaluationEnv(ParametricActionWrapper):
    """
    Wrapper around ParametricActionWrapper for implementing implementation
    """

    def step(self, action_dict):
        """
        Wrapper around original step function, add target output to episode class
        """
        #tansform targets to square numpy matrix
        targets=sorted(action_dict.items(), key=lambda v: int(v[0].split("_")[1]))
        targets=[elem[1] for elem in targets]
        targets=np.stack(targets)
        
        self.episode.add_target(targets)

        return super().step(action_dict)

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """
        # if is time to log then do it
        if self.episode_count%self.proff.log_step==0:
            # add episode to proff and reset counter
            self.proff.add_episode(self.episode_count, self.episode)
            self.episode_count=0

        # initialize episode and increase counter
        self.episode = Episode()
        self.episode_count += 1
        return super().reset()

    def __init__(self, configs, roles=None, flex=0):

        self.proff = Proff()
        self.episode = Episode()
        self.episode_count = 0

        super().__init__(configs, roles=roles, flex=flex)
