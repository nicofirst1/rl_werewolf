import numpy as np

from evaluation import Prof, Episode
from wrappers.PaEnv import ParametricActionWrapper


class EvaluationEnv(ParametricActionWrapper):
    """
    Wrapper around ParametricActionWrapper for implementing implementation
    """

    def step(self, action_dict):
        """
        Wrapper around original step function, add target output to episode class
        """

        # split signal from target
        signals, targets=self.split_target_signal(action_dict)



        # stack all targets into a matrix
        targets = np.stack(list(targets.values()))
        obs = dict(
            day=self.day_count,
            status_map=self.status_map,
            phase=self.phase,

        )

        self.episode.add_observation(obs)
        self.episode.add_target(targets)
        self.episode.add_signals(signals)

        return super().step(action_dict)

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """

        self.episode.days=self.day_count

        # if is time to log then do it
        if self.episode_count % self.prof.log_step == 0:
            # add episode to prof and reset counter
            self.prof.add_episode(self.episode_count, self.episode)

        # initialize episode and increase counter
        self.episode = Episode(self.num_players)
        self.episode_count += 1
        return super().reset()

    def __init__(self, configs, roles=None, flex=0):

        super().__init__(configs, roles=roles, flex=flex)

        # todo: find a way to split when there are multiple workes
        self.prof = Prof()
        self.episode = Episode(self.num_players)
        self.episode_count = 1
