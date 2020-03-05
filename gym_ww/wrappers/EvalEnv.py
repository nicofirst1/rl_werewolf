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
        # tansform targets to square numpy matrix
        unshuffler = np.vectorize(
            lambda x: self.wrapped.unshuffle_map[x] if x in self.wrapped.unshuffle_map.keys() else x)
        targets = {k: unshuffler(v) for k, v in action_dict.items()}
        targets = sorted(targets.items(), key=lambda v: int(v[0].split("_")[1]))
        targets = [elem[1] for elem in targets]

        # if some agents are dead insert -1 row
        if len(targets) != self.wrapped.num_players:
            dead = [i for i, x in enumerate(self.wrapped.status_map) if x == 0]
            for idx in dead:
                to_insert = np.asarray([-1] * self.wrapped.num_players)
                targets.insert(idx, to_insert)

        # stack all targets into a matrix
        targets = np.stack(targets)
        self.episode.add_target(targets.copy(), sum(self.wrapped.status_map), is_night=self.wrapped.is_night,
                                is_comm=self.wrapped.is_comm)

        return super().step(action_dict)

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """
        # if is time to log then do it
        if self.episode_count % self.prof.log_step == 0:
            # add episode to prof and reset counter
            self.prof.add_episode(self.episode_count, self.episode)

        # initialize episode and increase counter
        self.episode = Episode(self.wrapped.num_players)
        self.episode_count += 1
        return super().reset()

    def __init__(self, configs, roles=None, flex=0):

        super().__init__(configs, roles=roles, flex=flex)

        # todo: find a way to split when there are multiple workes
        self.prof = Prof()
        self.episode = Episode(self.wrapped.num_players)
        self.episode_count = 1
