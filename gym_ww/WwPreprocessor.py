from functools import reduce

import gym
import numpy as np
from ray.rllib.models import Preprocessor
from ray.rllib.utils.error import UnsupportedSpaceException


class WwPreprocessor(Preprocessor):
    """
    Custom preprocessor for Ww env
    """

    def __init__(self, obs_space, options=None):
        super(WwPreprocessor, self).__init__(obs_space, options=options)

    def _init_shape(self, obs_space, options):
        """
        Return the shape of the flatten observation space
        :param obs_space: gym.spaces.Dict, observation space as output from env
        :param options: dict
        :return: tuple(int,None), return a tuple where the first element is the shape of the flatten dict
        """

        space_shape = 0
        # if obs space is a dict
        if isinstance(obs_space, gym.spaces.Dict):

            # for every space in the dict
            for k, v in obs_space.spaces.items():

                # if discrete then add one
                if isinstance(v, gym.spaces.Discrete):
                    space_shape += 1
                # if multibinary then add number of multi
                elif isinstance(v, gym.spaces.MultiBinary):
                    space_shape += v.n

                # if box add by multiplying shapes
                elif isinstance(v, gym.spaces.Box):
                    space_shape += reduce(lambda x, y: x * y, v.shape)

                else:
                    raise UnsupportedSpaceException(
                        "Space {} is not supported for custom preprocessor.".format(v))

        elif isinstance(obs_space, gym.spaces.Box):
            return obs_space.shape

        else:
            raise UnsupportedSpaceException(
                "Space {} is not supported for custom preprocessor.".format(obs_space))

        return space_shape,

    def transform(self, observation):
        """
        Transform observation, i.e. flatten out
        :param observation: gym.spaces.Dict, observation space as output from env
        :return: np.array(space_shape)
        """

        ret = []
        # for every observation
        for v in observation.values():

            # if is int then append it to list
            if isinstance(v, int):
                ret.append(v)
            # if is np array then flatten, convert to list and add to ret
            elif isinstance(v, np.ndarray):
                ret += v.flatten().tolist()
            else:
                raise NotImplementedError(f"Transformation for type {type(v)} not implemented")

        return np.asarray(ret)
