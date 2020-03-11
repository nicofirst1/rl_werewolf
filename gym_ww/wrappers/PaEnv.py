from functools import reduce

import gym
import numpy as np
from ray.rllib import MultiAgentEnv
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.error import UnsupportedSpaceException

from envs.TurnEnv import TurnEnvWw
from gym_ww import ww


class ParametricActionWrapper(TurnEnvWw):
    """
    Wrapper around TurnEnvWw for implementing parametric actions
    """

    def reset(self):
        """
        Reset the environment
        :return:
        """
        # wrap observation from original reset function
        obs = super().reset()
        obs = self.wrap_obs(obs)

        return obs

    def step(self, action_dict):
        """
        Wrapper around original step function
        :param action_dict:
        :return:
        """
        obs, rewards, dones, info = super().step(action_dict)
        obs = self.wrap_obs(obs)
        return obs, rewards, dones, info

    def get_action_mask(self):
        """
        Estimates action mask for current observation
        Return a boolean vector in which indexOf(zeros) are invalid actions
        :return: np.array
        """

        def mask_targets():
            """
            Generate mask for targets
            :return:
            """
            # filter out dead agents
            mask = self.status_map.copy()

            # if is night
            if self.is_night:
                # filter out wolves
                ww_ids = self.get_ids(ww, alive=True)
                for idx in ww_ids:
                    mask[idx] = 0

            # apply shuffle to mask
            mask = [mask[self.unshuffle_map[idx]] for idx in range(len(mask))]

            return mask

        def mask_signal():
            """
            Generate mask for signal
            :return: list[bool]: 1 for allowable returns, 0 otehrwise
            """
            mask = [0 for _ in range(self.num_players)]
            range_ = self.signal_range

            mask[:range_] = [1] * range_
            return mask

        mask = mask_targets()
        mask += mask_signal()
        return np.asarray(mask)

    def wrap_obs(self, observations):
        """
        Wrap the original observation adding custom action mask
        :param observations: dict[int]->dict(), observation as outputted by the wrapped environment
        :return: dict, augmented observations
        """

        new_obs = {}
        # define the action mask and convert it to array
        action_mask = self.get_action_mask()

        # for every agent
        for agent_id, obs in observations.items():
            # make array out of observation (flatten)
            obs = _make_array_from_obs(obs, super().observation_space)

            # add action mask
            new_obs[agent_id] = dict(
                action_mask=action_mask,
                original_obs=obs
            )

        return new_obs

    def __init__(self, configs, roles=None, flex=0):
        # use TurnEnvWw as wrapped env
        super().__init__(configs, roles, flex)



    @TurnEnvWw.observation_space.getter
    def observation_space(self):
        return super().action_space

    @TurnEnvWw.observation_space.getter
    def observation_space(self):

        super_obs=super().observation_space
        # transform original space to box
        obs = _make_box_from_obs(super_obs)

        # define wrapped obs space
        observation_space = gym.spaces.Dict({

            "action_mask": gym.spaces.Box(low=0, high=1, shape=(sum(self.action_space.nvec),)),
            "original_obs": obs,
        })

        return observation_space

def _make_box_from_obs(space):
    """
    Convert a spaces.Dict to a spaces.Box given highs/lows vectors initialization.

    :param space: gym.spaces.Dict
    :return: gym.spaces.Box
    """
    sp = list(space.spaces.values())
    lows = []
    highs = []

    # for every space
    for s in sp:

        # if discrete then the observation will be transformed to a OneHotVector rapresentation to deal with
        # discrete values, so add n 0/1 as lows/highs
        if isinstance(s, gym.spaces.Discrete):
            highs += [1] * s.n
            lows += [0] * s.n

        # if multibinary then do the same as before but get shape with reduce
        elif isinstance(s, gym.spaces.MultiBinary):
            sh = reduce(lambda x, y: x * y, s.shape)
            highs += [1] * sh
            lows += [0] * sh

        # if box then just flatten highs and flows
        elif isinstance(s, gym.spaces.Box):
            highs += s.high.flatten().tolist()
            lows += s.low.flatten().tolist()

        # else raise exception
        else:
            raise UnsupportedSpaceException(
                "Space {} is not supported.".format(space))

    # convert to array
    highs = np.asarray(highs)
    lows = np.asarray(lows)
    # return box as high/low initialization
    return gym.spaces.Box(high=highs, low=lows)


def _make_array_from_obs(obs, original_spaces):
    """
    Transform original obs dict to one dimensional np.array
    :param obs: dict, original observation dictionary
    :param original_spaces: gym.spaces.Dict, gym observation space as given by the wrapped env
    :return: np.array, flatten out array of observations
    """
    # get size of space
    size = get_preprocessor(original_spaces)(original_spaces, None).size
    # initialize zeros array with correct shape
    array = np.zeros(size)
    # get space dict
    spaces = original_spaces.spaces
    offset = 0
    # for every observation
    for k in spaces.keys():

        # get gym space related to observation
        sp = spaces[k]
        v = obs[k]

        # if MultiBinary, get shape and add values to array
        if isinstance(sp, gym.spaces.MultiBinary):
            size = reduce(lambda x, y: x * y, sp.shape)
            array[offset:offset + size] = v

        # if Discrete then we need to use the OHV rappresentation, and set n to be one
        elif isinstance(sp, gym.spaces.Discrete):
            size = sp.n
            array[offset + v] = 1

        # if Box, then get size and assign flatten value
        elif isinstance(sp, gym.spaces.Box):
            size = reduce(lambda x, y: x * y, sp.shape)
            array[offset:offset + size] = v.flatten()

        # else raise exception
        else:
            raise UnsupportedSpaceException(f"space {type(sp)} is not supported for ParametricWrapper")

        # update offset
        offset += size

    return np.asarray(array)
