import numpy as np


class Episode:
    """
    Class to hold info about a training episode, that is from the start of a game till one wins
    the target dict has the following form:
    {
        night:
        {
            exec:
            {
                original -> ndarray[num player, num player, days]
                stacked :
                {
                    0 -> ndarray[days, num player],
                    1 -> ndarray[days, num player],
                    2 -> ndarray[days, num player],
                    ......
                    num player -> ndarray[days, num player],
                }
            }
            comm: same as exec

        },
        day: same as night

    }
    """

    def __init__(self, num_player):
        self.days = 0
        self.observations=[]
        self.targets=[]
        self.signals=[]

        self.alive = []
        self.num_players = num_player

    def add_observation(self, obs):
        self.observations.append(obs)

    def add_target(self, targets):
        self.targets.append(targets)

    def add_signals(self, signal):
        self.signals.append(signal)

