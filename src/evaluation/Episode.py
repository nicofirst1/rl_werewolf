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
        self.targets = dict(
            night={},
            day={},
        )

        for v in self.targets.values():
            v['exec'] = dict(
                original=[],
                stacked=[]
            )
            v['comm'] = dict(
                original=[],
                stacked=[]
            )

        self.alive = []
        self.num_players = num_player

    def add_target(self, target_mat, alive, is_night=False, is_comm=False):
        """
        Add a target to the target list and increment day
        :param is_comm: bool flag for communication phase
        :param is_night: bool, if it is night time
        :param alive: int, number of agents alive
        :param target_mat: nxn matrix, the target output
        :return: None
        """

        if is_night and is_comm:
            self.targets['night']['comm']['original'].append(target_mat)
        elif is_night and not is_comm:
            self.targets['night']['exec']['original'].append(target_mat)
        elif not is_night and is_comm:
            self.targets['day']['comm']['original'].append(target_mat)
        else:
            self.targets['day']['exec']['original'].append(target_mat)
            self.alive.append(alive)
            self.days += 1

    def complete_episode_info(self):
        """
        Add infos when episode is done and convert target list to tensor [num player, num player, days]
        :return:
        """

        self.targets['day']['comm']['original'] = np.stack(self.targets['day']['comm']['original'], axis=2)
        self.targets['day']['exec']['original'] = np.stack(self.targets['day']['exec']['original'], axis=2)
        self.targets['night']['exec']['original'] = np.stack(self.targets['night']['exec']['original'], axis=2)
        self.targets['night']['comm']['original'] = np.stack(self.targets['night']['comm']['original'], axis=2)

    def stack_agent_targets(self):
        """
        Stack agent target throughout the episode into a same matrix of size [episode len, num player]
        :return: None
        """

        def stack_single(targets):
            stacked_targets = {idx: np.zeros((self.days, self.num_players)) for idx in range(self.num_players)}

            for idx in range(self.days):
                for jdx in range(self.num_players):
                    stacked_targets[jdx][idx] = targets[jdx, :, idx]

            return stacked_targets

        for v1 in self.targets.values():
            for v2 in v1.values():
                v2['stacked'] = stack_single(v2['original'])

    def agent_diff(self, agent_id, stm):
        """
        estimate te difference between adjacent rows.
        Given row1 and row2, the difference is estimated as follows:
        for every elem1i in row1, get the position of elem1i in row2 (elem2j), compute the difference |elem1i-elem2j|
        :param agent_id:
        :return:
        """

        agent_votes = stm[agent_id]
        diff = 0
        for idx in range(self.days - 1):
            alive = self.alive[idx + 1]
            cur_vote = agent_votes[idx][:alive]
            next_vote = agent_votes[idx + 1][:alive]

            inner_diff = 0
            for jdx in range(len(cur_vote)):
                vote = cur_vote[jdx]
                next_idx = np.where(next_vote == vote)
                try:
                    to_add = abs(jdx - next_idx[0][0]) / (alive - 1)

                    if to_add > 1: raise Exception("more than one")

                    inner_diff += to_add
                except IndexError:
                    pass

            inner_diff /= len(cur_vote)

            if inner_diff > 1: raise Exception("more than one")

            diff += inner_diff

        diff /= self.days
        if diff > 1: raise Exception("more than one")

        return diff
