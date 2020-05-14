import copy
import json
import logging

import numpy as np

from envs.PaEnv import CONFIGS
from evaluation import Prof, Episode
from gym_ww import logger, ww, vil
from other.custom_utils import pprint, suicide_num, most_frequent
from utils import Params
from wrappers.PaWrapper import ParametricActionWrapper


class EvaluationWrapper(ParametricActionWrapper):
    """
    Wrapper around ParametricActionWrapper for implementing implementation
    """

    #########################
    # ENV METHODS
    #########################

    def step(self, action_dict):
        """
        Wrapper around original step function, add target output to episode class
        """

        # split signal from target
        signals, original_target = self.split_target_signal(action_dict)

        # stack all targets into a matrix
        targets = np.stack(list(original_target.values()))
        eval_obs = dict(
            day=self.day_count,
            status_map=copy.copy(self.status_map),
            phase=self.phase,

        )

        self.episode.add_observation(eval_obs)
        self.episode.add_target(targets)
        self.episode.add_signals(signals)

        # save current config before changing
        prev=dict(
            phase=copy.copy(self.phase),
            day_count = copy.copy(self.day_count)
        )

        # execute step in super to change info, do not move this line
        obs, rewards, dones, info = super().step(action_dict)

        # remove names from ids
        signals = {int(k.split("_")[1]): v for k, v in signals.items()}

        targets = {int(k.split("_")[1]): v for k, v in original_target.items()}

        #fixme: comment + fix
        self.log_diffs(prev, targets, signals)
        self.update_metrics(targets)


        return obs, rewards, dones, info

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """
        # self.log("Reset called")

        self.episode.days = self.day_count

        # if is time to log then do it
        if self.episode_count % self.prof.log_step == 0:
            # add episode to prof and reset counter
            self.prof.add_episode(self.episode_count, self.episode)

        # initialize episode and increase counter
        self.episode = Episode(self.num_players)
        self.episode_count += 1

        # reset info dict
        self.initialize_info()

        # step used for logging matches
        if self.ep_step == Params.log_step:
            self.ep_step = 0
        else:
            self.ep_step += 1

        return super().reset()

    def __init__(self, configs, roles=None, flex=0):
        super().__init__(configs, roles=roles, flex=flex)

        self.log(
            f"Starting game with {self.num_players} players: {self.num_players - self.num_wolves}"
            f" {vil} and {self.num_wolves} {ww}")

        self.log(f"Config follows:\n{json.dumps(CONFIGS)}")

        # todo: find a way to split when there are multiple workers
        self.prof = Prof()
        self.episode = Episode(self.num_players)
        self.episode_count = 1

        self.win_brackets = win_brackets()

    #########################
    # UTILS
    #########################

    def normalize_metrics(self):
        """
        In here normalization for custom metrics should be executed.
        Notice that this method needs to be called before the reset.
        :return: None
        """

        self.custom_metrics["suicide"] /= (self.day_count + 1)

        self.custom_metrics["accord"] /= (self.day_count + 1) * 2

    def initialize_info(self):

        self.custom_metrics = dict(
            suicide=0,  # number of times a player vote for itself
            win_wolf=0,  # number of times wolves win
            win_vil=0,  # number of times villagers win
            tot_days=0,  # total number of days before a match is over
            accord=0,  # number of agents that voted for someone which was not killed
        )

    #########################
    # LOGGING
    #########################

    def update_metrics(self, targets):
        """
        Update metrics at each step
        @param targets: dict[int->int], maps agents to target
        @return:
        """

        # update suicide num
        if self.phase == 3:
            # add number of suicides
            self.custom_metrics["suicide"] += suicide_num(targets)/len(targets)
            # update number of differ
            chosen = most_frequent(targets)
            accord = sum([1 for t in targets.values() if t == chosen]) / len(targets)
            self.custom_metrics["accord"] += accord

            if accord > 1: raise AttributeError("Accord garter than 1")

        elif self.phase == 1:
            were_wolves = self.get_ids(ww, alive=True, include_just_died=True)
            were_wolves = {k: v for k, v in targets.items() if k in were_wolves}
            chosen = most_frequent(were_wolves)
            accord = sum([1 for t in were_wolves.values() if t == chosen]) / len(were_wolves)
            self.custom_metrics["accord"] += accord

            if accord > 1: raise AttributeError("Accord garter than 1")

        if self.is_done:

            # update day count
            self.custom_metrics["tot_days"] = self.day_count
            self.normalize_metrics()

            # if episode is over print winner
            alive_ww = self.get_ids(ww, alive=True)

            if len(alive_ww) > 0:
                self.custom_metrics['win_wolf'] += 1
            else:
                self.custom_metrics['win_vil'] += 1

    def log_diffs(self, prev, targets, signals):
        """
            Logs difference between status from step to step
            @param prev: EvaluationEnv, state before step
            @param targets: dict[int->int], maps agents to target
            @param signals: dict[int->np.array], maps agents to array of signals
            @return: None
        """

        # is it's not the log step yet, return
        if Params.log_step != self.ep_step:
            return

        # if there is no difference between phases then return
        if prev['phase'] == self.phase:
            return

        # log day
        self.log(f"Day {prev['day_count']})")

        # log phase
        if self.phase == 0:
            self.log(f"Phase {self.phase} | Night Time | Voting")

        elif self.phase == 1:
            self.log(f"Phase {self.phase} | Night Time| Eating")

        elif self.phase == 2:
            self.log(f"Phase {self.phase} | Day Time| Voting")

        else:
            self.log(f"Phase {self.phase} | Day Time| Executing")

        # print actions
        #fixme
        filtered_ids = self.get_ids(ww, alive=True) if self.phase in [0, 1] else self.get_ids('all', alive=True)

        if self.phase in [1,3]:
            filtered_ids.append(self.just_died)

        pprint(targets, signals, self.roles, signal_length=self.signal_length, logger=logger,
               filtered_ids=filtered_ids)

        # notify of dead agents
        if self.phase in [1, 3]:
            # get dead ids
            dead = self.just_died

            # build msg
            msg = f"Player {dead} ({self.role_map[dead]}) has been "

            # personalize for role
            if self.phase == 1:
                msg += "eaten"
            else:
                msg += "executed"

            self.log(msg=msg)

        # else report most voted
        else:
            choice = most_frequent([v for k,v in targets.items() if k in filtered_ids])
            self.log(msg=f"Most voted is {choice} ({self.role_map[choice]})")

        if self.is_done:

            # if episode is over print winner
            alive_ww = self.get_ids(ww, alive=True)

            msg = copy.copy(self.win_brackets)

            if len(alive_ww) > 0:
                msg = msg.replace("-", "Wolves won")

            else:
                msg = msg.replace("-", "Villagers won")

            self.log(msg)

        self.log("\n")

    def log(self, msg, level=logging.INFO):

        logger.log(msg=msg, level=level)


def win_brackets(num_lines=5):
    """
    Return a bracket made of hashtags for the winner of a match
    """

    msg = "\n"
    hash_num = 1
    for _ in range(num_lines):
        hash_num *= 2
        msg += "#" * (hash_num) + "\n"

    msg += "-\n"
    for j in range(num_lines):
        msg += "#" * (hash_num) + "\n"
        hash_num //= 2

    return msg
