import copy
import logging

import numpy as np

from evaluation import Prof, Episode
from gym_ww import logger, ww, vil
from other.custom_utils import pprint
from utils import Params
from wrappers.PaEnv import ParametricActionWrapper


class EvaluationEnv(ParametricActionWrapper):
    """
    Wrapper around ParametricActionWrapper for implementing implementation
    """


    def log_diffs(self, prev,targets,signals):

        # is it's not the log step yet, return
        if Params.log_step != self.ep_step:
            return

        # if there is no difference between phases then return
        if prev.phase== self.phase:
            return

        # log day
        self.log(f"Day {prev.day_count})")

        # log phase
        if self.phase==0:
            self.log(f"Phase {self.phase} | Night Time | Voting")

        elif self.phase==1:
            self.log(f"Phase {self.phase} | Night Time| Eating")

        elif self.phase==2:
            self.log(f"Phase {self.phase} | Day Time| Voting")

        else:
            self.log(f"Phase {self.phase} | Day Time| Executing")

        # print actions
        filter_ids = self.get_ids(ww, alive=True) if self.phase in [0, 1] else self.get_ids('all', alive=True)
        targets={int(k.split("_")[1]):v for k,v in targets.items()}
        signals={int(k.split("_")[1]):v for k,v in signals.items()}
        pprint(targets, signals, self.roles, signal_length=self.signal_length, logger=logger,
               filter_ids=filter_ids)

        # notify of dead agents
        if self.phase in [1,3]:
            # get dead ids
            dead=[p-c for p,c in zip(prev.status_map,self.status_map)]
            dead=np.asarray(dead)
            dead=np.nonzero(dead)[0][0]

            # build msg
            msg=f"Player {dead} ({self.role_map[dead]}) has been "

            # personalize for context
            if self.phase==1:
                msg+="eaten"
            else:
                msg+="executed"

            self.log(msg=msg)

        if self.is_done:
            # if episode is over print winner
            alive_ww=self.get_ids(ww,alive=True)

            msg=copy.copy(self.win_brakets)


            if len(alive_ww)>0:
                msg=msg.replace("-","Wolves won")
                self.custom_metrics['win_wolf'] += 1

            else:
                msg=msg.replace("-","Villagers won")
                self.custom_metrics['win_vil'] += 1

            self.log(msg)

        self.log("\n")

    def log(self, msg, level=logging.INFO):

        logger.log(msg=msg,level=level)

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
            status_map=self.status_map,
            phase=self.phase,

        )

        self.episode.add_observation(eval_obs)
        self.episode.add_target(targets)
        self.episode.add_signals(signals)

        prev=copy.deepcopy(self)
        obs, rewards, dones, info = super().step(action_dict)

        self.log_diffs(prev, original_target,signals)

        return obs, rewards, dones, info

    def reset(self):
        """
        Calls reset function initializing the episode class again
        """
        self.log("Reset called")

        # step used for logging matches
        if self.ep_step == Params.log_step:
            self.ep_step = 0
        else:
            self.ep_step += 1

        self.episode.days = self.day_count

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

        logger.info(f"Starting game with {self.num_players} players: {self.num_players-self.num_wolves} {vil} and {self.num_wolves} {ww}")

        # todo: find a way to split when there are multiple workes
        self.prof = Prof()
        self.episode = Episode(self.num_players)
        self.episode_count = 1

        self.win_brakets=win_brakets()



def win_brakets(num_hash=16, num_lines=4):
    """
    Return a bracket made of hashtags for the winner of a match
    """

    msg="\n"
    new_hash=num_hash
    for _ in range(num_lines):
        msg+="#"*(new_hash)+"\n"
        new_hash//=2

    msg+="-\n"
    new_hash+=1
    for j in range(num_lines):
        msg+="#"*(new_hash)+"\n"
        new_hash*=2



    return msg


