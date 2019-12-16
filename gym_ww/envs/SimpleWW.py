import math
import random

import gym

from gym_ww import logger
from utils import str_id_map, most_frequent

# names for roles
ww = "werewolf"
vil = "villager"

CONFIGS = dict(

    existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
    penalties=dict(  # penalty dictionary
        day=-1,
        kill=5,
        execution=2,
        death=-5,
        victory=+10,
        lost=-10,

    ),

    # {'agent': 5, 'attackVoteList': [], 'attackedAgent': -1, 'cursedFox': -1, 'divineResult': None, 'executedAgent': -1,  'guardedAgent': -1, 'lastDeadAgentList': [], 'latestAttackVoteList': [], 'latestExecutedAgent': -1, 'latestVoteList': [], 'mediumResult': None,  , 'talkList': [], 'whisperList': []}

)
CONFIGS['role2id'], CONFIGS['id2role'] = str_id_map(CONFIGS['existing_roles'])


class SimpleWW(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, num_players, roles=None):
        # number of player should be more than 5
        assert num_players >= 5

        if roles is None:
            # number of wolfes should be less than villagers
            num_wolfes = math.floor(num_players / 2)
            num_villagers = num_players - num_wolfes
            roles = [ww] * num_wolfes + [vil] * num_villagers
            random.shuffle(roles)
        else:
            raise AttributeError(
                f"Length of role list ({len(roles)}) should be equal to number of players ({num_players})")

        self.num_players = num_players
        self.roles = roles
        self.penalties = CONFIGS['penalties']

        # action space is just an agent id
        self.action_space = gym.spaces.Discrete(num_players)

        # todo: define obs space
        self.observation_space = gym.spaces.Discrete(33)

        # define empty attributes, refer to initialize method for more info
        self.role_map = None
        self.status_map = None
        self.votes = []
        self.is_night = True
        self.day_count = 0
        self.is_done = False

        self.initialize()

    def initialize(self):
        """
        Initialize attributes for new run
        :return:
        """

        # maps agent indices to a role
        self.role_map = {i: self.roles[i] for i in range(self.num_players)}

        # maps agent indices to bool (dead=0, alive=1)
        self.status_map = {i: 1 for i in range(self.num_players)}

        # list mapping agent id with list of votes
        self.votes = {i: [] for i in range(self.num_players)}

        # bool flag to keep track of turns
        self.is_night = True

    def day(self, actions, rewards):
        """
        Run the day phase, that is execute target based on votes and reward accorndingly
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: updated rewards
        """

        # todo: make so that agents cannot choos dead player

        # update vote list
        for idx in range(self.num_players):
            # use -1 if agent is dead
            self.votes[idx].append(actions.get(idx, -1))

        # get the agent to be executed
        target = most_frequent(actions)

        # if target is alive
        if self.status_map[target]:
            # log
            logger.debug(f"Player {target} ({self.role_map[target]}) has been executed")
            logger.debug(f"Villagers votes {[elem for elem in actions.values()]}")
            # kill target
            self.status_map[target] = 0

            # for every agent alive
            for id in [elem for elem in rewards.keys() if self.status_map[elem]]:
                # add/subtract penality
                if id == target:
                    rewards[id] += self.penalties.get("death")
                else:
                    rewards[id] += self.penalties.get("execution")
        else:
            logger.debug(f"Players tried to execute dead agent {target}")

        # update day
        self.day_count += 1

        return rewards

    def night(self, actions, rewards):
        """
        Is night, time to perform actions!
        During this phase, villagers action are not considered
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: return updated rewards
        """

        # execute wolf actions
        rewards = self.wolf_action(actions, rewards)

        # todo: implement other roles actions

        return rewards

    def wolf_action(self, actions, rewards):
        """
        Perform wolf action, that is kill agent based on votes and reward
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: updated rewards
        """

        # get wolves ids
        wolves_ids = self.get_ids(ww, alive=True)

        if not len(wolves_ids):
            raise Exception("Game not done but wolves are dead")

        # get choices by wolves
        actions = [actions[id] for id in wolves_ids]

        logger.debug(f"wolves votes :{actions}")

        # todo: add penalty if wolves do not agree
        # get agent to be eaten
        target = most_frequent(actions)
        # if target is alive
        if self.status_map[target]:
            # kill him
            self.status_map[target] = 0
            # reward wolves
            for id in wolves_ids:
                rewards[id] += self.penalties.get("kill")
            logger.debug(f"Wolves killed {target} ({self.role_map[target]})")
        else:
            logger.debug(f"Wolves tried to kill dead agent {target}")

        return rewards

    def step(self, actions):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            actions (list): a list of action provided by the agents

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # filter out actions from dead agents
        actions = {k: actions[k] for k in range(self.num_players) if self.status_map[k]}

        # rewards start from zero
        rewards = {id: 0 for id in range(self.num_players)}

        # execute night action
        if self.is_night:
            logger.debug("Night Time")
            rewards = self.night(actions, rewards)
            self.is_night = not self.is_night
        else: # else go with day
            logger.debug("Day Time")
            # penalize since a day has passed
            # todo: should penalize dead players?
            rewards = {id: val + self.penalties.get('day') for id, val in rewards.items()}
            rewards = self.day(actions, rewards)
            self.is_night = not self.is_night

        # update dones
        dones, rewards = self.check_done(rewards)
        obs = self.get_observations()
        info = []

        # if game over reset
        if self.is_done:
            self.reset()

        return self.convert(obs, rewards, dones, info)

    def convert(self, obs, rewards, dones, info):
        """
        Convert everything in correct format
        :param obs:
        :param rewards:
        :param dones:
        :param info:
        :return:
        """
        return obs, rewards, dones, info

    def get_observations(self):
        """
        Return observation object
        :return:
        """

        obsvs = []

        for idx in range(self.num_players):
            # get the reward from the dict, if not there (player dead) return -1
            obs = dict(
                agent_role=CONFIGS["role2id"][self.role_map[idx]], # role of the agent, mapped as int
                status_map=self.status_map, # agent_id:alive?
                day=self.day_count,# day passed
                votes=self.votes# list of votes
            )
            obsvs.append(obs)

        return obsvs

    def check_done(self, rewards):
        """
        Check if the game is over, moreover return true for dead agent in done
        :param rewards: dict, maps agent id to curr reward
        :return:
            dones: list of bool statement
            rewards: update rewards
        """
        dones = [0 for _ in range(self.num_players)]

        for idx in range(self.num_players):
            # done is just the player status
            done = self.status_map[idx]
            dones[idx] = done

        # get list of alive agents
        alives = [id for id, alive in self.status_map.items() if alive]

        # check if either wolves or villagers won
        wolf_won = all([role == ww for id, role in self.role_map.items() if id in alives])
        village_won = all([role == vil for id, role in self.role_map.items() if id in alives])

        if wolf_won: # if wolves won
            # set flag to true (for reset)
            self.is_done = True
            # reward
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('lost')
            logger.info("Wolves won")

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')
            logger.info("Villagers won")

        return dones, rewards

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

            Returns:
                observation (object): the initial observation.
            """
        self.initialize()
        return self.get_observations()



    def get_ids(self, role, alive=True):
        """
        Return a list of ids given a role
        :param role: str, the role of the wanted ids
        :param alive: bool, if to get just alive players or everyone
        :return: list of ints
        """

        # get all the ids for a given role
        ids = [id for id, rl in self.role_map.items() if rl == role]

        # filter out dead ones
        if alive:
            ids = [id for id in ids if self.status_map[id]]

        return ids
