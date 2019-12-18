import math
import random

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from gym_ww import logger
from utils import str_id_map, most_frequent, suicide_num, pprint

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
        execute_dead=-2,
        kill_wolf=-5,

    ),

    # {'agent': 5, 'attackVoteList': [], 'attackedAgent': -1, 'cursedFox': -1, 'divineResult': None, 'executedAgent': -1,  'guardedAgent': -1, 'lastDeadAgentList': [], 'latestAttackVoteList': [], 'latestExecutedAgent': -1, 'latestVoteList': [], 'mediumResult': None,  , 'talkList': [], 'whisperList': []}

)
CONFIGS['role2id'], CONFIGS['id2role'] = str_id_map(CONFIGS['existing_roles'])


class ComMaWw(MultiAgentEnv):
    """
    MaWw:
    In this basic version ww cannot communicate with each other and they are not aware of the identities of their companions.
    Moreover player does not have memory of past events.

    ComMaWw:

    """
    metadata = {'players': ['human']}

    def __init__(self, num_players, roles=None, flex=0):
        """

        :param num_players: int, number of player, must be grater than 4
        :param roles: list of str, list of roles for each agent
        :param flex: float [0,1), percentage of targets to consider when voting, 0 is just one, depend on the number of player.
            EG:  if num_players=10 -> targets are list of 10 elements, 10*0.5=5 -> first 5 player are considered when voting
        """

        if isinstance(num_players, EnvContext):
            try:
                num_players = num_players['num_players']
            except KeyError:
                raise AttributeError(f"Attribute 'num_players' should be present in the EnvContext")

        # number of player should be more than 5
        assert num_players >= 5, "Number of player should be >= 5"

        if roles is None:
            # number of wolfes should be less than villagers
            num_wolves = math.floor(math.sqrt(num_players))
            num_villagers = num_players - num_wolves
            roles = [ww] * num_wolves + [vil] * num_villagers
            random.shuffle(roles)
            logger.info(f"Starting game with {num_players} players: {num_villagers} {vil} and {num_wolves} {ww}")
        else:
            assert len(
                roles) == num_players, f"Length of role list ({len(roles)}) should be equal to number of players ({num_players})"

        self.num_players = num_players
        self.roles = roles
        self.penalties = CONFIGS['penalties']
        if flex == 0:
            self.flex = 1
        else:
            self.flex = math.floor(num_players * flex)

        # define empty attributes, refer to initialize method for more info
        self.role_map = None
        self.status_map = None
        self.is_night = True
        self.is_comm = True
        self.day_count = 0
        self.is_done = False
        self.targets = None

        self.initialize()

    #######################################
    #       INITALIZATION
    #######################################

    def initialize_info(self):

        self.infos = dict(
            dead_man_execution=0,  # number of times players vote to kill dead agent
            dead_man_kill=0,  # number of times wolves try to kill dead agent
            cannibalism=0,  # number of times wolves eat each other
            suicide=0,  # number of times a player vote for itself
            win_wolf=0,  # number of times wolves win
            win_vil=0,  # number of times villagers win
            tot_days=0,  # total number of days before a match is over
        )

    def initialize(self):
        """
        Initialize attributes for new run
        :return:
        """

        # shuffle roles
        random.shuffle(self.roles)

        # maps agent indices to a role
        self.role_map = {i: self.roles[i] for i in range(self.num_players)}

        # list for agent status (dead=0, alive=1)
        self.status_map = [1 for _ in range(self.num_players)]

        # bool flag to keep track of turns
        self.is_night = True

        # first phase is communication night phase
        self.is_comm = True

        # reset is done
        self.is_done = False

        # reset day
        self.day_count = 0

        # reset info dict
        self.initialize_info()

        # tensor of shape [num_players,num_players,num_players]
        # each row belongs to an agent, each column is the preference list for that agent
        # since some agents are not suppose to see changes in the previous matrix,
        # the third dimension is what each agent know/can see.
        # So it would be [agent who know this,agent who voted,target ]
        self.targets = np.zeros(shape=(self.num_players, self.num_players, self.num_players)) - 1

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

            Returns:
                observation (object): the initial observation.
            """
        logger.info("Reset called")
        self.initialize()
        return self.observe(phase=0)

    #######################################
    #       MAIN CORE
    #######################################

    def day(self, actions, rewards):
        """
        Run the day phase, that is execute target based on votes and reward accordingly or the voting
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: updated rewards
        """

        def execution(actions, rewards):
            """
            To be called when is execution phase
            :return:
            """

            self.infos["suicide"] += suicide_num(actions)

            # get the agent to be executed
            target = most_frequent(actions)
            logger.debug(f"Villagers votes {actions}")

            # if target is alive
            if self.status_map[target]:
                # log
                logger.debug(f"Player {target} ({self.role_map[target]}) has been executed")
                # kill target
                self.status_map[target] = 0

                # for every agent alive
                for id in [elem for elem in rewards.keys() if self.status_map[elem]]:
                    # add/subtract penalty
                    if id == target:
                        rewards[id] += self.penalties.get("death")
                    else:
                        rewards[id] += self.penalties.get("execution")
            else:
                # penalize agents for executing a dead one
                for id in self.get_ids("all", alive=True):
                    rewards[id] += self.penalties.get('execute_dead')
                logger.debug(f"Players tried to execute dead agent {target}")

                # increase the number of dead_man_execution in info
                self.infos["dead_man_execution"] += 1

            # update day
            self.day_count += 1

            return rewards


        # call the appropriate method depending on the phase
        if self.is_comm:
            logger.debug("Day Time| Voting")
            return rewards
        else:
            logger.debug("Day Time| Executing")
            rewards = {id: val + self.penalties.get('day') for id, val in rewards.items()}
            return execution(actions, rewards)

    def night(self, actions, rewards):
        """
        Is night, time to perform actions!
        During this phase, villagers action are not considered
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: return updated rewards
        """

        if self.is_comm:
            logger.debug("Night Time| Voting")
        else:
            logger.debug("Night Time| Eating")



        # execute wolf actions
        rewards = self.wolf_action(actions, rewards)

        # todo: implement other roles actions

        return rewards

    def step(self, actions):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            actions (dict): a list of action provided by the agents

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        pprint(actions,self.roles,logger=logger)
        # update target list
        actions = self.update_targets(actions)

        # rewards start from zero
        rewards = {id: 0 for id in self.get_ids("all", alive=False)}

        # execute night action
        if self.is_night:
            rewards = self.night(actions, rewards)
        else:  # else go with day
            # apply action by day
            rewards = self.day(actions, rewards)

        # prepare for phase shifting
        is_night, is_comm, phase = self.update_phase()


        # get dones
        dones, rewards = self.check_done(rewards)
        # get observation
        obs = self.observe(phase)
        # ad infos to every agent
        info = {id: self.infos for id in self.get_ids("all", alive=False)}

        # convert
        obs, rewards, dones, info = self.convert(obs, rewards, dones, info)

        # if game over reset
        if self.is_done:
            self.infos["tot_days"] = self.day_count

            dones["__all__"] = True
        else:
            dones["__all__"] = False


        # shift phase
        self.is_night=is_night
        self.is_comm=is_comm

        return obs, rewards, dones, info

    def wolf_action(self, actions, rewards):
        """
        Perform wolf action, that is kill agent based on votes and reward
        :param actions: dict, map id to vote
        :param rewards: dict, maps agent id to curr reward
        :return: updated rewards
        """

        def kill(actions, rewards):

            # upvote suicide info
            self.infos["suicide"] += suicide_num(actions)

            if not len(wolves_ids):
                raise Exception("Game not done but wolves are dead")

            # get choices by wolves
            #actions = [actions[id] for id in wolves_ids]

            logger.debug(f"wolves votes :{actions}")

            # todo: add penalty if wolves do not agree
            # get agent to be eaten
            target = most_frequent(actions)
            # if target is alive
            if self.status_map[target]:
                # kill him
                self.status_map[target] = 0
                # penalize dead player
                rewards[target] += self.penalties.get("death")
                # reward wolves
                for id in wolves_ids:
                    rewards[id] += self.penalties.get("kill")
                logger.debug(f"Wolves killed {target} ({self.role_map[target]})")



            else:
                logger.debug(f"Wolves tried to kill dead agent {target}")
                # penalize the wolves for eating a dead player
                for id in wolves_ids:
                    rewards[id] += self.penalties.get('execute_dead')
                # log it
                self.infos["dead_man_kill"] += 1

            if target in wolves_ids:
                # penalize the agent for eating one of their kind
                for id in wolves_ids:
                    rewards[id] += self.penalties.get('kill_wolf')
                # log it
                self.infos["cannibalism"] += 1

            return rewards

        wolves_ids = self.get_ids(ww, alive=True)
        # filter action to get only wolves
        actions = {k: v for k, v in actions.items() if k in wolves_ids}

        # call the appropriate method depending on the phase
        if self.is_comm:
            logger.debug(f"Wolves votes : {actions}")
            return rewards
        else:
            return kill(actions, rewards)

    #######################################
    #       UPDATER
    #######################################
    def update_targets(self, actions_dict):
        """
        Update target attribute based on phase
        :param actions: dict, action for each agent
        :return: None
        """

        # if its night then update targets just for the ww
        if self.is_night:
            ww_ids = self.get_ids(ww, alive=True)
            for id in ww_ids:
                for id2 in ww_ids:
                    try:
                        self.targets[id2][id] = actions_dict[id]
                    except KeyError:
                        pass
        # if day update for everyone
        else:
            for id in range(self.num_players):
                for id2 in actions_dict.keys():
                    self.targets[id][id2] = actions_dict[id2]

        # apply flexibility on agreement
        actions = {id: trgs[:self.flex] for id, trgs in actions_dict.items()}
        return actions

    def update_phase(self):
        """
        Shift the phase to the next one, keep elif explicit for readability
        :return:
            night: bool, value of is_night
            comm: bool, value of is_com
            phase: int, value of phase
        """



        if self.is_night and self.is_comm:
            comm = False
            phase = 0
            night=True

        elif self.is_night and not self.is_comm:
            night = False
            comm = True
            phase = 1

        elif not self.is_night and self.is_comm:
            comm = False
            phase = 2
            night=False

        elif not self.is_night and not self.is_comm:
            night = True
            comm = True
            phase = 3

        else:
            raise ValueError("Something wrong when shifting phase")

        return night, comm, phase

    #######################################
    #       UTILS
    #######################################

    def convert(self, obs, rewards, dones, info):
        """
        Convert everything in correct format
        :param obs:
        :param rewards:
        :param dones:
        :param info:
        :return:
        """

        # if the match is not done yet remove dead agents
        if not self.is_done:
            # filter out dead agents from rewards
            rewards = {id: rw for id, rw in rewards.items() if self.status_map[id]}
            obs = {id: rw for id, rw in obs.items() if self.status_map[id]}
            dones = {id: rw for id, rw in dones.items() if self.status_map[id]}
            info = {id: rw for id, rw in info.items() if self.status_map[id]}

        return obs, rewards, dones, info

    def check_done(self, rewards):
        """
        Check if the game is over, moreover return true for dead agent in done
        :param rewards: dict, maps agent id to curr reward
        :return:
            dones: list of bool statement
            rewards: update rewards
        """
        dones = {id: 0 for id in rewards.keys()}

        for idx in range(self.num_players):
            # done if the player is not alive
            done = not self.status_map[idx]
            dones[idx] = done

        # get list of alive agents
        alives = self.get_ids('all', alive=True)

        # if there are more wolves than villagers than they won
        wolf_won = len(self.get_ids(ww)) > len(self.get_ids(vil))
        # if there are no more wolves than the villager won
        village_won = all([role == vil for id, role in self.role_map.items() if id in alives])

        if wolf_won:  # if wolves won
            # set flag to true (for reset)
            self.is_done = True
            # reward
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('lost')

            logger.info(f"\n{'#' * 10}\nWolves won\n{'#' * 10}\n")
            self.infos['win_wolf'] += 1

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')
            logger.info(f"\n{'#' * 10}\nVillagers won\n{'#' * 10}\n")
            self.infos['win_vil'] += 1

        return dones, rewards

    def get_ids(self, role, alive=True):
        """
        Return a list of ids given a role
        :param role: str, the role of the wanted ids
        :param alive: bool, if to get just alive players or everyone
        :return: list of ints
        """

        if role == "all":
            ids = list(self.role_map.keys())
        else:
            # get all the ids for a given role
            ids = [id for id, rl in self.role_map.items() if rl == role]

        # filter out dead ones
        if alive:
            ids = [id for id in ids if self.status_map[id]]

        return ids

    #######################################
    #       SPACES
    #######################################

    @property
    def action_space(self):
        """
        :return:
        """
        # should be a list of targets
        return gym.spaces.MultiDiscrete([self.num_players] * self.num_players)

    @property
    def observation_space(self):
        """
        Return observation space in gym box
        :return:
        """
        obs = dict(
            # the agent role is an id in range 'existing_roles'
            agent_role=spaces.Discrete(len(CONFIGS['existing_roles'])),
            # number of days passed, todo: have inf or something
            day=spaces.Discrete(999),
            # idx is agent id, value is boll for agent alive
            status_map=spaces.MultiBinary(self.num_players),
            # number in range number of phases [com night, night, com day, day]
            phase=spaces.Discrete(4),
            # targets are the preferences for each agent,
            # it is basically a matrix in which rows are agents and cols are targets
            targets=spaces.Box(low=-1, high=self.num_players, shape=(self.num_players, self.num_players)),
        )
        # should be a list of targets
        return gym.spaces.Dict(obs)

    def observe(self,phase):
        """
        Return observation object
        :return:
        """

        observations = {}


        for idx in self.get_ids("all", alive=False):
            # build obs dict
            obs = dict(
                agent_role=CONFIGS["role2id"][self.role_map[idx]],  # role of the agent, mapped as int
                status_map=np.array(self.status_map),  # agent_id:alive?
                day=self.day_count,  # day passed
                phase=phase,
                targets=self.targets[idx]
            )

            observations[idx] = obs

        return observations


