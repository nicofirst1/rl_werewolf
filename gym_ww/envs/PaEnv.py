import math
import random

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from gym_ww import ww, vil
from src.other.custom_utils import str_id_map, most_frequent

####################
# global vars
####################
# penalty fro breaking a rule

####################
# names for roles
####################


CONFIGS = dict(

    existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
    penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=-1,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
        # a penalty equal to index_of(agent2,targets)*penalty
        trg_accord=-1,

    ),
    max_days=10,

    # signal is used in the communication phase to signal other agents about intentions
    # the length concerns the dimension of the signal while the components is the range of values it can fall into
    # a range value of 2 is equal to binary variable
    signal_length=10,
    signal_range=2,

    # {'agent': 5, 'attackVoteList': [], 'attackedAgent': -1, 'cursedFox': -1, 'divineResult': None, 'executedAgent': -1,  'guardedAgent': -1, 'lastDeadAgentList': [], 'latestAttackVoteList': [], 'latestExecutedAgent': -1, 'latestVoteList': [], 'mediumResult': None,  , 'talkList': [], 'whisperList': []}

)
CONFIGS['role2id'], CONFIGS['id2role'] = str_id_map(CONFIGS['existing_roles'])


class PaEnv(MultiAgentEnv):
    """


    """
    metadata = {'players': ['human']}

    def __init__(self, configs, roles=None, flex=0):
        """

        :param num_players: int, number of player, must be grater than 4
        :param roles: list of str, list of roles for each agent
        :param flex: float [0,1), percentage of targets to consider when voting, 0 is just one, depend on the number of player.
            EG:  if num_players=10 -> targets are list of 10 elements, 10*0.5=5 -> first 5 player are considered when voting
        """

        # if config is dict
        if isinstance(configs, EnvContext) or isinstance(configs, dict):
            # get num player
            try:
                num_players = configs['num_players']
            except KeyError:
                raise AttributeError(f"Attribute 'num_players' should be present in the EnvContext")


        elif isinstance(configs, int):
            # used for back compatibility
            num_players = configs
        else:
            raise AttributeError(f"Type {type(configs)} is invalid for config")

        # number of player should be more than 5
        assert num_players >= 5, "Number of player should be >= 5"

        if roles is None:
            # number of wolves should be less than villagers
            num_wolves = math.floor(math.sqrt(num_players))
            num_villagers = num_players - num_wolves
            roles = [ww] * num_wolves + [vil] * num_villagers
            # random.shuffle(roles)

        else:
            assert len(
                roles) == num_players, f"Length of role list ({len(roles)}) should be equal to number of players ({num_players})"
            num_wolves = len([elem for elem in roles if elem == ww])

        self.num_players = num_players
        self.num_wolves = num_wolves
        self.roles = roles
        self.penalties = CONFIGS['penalties']
        self.max_days = CONFIGS['max_days']
        self.signal_length = CONFIGS['signal_length']
        self.signal_range = CONFIGS['signal_range']

        # used for logging game
        self.ep_step = 0

        if flex == 0:
            self.flex = 1
        else:
            self.flex = math.floor(num_players * flex)

        # define empty attributes, refer to initialize method for more info
        self.status_map = None
        self.shuffle_map = None
        self.unshuffle_map = None
        self.is_night = True
        self.is_comm = True
        self.day_count = 0
        self.phase = 0
        self.is_done = False
        self.custom_metrics = None
        self.role_map = None
        self.initialize()

    #######################################
    #       INITALIZATION
    #######################################

    def initialize(self):
        """
        Initialize attributes for new run
        :return:
        """

        self.role_map = {idx: self.roles[idx] for idx in range(self.num_players)}

        # map to shuffle player ids at the start of each game, check the readme under PolicyWw for more info
        sh = sorted(range(self.num_players), key=lambda k: random.random())
        self.shuffle_map = {idx: sh[idx] for idx in range(self.num_players)}
        self.unshuffle_map = {sh[idx]: idx for idx in range(self.num_players)}

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

    def reset(self):
        """Resets the state of the environment and returns an initial observation.

            Returns:
                observation (object): the initial observation.
            """

        self.initialize()
        init_signal = {p: [-1] * self.signal_length for p in range(self.num_players)}
        obs = self.observe(phase=0, signal=init_signal, targets={k: -1 for k in range(self.num_players)})
        obs, _, _, _ = self.convert(obs, {}, {}, {}, 0)
        return obs

    #######################################
    #       MAIN CORE
    #######################################

    def day(self, actions, rewards):
        """
        Run the day phase, that is execute target based on votes and reward accordingly or the voting
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def execution(actions, rewards):
            """
            To be called when is execution phase
            :return:
            """

            # get the agent to be executed
            target = most_frequent(actions)

            # penalize for non divergent target
            rewards = self.target_accord(target, rewards, actions)

            # penalize target agent
            rewards[target] += self.penalties.get("death")
            # kill him
            self.status_map[target] = 0

            # update day
            self.day_count += 1

            return rewards

        # call the appropriate method depending on the phase
        if self.is_comm:
            return rewards
        else:

            rewards = {id_: val + self.penalties.get('day') for id_, val in rewards.items()}
            return execution(actions, rewards)

    def night(self, actions, rewards):
        """
        Is night, time to perform actions!
        During this phase, villagers action are not considered
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: return updated rewards
        """

        if not self.is_comm:
            # execute wolf actions
            rewards = self.wolf_action(actions, rewards)

        return rewards

    def step(self, actions_dict):
        """
        Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            actions_dict (dict): a list of action provided by the agents

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """

        # remove roles from ids
        actions_dict = {int(k.split("_")[1]): v for k, v in actions_dict.items()}

        signals, targets = self.split_target_signal(actions_dict)

        # rewards start from zero
        rewards = {id_: 0 for id_ in self.get_ids("all", alive=False)}

        # execute night action
        if self.is_night:
            rewards = self.night(targets, rewards)
        else:  # else go with day
            # apply action by day
            rewards = self.day(targets, rewards)

        # prepare for phase shifting
        is_night, is_comm, phase = self.update_phase()

        # get dones
        dones, rewards = self.check_done(rewards)
        # get observation
        obs = self.observe(phase, signals, targets)

        # initialize infos with dict
        infos = {idx: {'role': self.roles[idx]} for idx in self.get_ids("all", alive=False)}

        # convert to return in correct format, do not modify anything except for dones
        obs, rewards, dones, info = self.convert(obs, rewards, dones, infos, phase)

        # if game over reset
        if self.is_done:

            dones["__all__"] = True
            # normalize infos
        else:
            dones["__all__"] = False

        # shift phase
        self.is_night = is_night
        self.is_comm = is_comm

        return obs, rewards, dones, info

    def wolf_action(self, actions, rewards):
        """
        Perform wolf action, that is kill agent based on votes and reward
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def kill(actions, rewards):

            if not len(wolves_ids):
                raise Exception("Game not done but wolves are dead, have reset been called?")

            # get agent to be eaten
            target = most_frequent(actions)

            # penalize for different ids
            rewards = self.target_accord(target, rewards, actions)

            # kill agent
            self.status_map[target] = 0
            # penalize dead player
            rewards[target] += self.penalties.get("death")

            return rewards

        wolves_ids = self.get_ids(ww, alive=True)
        # filter action to get only wolves
        actions = {k: v for k, v in actions.items() if k in wolves_ids}

        # call the appropriate method depending on the phase
        if self.is_comm:
            return rewards
        else:
            return kill(actions, rewards)

    #######################################
    #       UPDATER
    #######################################

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
            night = True

        elif self.is_night and not self.is_comm:
            night = False
            comm = True
            phase = 1

        elif not self.is_night and self.is_comm:
            comm = False
            phase = 2
            night = False

        elif not self.is_night and not self.is_comm:
            night = True
            comm = True
            phase = 3

        else:
            raise ValueError("Something wrong when shifting phase")

        self.phase = phase
        return night, comm, phase

    #######################################
    #       UTILS
    #######################################

    def split_target_signal(self, actions_dict):
        """
        Split signal and target from the action dictionary
        :param actions_dict: dict[int->obj], map agent id to action
        :return: signals and target
        """

        # split signals from targets
        if self.signal_length > 0:
            signals = {k: v[1:] for k, v in actions_dict.items()}
            # remove signals from action dict
            targets = {k: v[0] for k, v in actions_dict.items()}
        else:
            signals = {}
            targets = actions_dict

        # apply unshuffle
        targets = {k: self.unshuffle_map[v] for k, v in targets.items()}

        return signals, targets

    def convert(self, obs, rewards, dones, info, phase):
        """
        Convert everything in correct format
        :param obs:
        :param rewards:
        :param dones:
        :param info:
        :return:
        """

        # remove villagers from night phase
        if phase in [0, 1] and False:
            rewards = {id_: rw for id_, rw in rewards.items() if self.get_ids(ww, alive=False)}
            obs = {id_: rw for id_, rw in obs.items() if self.get_ids(ww, alive=False)}
            dones = {id_: rw for id_, rw in dones.items() if self.get_ids(ww, alive=False)}
            info = {id_: rw for id_, rw in info.items() if self.get_ids(ww, alive=False)}

        # if the match is not done yet remove dead agents
        if not self.is_done:
            # filter out dead agents from rewards
            rewards = {id_: rw for id_, rw in rewards.items() if self.status_map[id_]}
            obs = {id_: rw for id_, rw in obs.items() if self.status_map[id_]}
            dones = {id_: rw for id_, rw in dones.items() if self.status_map[id_]}
            info = {id_: rw for id_, rw in info.items() if self.status_map[id_]}

        # add roles to ids for policy choosing
        rewards = {f"{self.roles[k]}_{k}": v for k, v in rewards.items()}
        obs = {f"{self.roles[k]}_{k}": v for k, v in obs.items()}
        dones = {f"{self.roles[k]}_{k}": v for k, v in dones.items()}
        info = {f"{self.roles[k]}_{k}": v for k, v in info.items()}

        # convert to float
        rewards = {k: float(v) for k, v in rewards.items()}

        return obs, rewards, dones, info

    def check_done(self, rewards):
        """
        Check if the game is over, moreover return true for dead agent in done
        :param rewards: dict, maps agent id_ to curr reward
        :return:
            dones: list of bool statement
            rewards: update rewards
        """
        dones = {id_: 0 for id_ in rewards.keys()}

        for idx in range(self.num_players):
            # done if the player is not alive
            done = not self.status_map[idx]
            dones[idx] = done

        # get list of alive agents
        alives = self.get_ids('all', alive=True)

        # if there are more wolves than villagers than they won
        wolf_won = len(self.get_ids(ww)) >= len(self.get_ids(vil))
        # if there are no more wolves than the villager won
        village_won = all([role == vil for id_, role in self.role_map.items() if id_ in alives])

        if wolf_won:  # if wolves won
            # set flag to true (for reset)
            self.is_done = True
            # reward
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('lost')

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')

        if self.day_count >= self.max_days - 1:
            self.is_done = True

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
            ids = [id_ for id_, rl in self.role_map.items() if rl == role]

        # filter out dead ones
        if alive:
            ids = [id_ for id_ in ids if self.status_map[id_]]

        return ids

    def target_accord(self, chosen_target, rewards, targets):
        """
        Reward/penalize agent based on the target chose for execution/kill depending on the choices it made.
        This kind of reward shaping is done in order for agents to output targets which are more likely to be chosen
        :param targets: dict[int->int], maps an agent to its target
        :param chosen_target: int, agent id_ chosen for execution/kill
        :param rewards: dict[int->int], map agent id_ to reward
        :return: updated rewards
        """

        for id_, vote in targets.items():
            # if the agent hasn't voted for the executed agent then it takes a penalty
            if vote != chosen_target:
                penalty = self.penalties["trg_accord"]
                rewards[id_] += penalty

        return rewards

    #######################################
    #       SPACES
    #######################################

    @property
    def action_space(self):
        """
        :return:
        """

        # the action space is made of two parts: the first element is the actual target they want to be executed
        # and the other ones are the signal space
        if self.signal_length > 0:
            space = gym.spaces.MultiDiscrete([self.num_players] * (1 + self.signal_length))
        else:
            space = gym.spaces.Discrete(self.num_players)
            space.nvec = [space.n]

        # high=[self.num_players]+[self.signal_range-1]*self.signal_length
        # low=[-1]+[0]*self.signal_length
        # space = gym.spaces.Box(low=np.array(low), high=np.array(high), dtype=np.int32)

        # should be a list of targets
        return space

    @property
    def observation_space(self):
        """
        Return observation space in gym box
        :return:
        """

        obs = dict(
            # number of days passed
            day=spaces.Discrete(self.max_days),
            # idx is agent id_, value is boll for agent alive
            status_map=spaces.MultiBinary(self.num_players),
            # number in range number of phases [com night, night, com day, day]
            phase=spaces.Discrete(4),
            # targets is now a vector, having an element outputted from each agent
            targets=gym.spaces.Box(low=-1, high=self.num_players, shape=(self.num_players,), dtype=np.int32),
            # own id
            own_id=gym.spaces.Discrete(self.num_players),

        )

        # add signal if the required
        if self.signal_length > 0:
            # signal is a matrix of dimension [num_player, signal_range]
            signal = dict(
                signal=gym.spaces.Box(low=-1, high=self.signal_range - 1, shape=(self.num_players, self.signal_length),
                                      dtype=np.int32))
            obs.update(signal)

        obs = gym.spaces.Dict(obs)

        return obs

    def observe(self, phase, signal, targets):
        """
        Return observation object
        :return:
        """

        def add_missing(signal, targets):
            """
            Add missing values (for dead agents) to targets and signal
            :param signal: ndarray, signal of size [num_player, signal_lenght]
            :param targets: dict[int->int], mapping agent ids to targets
            :return: tuple
                1: signal
                2: targets
            """

            # if the list of outputs is full then do nothing
            if len(targets) == self.num_players:
                return signal, targets

            # get indices to add
            to_add = set(range(self.num_players)) - set(targets.keys())

            # add a list of -1 of length signal_length to the signal
            sg = [-1] * self.signal_length

            # update dict with -1
            targets.update({elem: -1 for elem in to_add})

            if len(signal) > 0:
                signal.update({elem: sg for elem in to_add})

            return signal, targets

        def shuffle_sort(dictionary, shuffle_map, value_too=True):
            """
            Shuffle a dictionary given a map
            @param dictionary: dict, dictionary to shuffle
            @param shuffle_map: dict, map
            @param value_too: bool, if to shuffle the value too
            @return: shuffled dictionary
            """

            new_dict = {}
            for k, v in dictionary.items():
                nk = shuffle_map[k]

                if value_too and v in shuffle_map.keys():
                    nv = shuffle_map[v]
                    new_dict[nk] = nv
                else:
                    new_dict[nk] = v

            new_dict = {k: v for k, v in sorted(new_dict.items(), key=lambda item: item[0])}

            return new_dict

        observations = {}

        # add missing targets
        signal, targets = add_missing(signal, targets)

        # shuffle
        targets = shuffle_sort(targets, self.shuffle_map)
        signal = shuffle_sort(signal, self.shuffle_map, value_too=False)

        # stack observations
        # make matrix out of signals of size [num_player,signal_length]
        tg = np.asarray(list(targets.values()))
        if len(signal) > 0:
            sg = np.stack(list(signal.values()))
        else:
            sg = {}

        # apply shuffle to status map
        st = [self.status_map[self.shuffle_map[idx]] for idx in range(self.num_players)]

        for idx in self.get_ids("all", alive=False):
            # build obs dict
            obs = dict(
                day=self.day_count,  # day passed
                status_map=np.array(st),  # agent_id:alive?
                phase=phase,
                targets=tg,
                own_id=self.shuffle_map[idx],
            )

            if self.signal_length > 0:
                obs["signal"] = sg

            observations[idx] = obs

        return observations
