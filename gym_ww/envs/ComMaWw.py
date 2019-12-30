import math
import random

import gym
import numpy as np
from gym import spaces
from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from analysis import vote_difference, measure_influence
from gym_ww import logger
from utils import str_id_map, most_frequent, suicide_num, pprint

####################
# names for roles
####################
ww = "werewolf"
vil = "villager"

####################
# global vars
####################
# penalty fro breaking a rule
rule_break_penalty=-50

CONFIGS = dict(

    existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
    penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=-1,
        # when wolves kill someone
        kill=5,
        # when an execution is successful (no dead man execution)
        execution=2,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # when a dead man is executed
        execute_dead=rule_break_penalty,
        # given to wolves when they kill one of their kind
        kill_wolf=rule_break_penalty,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
        # a penalty equal to index_of(agent2,targets)*penalty
        trg_accord=-1,
        # targets should be a list of DIFFERENT ids for each agent, those which output same ids shall be punished
        trg_all_diff=rule_break_penalty,

    ),
    max_days=1000,

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
            # number of wolves should be less than villagers
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
        self.max_days=CONFIGS['max_days']
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
        self.previous_target = None
        self.custom_metrics = None

        self.initialize()

    #######################################
    #       INITALIZATION
    #######################################

    def initialize_info(self):

        self.custom_metrics = dict(
            dead_man_execution=0,  # number of times players vote to kill dead agent
            dead_man_kill=0,  # number of times wolves try to kill dead agent
            cannibalism=0,  # number of times wolves eat each other
            suicide=0,  # number of times a player vote for itself
            win_wolf=0,  # number of times wolves win
            win_vil=0,  # number of times villagers win
            tot_days=0,  # total number of days before a match is over
            trg_diff=0,  # percentage of different votes  between targets before and after the communication phase
            trg_influence=0, # measure of how much each agent is influenced by the others
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
        self.previous_target = np.zeros(shape=(self.num_players, self.num_players, self.num_players)) - 1

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
        :param actions: dict, map id_ to vote
        :param rewards: dict, maps agent id_ to curr reward
        :return: updated rewards
        """

        def execution(actions, rewards):
            """
            To be called when is execution phase
            :return:
            """

            self.custom_metrics["suicide"] += suicide_num(actions)

            # get the agent to be executed
            target = most_frequent(actions)

            # penalize for non divergent target
            rewards = self.target_accord(target, rewards, self.get_ids("all", alive=True))

            # if target is alive
            if self.status_map[target]:
                # log
                logger.debug(f"Player {target} ({self.role_map[target]}) has been executed")

                # for every agent alive, [to be executed agent too]
                for id_ in [elem for elem in rewards.keys() if self.status_map[elem]]:
                    # add/subtract penalty
                    if id_ == target:
                        rewards[id_] += self.penalties.get("death")
                    else:
                        rewards[id_] += self.penalties.get("execution")

                # kill target
                self.status_map[target] = 0
            else:
                # penalize agents for executing a dead one
                for id_ in self.get_ids("all", alive=True):
                    rewards[id_] += self.penalties.get('execute_dead')
                logger.debug(f"Players tried to execute dead agent {target}")

                # increase the number of dead_man_execution in info
                self.custom_metrics["dead_man_execution"] += 1

            # update day
            self.day_count += 1

            return rewards

        # call the appropriate method depending on the phase
        if self.is_comm:
            logger.debug("Day Time| Voting")
            return rewards
        else:
            logger.debug("Day Time| Executing")
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

        if self.is_comm:
            logger.debug("Night Time| Voting")
        else:
            logger.debug("Night Time| Eating")

        # execute wolf actions
        rewards = self.wolf_action(actions, rewards)

        # todo: implement other roles actions

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

        # rewards start from zero
        rewards = {id_: 0 for id_ in self.get_ids("all", alive=False)}

        # update target list
        actions,rewards = self.update_targets(actions_dict, rewards)

        # execute night action
        if self.is_night:
            rewards = self.night(actions, rewards)
        else:  # else go with day
            # apply action by day
            rewards = self.day(actions, rewards)

        pprint(actions_dict, self.roles, logger=logger)

        # prepare for phase shifting
        is_night, is_comm, phase = self.update_phase()

        # get dones
        dones, rewards = self.check_done(rewards)
        # get observation
        obs = self.observe(phase)

        # convert to return in correct format, do not modify anything except for dones
        obs, rewards, dones, info = self.convert(obs, rewards, dones, {})

        # if game over reset
        if self.is_done:
            self.custom_metrics["tot_days"] = self.day_count

            dones["__all__"] = True
            # normalize infos
            self.normalize_metrics()
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

            # upvote suicide info
            self.custom_metrics["suicide"] += suicide_num(actions)

            if not len(wolves_ids):
                raise Exception("Game not done but wolves are dead, have reset been called?")

            # get agent to be eaten
            target = most_frequent(actions)

            # todo: should penalize when dead man kill?
            # penalize for different ids
            rewards = self.target_accord(target, rewards, wolves_ids)

            # if target is alive
            if self.status_map[target]:
                # kill him
                self.status_map[target] = 0
                # penalize dead player
                rewards[target] += self.penalties.get("death")
                # reward wolves
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get("kill")
                logger.debug(f"Wolves killed {target} ({self.role_map[target]})")



            else:
                logger.debug(f"Wolves tried to kill dead agent {target}")
                # penalize the wolves for eating a dead player
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get('execute_dead')
                # log it
                self.custom_metrics["dead_man_kill"] += 1

            if target in wolves_ids:
                # penalize the agent for eating one of their kind
                for id_ in wolves_ids:
                    rewards[id_] += self.penalties.get('kill_wolf')
                # log it
                self.custom_metrics["cannibalism"] += 1

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
    def update_targets(self, actions_dict, rewards):
        """
        Update target attribute based on phase
        :param actions: dict, action for each agent
        :return: None
        """

        # punish agents when they do not output all different targets
        for id_,trgs in actions_dict.items():
            # if there are some same ids in the trgs vector
            if not len(np.unique(trgs))== len(trgs):
                rewards[id_]+=self.penalties["trg_all_diff"]


        self.previous_target = self.targets.copy()

        # if its night then update targets just for the ww
        if self.is_night:
            ww_ids = self.get_ids(ww, alive=True)
            for id_ in ww_ids:
                for id2 in ww_ids:
                    try:
                        self.targets[id2][id_] = actions_dict[id_]
                    except KeyError:
                        pass
        # if day update for everyone
        else:
            for id_ in range(self.num_players):
                for id2 in actions_dict.keys():
                    self.targets[id_][id2] = actions_dict[id2]
            # estimate difference
            self.custom_metrics["trg_diff"] += vote_difference(self.targets[0], self.previous_target[0])
            self.custom_metrics["trg_influence"]+=measure_influence(self.targets[0], self.previous_target[0],self.flex)

        # apply flexibility on agreement
        actions = {id_: trgs[:self.flex] for id_, trgs in actions_dict.items()}

        return actions,rewards

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

        return night, comm, phase

    #######################################
    #       UTILS
    #######################################

    def normalize_metrics(self):
        """
        In here normalization for custom metrics should be executed.
        Notice that this method needs to be called before the reset.
        :return: None
        """

        day_dep = ["dead_man_execution", "dead_man_kill", "cannibalism", "suicide", "trg_diff","trg_influence"]

        for k in day_dep:
            self.custom_metrics[k] /= (self.day_count+1)

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
            rewards = {id_: rw for id_, rw in rewards.items() if self.status_map[id_]}
            obs = {id_: rw for id_, rw in obs.items() if self.status_map[id_]}
            dones = {id_: rw for id_, rw in dones.items() if self.status_map[id_]}
            info = {id_: rw for id_, rw in info.items() if self.status_map[id_]}

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

            logger.info(f"\n{'#' * 10}\nWolves won\n{'#' * 10}\n")
            self.custom_metrics['win_wolf'] += 1

        if village_won:
            self.is_done = True
            for idx in self.get_ids(vil, alive=False):
                rewards[idx] += self.penalties.get('victory')
            for idx in self.get_ids(ww, alive=False):
                rewards[idx] += self.penalties.get('lost')
            logger.info(f"\n{'#' * 10}\nVillagers won\n{'#' * 10}\n")
            self.custom_metrics['win_vil'] += 1

        if self.day_count>=self.max_days-1:
            self.is_done=True

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

    def target_accord(self, chosen_target, rewards, voter_ids):
        """
        Reward/penalize agent based on the target chose for execution/kill depending on the choices it made.
        This kind of reward shaping is done in order for agents to output targets which are more likely to be chosen
        :param voter_ids: list[int], list of agents that voted
        :param chosen_target: int, agent id_ chosen for execution/kill
        :param rewards: dict[int->int], map agent id_ to reward
        :return: updated rewards
        """

        for id_ in voter_ids:
            votes = self.targets[id_][id_]
            # fixme: remove this when targets are exclusive
            try:
                target_idx = np.where(votes == chosen_target)[0][0]
            except IndexError:
                target_idx = self.num_players - 1
            penalty = self.penalties["trg_accord"] * target_idx
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
        # fixme: make targets exclusive

        # should be a list of targets
        return gym.spaces.MultiDiscrete([self.num_players] * self.num_players)

    @property
    def observation_space(self):
        """
        Return observation space in gym box
        :return:
        """
        obs = dict(
            # the agent role is an id_ in range 'existing_roles'
            agent_role=spaces.Discrete(len(CONFIGS['existing_roles'])),
            # number of days passed
            day=spaces.Discrete(self.max_days),
            # idx is agent id_, value is boll for agent alive
            status_map=spaces.MultiBinary(self.num_players),
            # number in range number of phases [com night, night, com day, day]
            phase=spaces.Discrete(4),
            # targets are the preferences for each agent,
            # it is basically a matrix in which rows are agents and cols are targets
            targets=spaces.Box(low=-1, high=self.num_players, shape=(self.num_players, self.num_players)),
        )
        # should be a list of targets
        return gym.spaces.Dict(obs)

    def observe(self, phase):
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
