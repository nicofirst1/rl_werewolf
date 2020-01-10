import logging
from functools import reduce

import gym
import ray
import os

from ray import tune
from ray.rllib.agents.pg import PGTrainer
from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ActionDistribution, ModelCatalog, Preprocessor
from ray.rllib.utils.error import UnsupportedSpaceException
from tqdm import tqdm

from callbacks import on_episode_end
from gym_ww.envs import PolicyWw
from other.utils import trial_name_creator
from utils import Params

Params()

ray.init(local_mode=Params.debug ,logging_level=logging.WARN,num_cpus=Params.n_cpus)


env_configs={'num_players': Params.num_player,"use_act_box":True}

env=PolicyWw(env_configs)
space=(None,env.observation_space,env.action_space,{})

policies=dict(
    wolf_p=space,
    vill_p=space,
)

def mapping(agent_id):

    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")



configs = {
    "env": PolicyWw,
    "env_config": env_configs,
    "eager": False,
    "eager_tracing":False,
    "num_workers": 0,
    "batch_mode":"complete_episodes",

    "callbacks": { "on_episode_end": on_episode_end,},
    "model": {
        "use_lstm": True,
        #"max_seq_len": 10,
        "custom_preprocessor": "wwPreproc",
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn":mapping,


    },

}


analysis = tune.run(
    "DDPG",
    local_dir=Params.RAY_DIR,
    config=configs,
    trial_name_creator=trial_name_creator,

)
# trainer = PGTrainer(configs, PolicyWw)
#
# for i in tqdm(range(20)):
#     trainer.train()