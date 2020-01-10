import logging
from functools import reduce

import gym
import ray
import os

from ray.rllib.agents.ppo import PPOTrainer
from ray.rllib.models import ActionDistribution, ModelCatalog, Preprocessor
from ray.rllib.utils.error import UnsupportedSpaceException
from tqdm import tqdm

from callbacks import on_episode_end
from gym_ww.envs import PolicyWw
from src.utils import path



ray.init(local_mode=path.debug ,logging_level=logging.WARN,num_cpus=path.n_cpus)




env=PolicyWw(path.num_player)
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
    "env_config": {'num_players': path.num_player},  # config to pass to env class

    "callbacks": { "on_episode_end": on_episode_end,},
    "model": {
        "use_lstm": True,
        "max_seq_len": 10,
        "vf_share_layers": True,
        "custom_preprocessor": "wwPreproc",
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn":mapping,


    },

}


trainer = PPOTrainer(configs, PolicyWw)

for i in tqdm(range(20)):
    print("Start training")
    trainer.train()