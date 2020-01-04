import logging

import ray
from ray import tune
import os
from gym_ww.envs import PolicyWw
from src.utils import path

ray.init(local_mode=path.debug ,logging_level=logging.WARN,num_cpus=path.n_cpus)


def on_episode_end(info):
    episode = info['episode']
    cm = info['env'].envs[0].custom_metrics
    for k, v in cm.items():
        episode.custom_metrics[k] = v



env=PolicyWw(path.num_player)
space=(None,env.observation_space,env.action_space,{})
policies={f"p_{idx}":space for idx in range(path.num_player)}

configs = {
    "env": PolicyWw,
    "env_config": {'num_players': path.num_player},  # config to pass to env class

    "callbacks": { "on_episode_end": on_episode_end,},
    "model": {
        "use_lstm": True,
        "max_seq_len": 10,
    },
    "multiagent": {
        "policies": policies,
        "policy_mapping_fn":
            lambda agent_id: f"p_{agent_id}"
    },

}




analysis = tune.run(
    "PG",
    local_dir=path.RAY_DIR,
    config=configs,
)
