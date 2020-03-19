import logging
import os

import ray
from ray import tune

from gym_ww.envs import PaEnv

ray.init(local_mode=False, logging_level=logging.WARN, num_cpus=4)


def on_episode_end(info):
    episode = info['episode']
    cm = info['env'].envs[0].custom_metrics
    for k, v in cm.items():
        episode.custom_metrics[k] = v


configs = {
    "env": PaEnv,
    "env_config": {'num_players': 10},  # config to pass to env class

    "callbacks": {"on_episode_end": on_episode_end, },
    "model": {
        "use_lstm": True,
        "max_seq_len": 10,
    },
}

pwd = os.getcwd()
pwd = pwd.rsplit("/", 1)[0]

analysis = tune.run(
    "PG",
    local_dir=f"{pwd}/ray_results",
    config=configs,
)

#
#     print(trainer.train())
