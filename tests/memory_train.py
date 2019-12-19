import logging

import ray
from ray import tune

from gym_ww.envs import ComMaWw

ray.init(local_mode=False ,logging_level=logging.WARN,num_cpus=4)


def on_episode_end(info):
    episode = info['episode']
    cm = info['env'].envs[0].custom_metrics
    for k, v in cm.items():
        episode.custom_metrics[k] = v


configs = {
    "env": ComMaWw,
    "env_config": {'num_players': 10},  # config to pass to env class

    "callbacks": { "on_episode_end": on_episode_end,},
    "model": {
        "use_lstm": True,
        "max_seq_len": 10,
    },
}

analysis = tune.run(
    "PG",
    local_dir="/Users/giulia/Desktop/rl-werewolf/ray_results",
    config=configs,
)

#
#     print(trainer.train())
