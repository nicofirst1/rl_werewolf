import logging

import ray
from ray import tune

from gym_ww.envs import ComMaWw

ray.init(local_mode=True ,logging_level=logging.WARN,num_cpus=4)


def on_episode_end(info):
    episode = info['episode']
    infos = episode._agent_to_last_info[0]
    for k, v in infos.items():
        val=v
        if k!="tot_days" and "win" not in k:
            #normalize
            val=val/infos['tot_days']
        episode.custom_metrics[k] = val


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
