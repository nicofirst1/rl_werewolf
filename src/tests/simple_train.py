import logging
import random

import ray
from ray import tune
from ray.rllib.agents import ppo


from callbacks import on_episode_end
from gym_ww.envs import ComMaWw
from utils import Params

ray.init(local_mode=True,logging_level=logging.WARN)

def trial_name_creator(something):
    name=str(something).rsplit("_",1)[0]
    name=f"{name}_{Params.unique_id}"
    return name



configs={
        "env": ComMaWw,
        "env_config": {'num_players': 5},  # config to pass to env class

        "callbacks": {

            "on_episode_end": on_episode_end,
        },
    }

analysis = tune.run(
    "PG",
    local_dir=Params.RAY_DIR,
    config=configs,
    trial_name_creator=trial_name_creator,

)

#
# while True:
#     print(trainer.train())