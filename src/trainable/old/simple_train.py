import logging

import ray
from ray import tune

from callbacks import on_episode_end
from gym_ww.envs import WwEnv
from other.custom_utils import trial_name_creator
from utils import Params

ray.init(local_mode=True, logging_level=logging.WARN)

configs = {
    "env": WwEnv,
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
