# initialize param class
import sys, os

pwd=os.getcwd().split("rl-werewolf")[0]
sys.path.extend([f'{pwd}/rl-werewolf', f'{pwd}/rl-werewolf/src', f'{pwd}/rl-werewolf/gym_ww'])


from utils import Params
Params()

from models import ParametricActionsModel
from gym_ww.wrappers import EvaluationWrapper


from policies.SimpleQPolicy import MyTFPolicy

import logging
import ray
from ray import tune

from callbacks import on_episode_end
from other.custom_utils import trial_name_creator


def mapping(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


if __name__ == '__main__':
    _=ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.INFO,
             num_cpus=8,
             num_gpus=1,
             memory=1e+10,
             object_store_memory=6e+9,
             )

    env_configs = {'num_players': Params.num_player}

    env = EvaluationWrapper(env_configs)
    space = (MyTFPolicy, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p=space,
        vill_p=space,
    )

    configs = {
        "env": EvaluationWrapper,
        "env_config": env_configs,
        "eager": False,
        "eager_tracing": False,
        "num_workers": Params.n_workers,
        #"num_gpus": Params.n_gpus,
        "batch_mode": "complete_episodes",

        "callbacks": {"on_episode_end": on_episode_end, },
        "model": {
            "use_lstm": True,
            # "max_seq_len": 10,
            "custom_model": "pa_model",  # using custom parametric action model
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping,

        },

    }

    analysis = tune.run(
        "A2C",
        local_dir=Params.RAY_DIR,
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=Params.checkpoint_freq,
        keep_checkpoints_num=Params.max_checkpoint_keep,
        queue_trials=True,

    )
