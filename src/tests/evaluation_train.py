from models import ParametricActionsModel
from utils import Params
from wrappers import EvaluationEnv

Params()

from policies.SimpleQPolicy import MyTFPolicy

import logging
import ray
from ray import tune

from callbacks import on_episode_end
from other.utils import trial_name_creator


def mapping(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


if __name__ == '__main__':
    ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.WARN, num_cpus=Params.n_cpus)

    env_configs = {'num_players': Params.num_player}

    env = EvaluationEnv(env_configs)
    space = (MyTFPolicy, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p=space,
        vill_p=space,
    )

    configs = {
        "env": EvaluationEnv,
        "env_config": env_configs,
        "eager": False,
        "eager_tracing": False,
        "num_workers": 0,
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

    )
