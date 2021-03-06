from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from models import ParametricActionsModel
from utils import Params

Params()

from wrappers import ParametricActionWrapper

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
    mod = ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.WARN, num_cpus=Params.n_cpus)

    env_configs = {'num_players': Params.num_player}

    env = ParametricActionWrapper(env_configs)
    space = (PPOTFPolicy, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p=space,
        vill_p=space,
    )

    configs = {
        "env": ParametricActionWrapper,
        "env_config": env_configs,
        "eager": False,
        "eager_tracing": False,
        "num_workers": 2,
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
        checkpoint_freq=Params.checkpoint_freq

    )
