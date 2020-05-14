# initialize param class
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy

from policies.RevengeTarget import RevengeTarget
from utils import Params

Params()

from models import ParametricActionsModel
from wrappers import EvaluationWrapper

from policies.RandomTarget import  RandomTarget

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
    _ = ParametricActionsModel
    ray.init(local_mode=Params.debug, logging_level=logging.WARN)

    env_configs = {'num_players': Params.num_player}

    env = EvaluationWrapper(env_configs)

    # define policies
    vill_p = (PPOTFPolicy, env.observation_space, env.action_space, {})
    ww_p=(RevengeTarget, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p=ww_p,
        vill_p=vill_p,
    )

    configs = {
        "env": EvaluationWrapper,
        "env_config": env_configs,
        "eager": False,
        "eager_tracing": False,
        "num_workers": Params.n_workers,
        "num_gpus": Params.n_gpus,
        "batch_mode": "complete_episodes",

        # PPO parameter taken from OpenAi paper
        "lr": 3e-4,
        "lambda": .95,
        "gamma": .998,
        "entropy_coeff": 0.01,
        "kl_coeff": 1.0,
        "clip_param": 0.2,
        "use_critic": True,
        "use_gae": True,
        "grad_clip": 5,

        #todo: remove this [here](https://github.com/ray-project/ray/issues/7991)
        "simple_optimizer": True,

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
        "PPO",
        local_dir=Params.RAY_DIR,
        config=configs,
        trial_name_creator=trial_name_creator,
        checkpoint_freq=Params.checkpoint_freq,
        keep_checkpoints_num=Params.max_checkpoint_keep,

    )
