from envs.old.TurnEnv import TurnEnvWw
from utils import Params

Params()

import logging

import ray

from ray import tune

from gym_ww.envs import WwEnv

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
    ray.init(local_mode=Params.debug, logging_level=logging.WARN, num_cpus=Params.n_cpus)

    env_configs = {'num_players': Params.num_player, "use_act_box": True}

    env = WwEnv(env_configs)
    space = (None, env.observation_space, env.action_space, {})

    policies = dict(
        wolf_p=space,
        vill_p=space,
    )

    configs = {
        "env": TurnEnvWw,
        "env_config": env_configs,
        "eager": False,
        "eager_tracing": False,
        "num_workers": 0,
        "batch_mode": "complete_episodes",

        "callbacks": {"on_episode_end": on_episode_end, },
        "model": {
            "use_lstm": True,
            # "max_seq_len": 10,
            "custom_preprocessor": "wwPreproc",
        },
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": mapping,

        },

    }

    analysis = tune.run(
        "DDPG",
        local_dir=Params.RAY_DIR,
        config=configs,
        trial_name_creator=trial_name_creator,

    )
    # trainer = PGTrainer(configs, PolicyWw)
    #
    # for i in tqdm(range(20)):
    #     trainer.train()
