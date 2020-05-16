from ray.rllib.agents.ppo import DEFAULT_CONFIG
from ray.rllib.agents.ppo.ppo import get_policy_class, choose_policy_optimizer, update_kl, validate_config
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer


def mapping_dynamic(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


def post_train(trainer, result):
    win_ww = result['custom_metrics']['win_wolf_mean']
    #training_policy = trainer.config['multiagent']['policies_to_train'][0]
    mapping = trainer.config['multiagent']['policy_mapping_fn'].__name__

    # if the ww are loosing
    if win_ww <= 0.50:
        # is the start and switch the mapping to the dynamic one
        if "static" in mapping:
            trainer.config['multiagent']['policy_mapping_fn'] = mapping_dynamic

        trainer.config['multiagent']['policies_to_train'] = "wolf_p"

    # if is the start and the ww are loosing
    else:
        trainer.config['multiagent']['policies_to_train'] = "vill_p"

    result['custom_metrics']=trainer.config['multiagent']['policies_to_train']

AlternatePPOTrainer = build_trainer(
    name="AlternatePPO",
    default_config=DEFAULT_CONFIG,
    default_policy=PPOTFPolicy,
    get_policy_class=get_policy_class,
    make_policy_optimizer=choose_policy_optimizer,
    validate_config=validate_config,
    after_optimizer_step=update_kl,
    after_train_result=post_train)
