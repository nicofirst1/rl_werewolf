def on_episode_end(info):
    """
    Callback function to be called at the end of an episode, aggregates custom metrics in the episode dict
    :param info:
    :return:
    """
    episode = info['episode']

    # keep consistency for different versions
    try:
        cm = info['env'].envs[0].custom_metrics
    except AttributeError:
        cm = info['env'].envs[0].wrapped.custom_metrics

    for k, v in cm.items():
        episode.custom_metrics[k] = v


def mapping_dynamic(agent_id):
    if "wolf" in agent_id:
        return "wolf_p"
    elif "vil" in agent_id:
        return "vill_p"
    else:
        raise NotImplementedError(f"Policy for role {agent_id} not implemented")


def on_train_result(info):
    trainer, result=info.values()
    win_ww = result['custom_metrics']['win_wolf_mean']
    # training_policy = trainer.config['multiagent']['policies_to_train'][0]
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

    result['custom_metrics']['training_policy'] = trainer.config['multiagent']['policies_to_train']

