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
