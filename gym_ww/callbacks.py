from ray.rllib.agents.callbacks import DefaultCallbacks

from utils import Params


class CustomCallbacks(DefaultCallbacks):

    def __init__(self, legacy_callbacks_dict=None):
        super().__init__(legacy_callbacks_dict)
        self.training_policy = 0

    def on_episode_end(self, worker, base_env,
                       policies, episode,
                       **kwargs):
        """
        Callback function to be called at the end of an episode, aggregates custom metrics in the episode dict
        :param info:
        :return:
        """
        # keep consistency for different versions
        try:
            cm = base_env.envs[0].custom_metrics
        except AttributeError:
            cm = base_env.envs[0].wrapped.custom_metrics

        for k, v in cm.items():
            episode.custom_metrics[k] = v

        episode.custom_metrics['training_policy'] = self.training_policy

    @staticmethod
    def mapping_dynamic(agent_id):
        if "wolf" in agent_id:
            return "wolf_p"
        elif "vil" in agent_id:
            return "vill_p"
        else:
            raise NotImplementedError(f"Policy for role {agent_id} not implemented")

    def on_train_result(self, trainer, result, **kwargs):
        vill_ww = result['custom_metrics']['win_vil_mean']
        # training_policy = trainer.config['multiagent']['policies_to_train'][0]
        mapping = trainer.config['multiagent']['policy_mapping_fn'].__name__

        if Params.alternating:
            # if the ww are loosing
            if vill_ww >= 0.65:
                # is the start and switch the mapping to the dynamic one
                if "static" in mapping:
                    trainer.config['multiagent']['policy_mapping_fn'] = self.mapping_dynamic

                trainer.config['multiagent']['policies_to_train'] = "wolf_p"
                self.training_policy = 1
                print(f"Wolf Trainig: {vill_ww}")

            # if is the start and the ww are loosing
            elif vill_ww <= 0.35:
                trainer.config['multiagent']['policies_to_train'] = "vill_p"
                self.training_policy = 0
                print(f"Will Trainig {vill_ww}")

