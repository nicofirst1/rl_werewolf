from ray.rllib import Policy

from policies.utils import revenge_target


class RevengeTarget(Policy):
    """Hand-coded policy that returns the id of an agent who chose the current one in the last run, if none then random """

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        self.to_kill_list = []

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions on a batch of observations."""

        observations=[elem.get('obs',{}) for elem in info_batch]
        signal_conf = self.config['env_config']['signal_length'], self.config['env_config']['signal_range']

        actions, self.to_kill_list = revenge_target(self.action_space, observations, self.to_kill_list,signal_conf)

        return actions, [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}
