from ray.rllib import Policy

from policies.utils import random_non_wolf


class RandomTargetUnite(Policy):
    """Hand-coded policy that returns random actions. WW will always return a non ww index."""

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions on a batch of observations."""
        observations = [elem.get('obs', {}) for elem in info_batch]
        signal_conf = self.config['env_config']['signal_length'], self.config['env_config']['signal_range']

        action = random_non_wolf(self.action_space, observations, signal_conf, unite=True)

        return action, [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}
