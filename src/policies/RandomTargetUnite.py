from ray.rllib import Policy

from policies.utils import random_non_wolf


class RandomTarget(Policy):
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
        return random_non_wolf(self.action_space, info_batch, unite=True), [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}
