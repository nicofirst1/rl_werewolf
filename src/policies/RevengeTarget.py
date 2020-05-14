from ray.rllib import Policy

class RevengeTarget(Policy):
    """Hand-coded policy that returns the id of an agent who chose the current one in the last run, if none then random """

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):
        """Compute actions on a batch of observations."""
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}