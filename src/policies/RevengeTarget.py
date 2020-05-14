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
        assert len(state_batches) == len(self.get_initial_state())
        new_state_batches = [[
            t + 1 for t in state_batches[0]
        ]]
        return [], new_state_batches, {}

    def get_initial_state(self):
        """Returns initial RNN state for the current policy."""
        return [0]  # list of single state element (t=0)

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        """No learning."""
        return {}
