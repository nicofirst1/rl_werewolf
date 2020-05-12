from ray.rllib.models.tf.fcnet_v2 import FullyConnectedNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class ParametricActionsModel(TFModelV2):
    """
    Parametric action model used to filter out invalid action from environment
    """

    def value_function(self):
        pass

    def __init__(self,
                 obs_space,
                 action_space,
                 num_outputs,
                 model_config,
                 name,
                 ):

        name="Pa_model"
        super(ParametricActionsModel, self).__init__(
            obs_space, action_space, num_outputs, model_config, name)

        # get real obs space, discarding action mask
        real_obs_space = obs_space.original_space.spaces['original_obs']

        # define action embed model
        self.action_embed_model = FullyConnectedNetwork(real_obs_space, action_space, num_outputs, model_config,
                                                        name + "_action_embed")
        self.register_variables(self.action_embed_model.variables())

    def forward(self, input_dict, state, seq_lens):
        """
        Override forward pass to mask out invalid actions

               Arguments:
                   input_dict (dict): dictionary of input tensors, including "obs",
                       "obs_flat", "prev_action", "prev_reward", "is_training"
                   state (list): list of state tensors with sizes matching those
                       returned by get_initial_state + the batch dimension
                   seq_lens (Tensor): 1d tensor holding input sequence lengths

               Returns:
                   (outputs, state): The model output tensor of size
                       [BATCH, num_outputs]

               """
        obs = input_dict['obs']

        # extract action mask  [batch size, num players]
        action_mask = obs['action_mask']
        # extract original observations [batch size, obs size]
        original_obs = obs['original_obs']

        # Compute the predicted action embedding
        # size [batch size, num players * num players]
        action_embed, _ = self.action_embed_model({
            "obs": original_obs
        })

        # Mask out invalid actions (use tf.float32.min for stability)
        # size [batch size, num players * num players]
        inf_mask = tf.maximum(tf.log(action_mask), tf.float32.min)
        inf_mask = tf.cast(inf_mask, tf.float32)

        masked_actions = action_embed + inf_mask

        # return masked action embed and state
        return masked_actions, state
