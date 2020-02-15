This dir contains the wrappers for Gym envs.

### PaEnv
[PaEnv](gym_ww/envs/PaEnv.py) stands for ParametricActionEnvironment.
It is basically a wrapper around TurnEnvWw to allow custom action masking. 
Together with the original observation space a boolean numpy array is used as an action masking. IndexOf zeros in the
 mask array will be interpreted with non executable/eatable agents by the model. 

### EvalEnv
[EvalEnv](gym_ww/envs/EvalEnv.py) stands for Evaluation Environment.
It is built on top of the ParametricActionEnvironment and it uses classes from the [evalutation dir](src/evaluation) to understand what the agents are learning.

## TODO
