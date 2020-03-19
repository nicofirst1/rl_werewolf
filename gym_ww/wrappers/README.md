This dir contains the wrappers for Gym envs.
Wrapper do not contain logic for the game, they extend some functionalities such as parametric actions, metrics ... 

### PaWrapper
[PaWrapper](gym_ww/envs/PaWrapper.py) stands for ParametricActionEnvironment.
It is basically a wrapper around TurnEnvWw to allow custom action masking. 
Together with the original observation space a boolean numpy array is used as an action masking. IndexOf zeros in the
 mask array will be interpreted with non executable/eatable agents by the model. 

### EvaluationWrapper
[EvaluationWrapper](gym_ww/envs/EvalWrapper.py) stands for Evaluation Environment.
It is built on top of the ParametricActionEnvironment and it uses classes from the [evalutation dir](src/evaluation) to understand what the agents are learning.

## TODO
