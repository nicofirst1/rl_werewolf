
This dir contains tests for various parts of the project.

OLD files are in the [old](src/tests/old) directory.

- [single_env.py](src/tests/old/single_env.py) can be used to test the environment workflow. It cals step and reset when the env is done.
- [multi_env.py](src/tests/old/multi_env.py) extends the previous file to multi agent systems. Using dictionaries mapping intrs to actions/obs.
- [simple_train.py](src/tests/old/simple_train.py) uses a simple rllib alghotirmt (PG) for training.
- [memory_train.py](src/tests/old/memory_train.py) extends the previous file adding LSTMs for memory.
- [multi_policy_train.py](src/tests/old/multi_policy_train.py) introduces multi agents, custom preprocessor and others.
- [parametric_action.py](src/tests/old/parametric_action.py) Uses parametric action to mask out invalid actions.

The most recent one is the [train](src/tests/train.py) file. It uses the [Evaluatin Wrapper](gym_ww/wrappers/EvalWrapper.py) to log and train.