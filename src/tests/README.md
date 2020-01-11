
This dir contains tests for various parts of the project:

- [single_env.py](src/tests/single_env.py) can be used to test the environment workflow. It cals step and reset when the env is done.
- [multi_env.py](src/tests/multi_env.py) extends the previous file to multi agent systems. Using dictionaries mapping intrs to actions/obs.
- [simple_train.py](src/tests/simple_train.py) uses a simple rllib alghotirmt (PG) for training.
- [memory_train.py](src/tests/memory_train.py) extends the previous file adding LSTMs for memory.
- [multi_policy_train.py](src/tests/multi_policy_train.py) introduces multi agents, custom preprocessor and others.