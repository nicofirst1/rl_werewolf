
This dir contains tests for various parts of the project:

- [single_env.py](tests/single_env.py) can be used to test the environment workflow. It cals step and reset when the env is done.
- [multi_env.py](tests/multi_env.py) extends the previous file to multi agent systems. Using dictionaries mapping intrs to actions/obs.
- [simple_train.py](tests/simple_train.py) uses a simple rllib alghotirmt (PG) for training.
- [memory_train.py](tests/memory_train.py) extends the previous file adding LSTMs for memory.
- [simple_policy.py](tests/simple_policy.py) is used to test out policies in the [policies dir](/Users/giulia/Desktop/rl-werewolf/policies).