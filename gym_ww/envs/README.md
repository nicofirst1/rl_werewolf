This dir contains the environments in Gym format.

### SimpleWw
[SimpleWw](gym_ww/envs/SimpleWW.py) is the first approach to implementing the game.
It has almost every core feature the game has to offer but does not support multiple agents.
This environment does not support training, just testing with the [env test script](tests/env_test.py).


### MaWw
[MaWw](gym_ww/envs/MaWw.py) stands for MultiAgentWereWolf. It is based on _SimpleWw_ with the following upgrades:
- Support for multi agent
- reward shaping for invalid decisions (refer to the [Journal](MarkDowns/Journal.md), section Env/Constrain_Votes).
- Correct observation space
- info attribute for debug 
- shuffle roles at every reset
- more reset fixes
- Trainable 

### ComMaWw
[ComMaWw](gym_ww/envs/ComMaWw.py) stands from CommunicationMaWw.
Based on _MaWw_ tries to implement communication.