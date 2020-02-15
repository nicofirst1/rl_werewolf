This dir contains the environments in Gym format. 
Most of them are stored in the old dir

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
[ComMaWw](gym_ww/envs/ComMaWw.py) stands for CommunicationMaWw.
Based on _MaWw_ tries to implement communication. Updates:
- Use night/day phases in observation
- number of wolves is sqrt(total player)
- add maximum days
- normalized custom metrics
- add targets instead of votes
- add communication phases
- more

### TurnEnvWw
[TurnEnvWw](gym_ww/envs/TurnEnvWw.py) stands for CommunicationMaWw.
Based on _ComMaWw_. The goal is to keep agent ids fixed to a certain role, so agents can then be used with custom policies.

The problem becomes to hide the roles from each agent, so there should be some kind of vote mixing at each turn. 

The shifting should be done on _status_map_ and _targets_ when passing observations and first thing when getting actions. So the 
shifting works as a mask to the agents while in the env everything stays the same.
Learning this shifting from observation is technically the same as guessing roles.

Moreover this implementation skips villagers during night time

## TODO
- Make logging every n episode [X]
- Try to use decorator for logging rather than if else []
