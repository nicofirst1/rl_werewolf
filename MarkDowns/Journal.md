
## Reminders
- To install the gym env use: `pip install -e .` 

## Env

### Constrain votes
Player must be forced not to choose dead player to execute/eat, or else the game does not have an end. This throws the error:
```
File "/usr/local/anaconda3/envs/ww/lib/python3.6/site-packages/ray/rllib/evaluation/sample_batch_builder.py", line 154, in postprocess_batch_so_far
    "from a single trajectory.", pre_batch)
ValueError: ('Batches sent to postprocessing must only contain steps from a single trajectory.
````
- [here](https://github.com/openai/gym/issues/1264) they say to dynamically restrict actions
- [here](https://stackoverflow.com/questions/45001361/open-ai-enviroment-with-changing-action-space-after-each-step) they say to use the propriety decorator and change the action space

The latter requires to have as action-space a list of possible ids rather than a range.
This cannot be solved by using [this](https://stackoverflow.com/questions/57910677/openai-gym-action-space-how-to-limit-choices) since the action space varies continously and the agents cannot map an id with another agent.
This topic is discussed over [here](https://github.com/hill-a/stable-baselines/issues/108).

Solution for now is _reward shaping_, that is penalize for each inconsistent vote (execute/eat dead agent, eat wolf, suicide).

### Implementing communication
Agents need to communicate to either coordinate (wolves with wolves), or to choose lie (wolves vs vill).
There is a need to implement a communication phase in between night and day. 
A possible solution would be  to split the times in 4 part [night comm, night, day comm, day]. 
This will probably require different obs/action spaces for _communication_ and _standrd_ phase.
Moreover there is a need to implement an easy communication language. [Here](https://openai.com/blog/learning-to-communicate/) openAI has a paper on developing communication in MARLS.

#### First attempt
During communication phase each agent outputs a list of ids (of len _number of players_) describing their preferences on who to vote for.
This list will be fed to every other agent and mapped. The main idea is to sum up the communication part with a list of preferences. 
So each agent has a vague idea of what other agents are thinking. TThis needs to be tied somehow with the action in voting phase.
One way could be reward shaping (yet again), penalizing the agent in proportion to the target. Ex:

Agent 1 outputs [2,0,3,4,1] as favor list. The voting phase comes up and he votes for 2, but agent 4 is executed.
Then Agent 1 should be penalized by _w_*argof(target,favor list), where _w_ is a predefined weight.


Notice that this approach is useless if agents are not implemented with LSTM or some sort of memory.

##### Spaces
Both action and observation space need to change during training.
Based on [This link ](https://ai.stackexchange.com/questions/9491/inconsistent-action-space-in-reinforcement-learning) you shouldn't 
change spaces during iteration with DeepRL.


###### Action space
Action space can stay the same during standard phases `Discrete(num players)` but has to change during communication phase.
The change is just a list of the previous action so  `MultiDiscrete([self.num_players]*self.num_players)`.

OR 

Action space can permanently change to a multi MultiDiscrete. At communication phase nothing changes, at voting phase you take the first one (or most common in first n ones?).

###### Observation Space
Should the observation space stay the same with just an added value or should it change completely?
Again according to [this link ](https://ai.stackexchange.com/questions/9491/inconsistent-action-space-in-reinforcement-learning) you should keep the space should stay the same.
Introducing another map agent_id:candidates can be a solution, but then you should have one for keeping track of past scores?

This new obs should be updated during the day only for vil and everytime for wolves -> with this wolvees can communicate and know who is on their side (should it be more explicit?)

## Training

### Implementing Turns
The need to implement turns arises from the _night_ phase. The ideal implementation would skip vil during night phase.
That is, vil would be able to see just day phases not being aware of the night ones. This would prevent them from having double observations (night/day) and 
it would probably speed the training up since there is no need to understand that there is nothing you can do during night.

In [this Rllib question](https://github.com/ray-project/ray/issues/3547) the guy suggest to group the agents in different groups which will be treated differently.
If this ends up being the solution then it has to be solved during training time rather than in the env itself.
