
## Reminders
- To install the gym env use: `pip install -e .` 

# Env

## Constrain votes [Solved]
Player must be forced not to choose dead player to execute/eat, or else the game does not have an end. This throws the error:

```
File "/usr/local/anaconda3/envs/ww/lib/python3.6/site-packages/ray/rllib/evaluation/sample_batch_builder.py", line 154, in postprocess_batch_so_far
    "from a single trajectory.", pre_batch)
ValueError: ('Batches sent to postprocessing must only contain steps from a single trajectory.
```
- [here](https://github.com/openai/gym/issues/1264) they say to dynamically restrict actions
- [here](https://stackoverflow.com/questions/45001361/open-ai-enviroment-with-changing-action-space-after-each-step) they say to use the propriety decorator and change the action space

The latter requires to have as action-space a list of possible ids rather than a range.
This cannot be solved by using [this](https://stackoverflow.com/questions/57910677/openai-gym-action-space-how-to-limit-choices) since the action space varies continously and the agents cannot map an id with another agent.
This topic is discussed over [here](https://github.com/hill-a/stable-baselines/issues/108).

Solution for now is _reward shaping_, that is penalize for each inconsistent vote (execute/eat dead agent, eat wolf, suicide).

- [here](https://ai.stackexchange.com/questions/7755/how-to-implement-a-constrained-action-space-in-reinforcement-learning) they say 
to zero out the logits of illegal actions and re-normalize. This can work for everything but all different targets, can be done in ray using [this](https://ray.readthedocs.io/en/latest/rllib-models.html#custom-action-distributions)

- another option could be conditioning the action choosing on the status map, can be implemented with [this](https://ray.readthedocs.io/en/latest/rllib-models.html#custom-action-distributions).

- [here](https://www.quora.com/How-do-I-let-AI-know-that-only-some-actions-are-available-during-specific-states-in-reinforcement-learning) this guy talks about action precondition information are used in the BURLAP RL library.

- RLlib actually uses [parametric actions](https://ray.readthedocs.io/en/latest/rllib-models.html#variable-length-parametric-action-spaces) for this issue.

- I opened another [issue](https://github.com/ray-project/ray/issues/6783) which address this topic a little.

Solved using parametric actions.

Some penalties are now useless, such as the reward for killing (wolves) and execution (all).

## Implementing communication

Agents need to communicate to either coordinate (wolves with wolves), or to choose who to execute (wolves vs vill).
There is a need to implement a communication phase in between night and day. 
A possible solution would be  to split the times in 4 part [night comm, night, day comm, day]. 
This will probably require different obs/action spaces for _communication_ and _standrd_ phase.
Moreover there is a need to implement an easy communication language. [Here](https://openai.com/blog/learning-to-communicate/) openAI has a paper on developing communication in MARLS.

### First attempt 

During communication phase each agent outputs a list of ids (of len _number of players_) describing their preferences on who to vote for.
This list will be fed to every other agent and mapped. The main idea is to sum up the communication part with a list of preferences. 
So each agent has a vague idea of what other agents are thinking. This needs to be tied somehow with the action in voting phase.
One way could be reward shaping (yet again), penalizing the agent in proportion to the target. Ex:

Agent 1 outputs [2,0,3,4,1] as favor list. The voting phase comes up and he votes for 2, but agent 4 is executed.
Then Agent 1 should be penalized by _w*argof(target,favor list)_, where _w_ is a predefined weight. Having _argof(target,favor list)==3_ the penalization will be _-w*3_.


Notice that this approach is useless if agents are not implemented with LSTM or some sort of memory.

#### Spaces

Both action and observation space need to change during training.
Based on [This link ](https://ai.stackexchange.com/questions/9491/inconsistent-action-space-in-reinforcement-learning) you shouldn't 
change spaces during iteration with DeepRL.


##### Action space

Action space can stay the same during standard phases `Discrete(num players)` but has to change during communication phase.
The change is just a list of the previous action so  `MultiDiscrete([self.num_players]*self.num_players)`.

OR 

Action space can permanently change to a multi `MultiDiscrete`. At communication phase nothing changes, at voting phase you take the first one (or most common in first _n_ ones?). Currently this is the choosen approach.

##### Observation Space

Should the observation space stay the same with just an added value or should it change completely?
Again according to [this link ](https://ai.stackexchange.com/questions/9491/inconsistent-action-space-in-reinforcement-learning) you should keep the space the same.
Introducing another map agent_id:candidates can be a solution, but then you should have one for keeping track of past scores?

This new obs should be updated during the day only for vil and everytime for wolves -> with this wolves can communicate and know who is on their side (should it be more explicit?)

For this to be possible the target list must contain exclusive number....fuck

### Second attempt [Failed]
Given the high complexity which arises from the previous form of communication, we decided to make things simpler. 
Agents will now have a separate observation/action space for communication and for execution. For this a new branch has been created.

##### Action space
The now action space is a dict type containing the following values:
- target: `gym.spaces.Discrete(self.num_players)` an int for the execution part.
- signal: `gym.spaces.MultiDiscrete([self.signal_range]*self.signal_length)` parameter used for communication, 
in its simplest form is a boolean integer, both the length of the signal as well as the range can be chosen arbitrarily 
There is a problem using a dict as action space which is tracked ever [here](https://github.com/hill-a/stable-baselines/issues/133).
The problem is an active field of research, for now there is no way to output two different things using the same model.

##### Obs space
The observation space stays similar as the previous one but modifies the targets which now becomes a Discrete and adds the signal one too 

### Third attempt 
Another way to deal with this problem is to downsample the model output. 
No need to downsample if you use a parametric mask for the first signal range value of the signal.

#### Action space
In this instance the action space will be similar to the previous one, but restricted in the length.
The type will be a MultiDiscrete with the following params: 
`gym.spaces.MultiDiscrete([self.signal_range]*(self.signal_length+1))`

Technically speaking the output is the same but practically it will be split in two parts:
1. the first element will serve as effective target output, specifying the agent to be killed
2. the other part of the vector will be the signaling part which length can range from 0 to infinity. 

### 4th attempt [not working]

By using 
``` 
high=[self.num_players]+[self.signal_range-1]*self.signal_length
low=[-1]+[0]*self.signal_length
space = gym.spaces.Box(low=np.array(high), high=np.array(low), dtype=np.int32)
```
Instead there is no need to need to downsample the signal to the required range of values. As before the first value is
 used for the execution while the second one onward are used for communication

It doesn't work since a box output is considered a real number anyhow

# Training

## Implementing Turns [Solved]
The need to implement turns arises from the _night_ phase. The ideal implementation would skip vil during night phase.
That is, vil would be able to see just day phases not being aware of the night ones. This would prevent them from having double observations (night/day) and 
it would probably speed the training up since there is no need to understand that there is nothing you can do during night.

In [this Rllib question](https://github.com/ray-project/ray/issues/3547) the guy suggest to group the agents in different groups which will be treated differently.
If this ends up being the solution then it has to be solved during training time rather than in the env itself.

The previous link refers to the grouping of agents which will then behave as a single one, this is not what we need right now so next step would be trying to implment a [custom training step](https://ray.readthedocs.io/en/latest/rllib-training.html#custom-training-workflows). Maybe subclassing the [trainable](https://ray.readthedocs.io/en/latest/tune-usage.html#trainable-api) class can be an idea.

I opened an [issue on github](https://github.com/ray-project/ray/issues/6757). Solved

## Observations

- With low _dead_man_execution_ penalty agents learn to never make the game end by voting dead mans. The game crashes when total number of days exceeds the maximum. For this penalty has been increased from -2 to -10 (same as loosing)
- Should build abstract policy for player to avoid illegal actions, the class can then be overridden for roles

## Communication influence
As today three cases has been trained from zero, where the only difference lays in the communication:
- No communication 
- single bit communication
- 10 bit communication
The results are the following:
- There seems to be no difference between victories and losses
- The single bit case has a day length increase sharper than the other two at the start, but then it converges
- suicide is slightly lower for single bit communication
- accord is the same for everyone

Next step will be to increase the amount of communication available between agents.