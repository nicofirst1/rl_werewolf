
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

### Observation after ww turn (phase 1->2) [Solved]
Since we are skipping turns, at the end of phase 1, where the ww have killed someone, the observation is passed to the ww only.
So that, at the start of phase 2 (vill communication), the vill will not know immediatly who died. The dead agent will
 be the one with -1 as target and signal but the actual status map will be upated only on the next phase (3).
I can either leave it as it is and let the agent understand that the agent suddenly outputting -1 is dead, or do the following:
- at the end of phase 1 update only the targets for ww while keeping the previous one for everyone else
The second approach must be taken since, if we pass the just the ww at the end of phase 1, the observation for phase 2 will be just coming from ww

Solution:
The solution has been to pass a padded array (-1) to the vill while leaving the changes for the ww.


## Communication influence

# Evaluation
The problem of evaluation is strictly tied to the understanding of the agents policies. Since there is no easy way to do it, 
the project will focus on the evaluation of learning villagers against fixed policy werewolves.
The evaluation will be split in consecutive steps.



## Baseline
First we will run a sufficient number of iteration for stupid random players (both ww and vil). 
This will give us a baseline of how more probable is for ww to win.

For 8 players for 1000 runs:
- Mean value of win_wolf is : 0.619 +- 0.4856325771609643
- Mean value of win_vil is : 0.381 +- 0.4856325771609643
- Mean value of tot_days is : 4.796 +- 3.0578397603537044
- Mean value of accord is : 0.524575440287226 +- 0.12915321871016025

For 9 players for 10000 runs:
- Mean value of suicide is : 0.06612089805366592 +- 0.04744384486583312
- Mean value of win_wolf is : 0.813 +- 0.3899115284266419
- Mean value of win_vil is : 0.187 +- 0.3899115284266419
- Mean value of tot_days is : 4.996 +- 3.1698555172121017
- Mean value of accord is : 0.475078605914588 +- 0.1261403740559999

For 10 players for 10000 runs:
- Mean value of suicide is : 0.0626846341962711 +- 0.04274308308186182
- Mean value of win_wolf is : 0.806 +- 0.39542888108988705
- Mean value of win_vil is : 0.194 +- 0.39542888108988705
- Mean value of tot_days is : 6.101 +- 2.860209607703603
- Mean value of accord is : 0.4786981222442681 +- 0.11413458787911061

For 20 players for 1000 runs:
- Mean value of win_wolf is : 0.94 +- 0.23748684174075838
- Mean value of win_vil is : 0.06 +- 0.23748684174075838
- Mean value of tot_days is : 8.85 +- 0.7235329985563893
- Mean value of accord is : 0.327086161883342 +- 0.05682728858310628

Based on these results, the environment will be trained with 9 agents (3 ww and 6 vill)


## Learning Vill
Then the vill will be able to learn against random ww.
In this phase we will vary the communication length and range to see if there are differences.

### Revenge WW
Then the ww's policy will change, and they will become revengeful ww.
Again in this phase the communication will be varied 

### Defensive WW
Finally ww's the policy will switch again to the defense one.
