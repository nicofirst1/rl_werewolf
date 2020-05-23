## WareWolf Environment Flow

As shown in the following diagram, the workflow is straightforward. It is divided into 4 phases:

0. __Communication during Night__ : Non villager (aka wolves) agents cast a vote.
1. __Execution during Night__ : Non villager agents execute action. Wolves will eat an agent.
2. __Communication during Day__ : Every agents cast a vote.
3. __Communication during Night__ : Every agent vote and the majority wins.

This four phases are manages by the _is_night_ and _is_com_ boolean flags. Moreover each phase has an unique id which is fed as an observation to the model.

![this](Resources/Imgs/WwFlow.svg)


### Shuffling

Un/shuffling is done exclusively at the start/end of each _env.step()_ call. 

In a multi agent env each agent have to keep their roles during the whole training phase in order to effectively learn the best strategy for their role. 
This requires to have static ids which leads to agents to map id to role, defeating the purpose of the game.

To avoid this behavior a shuffling dictionary is randomly initialize at every _env.reset()_ call. 
This dictionary maps an agent id to another random id without any repetition. 
Each time a _env.step()_ function comes to an end, it shuffles the return ids using the latter dict.
An inverse unshuffle dictionary is then used at each _env.step()_ on the _action\_dict_ to fix the correct indices

## Rewards
A reward dictionary is initialize with 
`[agent_id]=0` at the start of the _env.step()_  function. Most of the rewards come from the _penalty_ entry in the __CONFIG__ dictionary:

```
 penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=-1,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4] as a target and agent2 get executed then agent1 get
        # a penalty of trg_accord
        trg_accord=-1,

    ),
    
  
```


### Target Rewards
As written in the Action space session, an agent action is a list of discrete values in range `[0,env.num_player-1]`.

There is a direct correlation between an agent output and its reward:

#### Meaningful vote
To incentive agents to vote in a meaningful way, at the end of each execution phase, the agent is penalized based on the index of the executed agent in its output vector. 

###### Example

- agent output for execution [0,3,4,4,2,1,5,6,0,8]
- Executed agent = 4
- Weight for penalty : `w=indexOf(executed, output)=2` (notice how the first occurence is selected).
- Agent penalty : `trg_accord*w`

### Everyone Reward
- penalty of `day` after each phase cycle (when day and after execution)
- penalty of `death` if agent dies (eaten, executed)
- reward of `victory` if agent group wins (either wolves or villagers)
- penalty of `lost` if agent group looses (either wolves or villagers)

## Metrics
To understand better the nature of the learning various metrics were added. All of them can be found in the `env.initialize_info()`, and are as follows:

```
    suicide=0,  # number of times a player vote for itself
    win_wolf=0,  # number of times wolves win
    win_vil=0,  # number of times villagers win
    tot_days=0,  # total number of days before a match is over
    accord=0,  # number of agents that voted for someone which was not killed
```

Some of them can be normalized to be in range [0,1].



## Action Space
The action space for each agent is the following:
 `gym.spaces.MultiDiscrete([self.num_players] * (self.signal_length + 1))`

Which can be seen as a vector with two components:
1. The first element is the target which an agent want to be killed. It is an integer in range [0,num_player-1]
2. From the second element on, the vector is a sequences of integers in range __signal range__. They constitute the basis for the communication system.

Using the `ray.rllib.MultiAgentEnv` wrapper the action object passed to the `env.step()` function is a dictionary mapping an agent id to a vector.

This vector is supposed to be used as a preference for voting. The first _n_ values from the target will be used to decide which player to eat/execute. When the time comes to kill an agent the first _n_ targets from every agent are counted and the most common is chosen (if there is no most common one then a random one is picked instead).


## Observations
The observation space is an instance of `gym.spaces.Dic` with the following entries:

```
# number of days passed
day=spaces.Discrete(self.max_days),

# idx is agent id_, value is boll for agent alive
status_map=spaces.MultiBinary(self.num_players),

# number in range number of phases [com night, night, com day, day]
phase=spaces.Discrete(4),

# targets is now a vector, having an element outputted from each agent
targets=gym.spaces.Box(low=-1, high=self.num_players, shape=(self.num_players,), dtype=np.int32),
# signal is a matrix of dimension [num_player, signal_range]
signal=gym.spaces.Box(low=-1, high=self.signal_range - 1, shape=(self.num_players, self.signal_length),
                      dtype=np.int32)

# own id
own_id=gym.spaces.Discrete(self.num_players),


```

Most of the observations are straight forward but will be described nonetheless:

- __day__: Discrete number counting the day passed during a match. In range `[1,max_days]` will be converted to a OneHotVector when feeded to the model. A day passes when the last phase (number 4) is concluded and an agent has been executed.
-  __status_map__ : MultiBinary vector of lenght _num player_. Used to map agent to the _being alive_ condition by index (0 dead, 1 alive). 
-  __phase__ : Discreate in range `[0,3]`, maps the phase integer to a OHV in the model.
- __targets__ : It is simply an integer in range `[0, num player -1]`.
- __signal__ : It is a vector of length `signal_length`. Each element is an integer in range `[0,signal_range-1]`


## PA Wrapper

To prevent the model to choose invalid action in the target vector a parametric wrapper is used around the original environment.

The wrapper keeps the original observation, flattening them in a numpy array, and adds a boolean action mask of size `[num. players,1]` which is then used in the model to set logits to zero for invalid actions. 

This speeds up the training and renders the reward shaping for such invalid actions useless. 

## Eval Wrapper
An evaluation wrapper has been used to implement logging, custom metrics and more. It is built around the Pa Wrapper.

## Complete Example
A diagram example game can be found ![here](Resources/Imgs/Example_run.svg).
The image shows how observations and action are handled by the environment during a match.