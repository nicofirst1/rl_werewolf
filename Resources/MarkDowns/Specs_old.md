## WareWolf Environment Flow

![this](Resources/Imgs/WwFlow.svg)

As shown in the image the workflow is straightforward. It is divided into 4 phases:

* 0. __Communication during Night__ : Non villager (aka wolves) agents cast a vote.
* 1. __Execution during Night__ : Non villager agents execute action. Wolves will eat an agent.
* 2. __Communication during Day__ : Every agents cast a vote.
* 3. __Communication during Night__ : Every agent vote and the majority wins.

This four phases are manages by the _is_night_ and _is_com_ boolean flags. Moreover each phase has an unique id which is fed as an observation to the model.

### Shuffeling

Un/Shuffeling is done exclusevely at the start/end of each _env.step()_ call. 

In a multi agent env each agent have to keep thier roles during the whole training phase in order to effectively learn the best strategy for their role. 
This requires to have static ids which leads to agents to map id to role, defeating the purpose of the game.

To avoid this behavior a shuffeling dictionary is randomly initialize at every _env.reset()_ call. This dictionary maps an agent id to another random id without any repetition. Each time a _env.step()_ function comes to an end, it shuffles the return ids using the latter dict.
An inverse unshuffel dictionary is then used at each _env.step()_ on the _action\_dict_ to fix the correct indices

## Rewards
A reward dictionary is initialize with 
`[agent_id]=0` at the start of the _env.step()_  funciton. Most of the rewards come from the _penalty_ entry in the __CONFIG__ dictionary:

```
 penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=-1,
        # when wolves kill someone
        kill=5,
        # when an execution is successful (no dead man execution)
        execution=2,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # when a dead man is executed
        execute_dead=rule_break_penalty,
        # given to wolves when they kill one of their kind
        kill_wolf=rule_break_penalty,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
        # a penalty equal to index_of(agent2,targets)*penalty
        trg_accord=-1,
        # targets should be a list of DIFFERENT ids for each agent, those which output same ids shall be punished
        trg_all_diff=2,

    ),
    
  
```
Where `rule_break_penalty = -50`.

### Wolves Rewards
Wolves rewards are updated as follows:

- Each time wolves kill an agent (any agent) that is in play (alive) they get a `kill`. 
- If the agent wasn't alive they get a penaly of `execute_dead`
- If the agent is another wolf then they get a penalty of `kill_wolf`

### Target Rewards
As written in the Action space session, an agent action is a list of discrete values in range `[0,env.num_player-1]`.

There is a direct correlation between an agent output and its reward:

#### Avoid Duplicates
To incentivate target difference, i.e. the number of different _agent_id_ in an output vector, a penalty is inflicet to the agent according to the following equation

```
duplicates = number of duplicates in an agent output
dead  =number of dead agents

duplicates-=dead
rewards[agent_id] += duplicates * trg_all_diff

```
 
The `duplicates-=dead` step is done in order to penalize proportionally to the number of agents still alive in the game. 

###### Example
- number of player = 10
- alive = 10
- agent output = [ 4 5 3 4 0 6 3 8 6 6 ]

Following the formula we have:

```
duplicates= 4
dead = 0
duplicate= 4-0=4
reward = 4 * -2=-8

```

- number of player = 10
- alive = 6
- agent output = [ 4 5 3 4 0 6 3 8 6 6 ]

Following the formula we have:

```
duplicates= 4
dead = 4
duplicate= 4-4=0
reward = 0 * -2=0

```

The agent is not penalized in the second case since the number of possilbe agent ids is 6, same size as its unique output vector.

#### Meaningful vote
To incentivate agents to vote in a meaningfull way, at the end of each execution phase, the agent is penalized based on the index of the executed agent in its output vector. 

###### Example

- agent output for execution [0,3,4,4,2,1,5,6,0,8]
- Executed agent = 4
- Weight for penalty : `w=indexOf(executed, output)=2` (notice how the first occurence is selected).
- Agent penalty : `trg_accord*w`

### Everyone Reward
- penalty of `day` after each phase cycle (when day and after execution)
- reward of `execution` during day if executed target is alive.
- penalty of `death` if agent dies (eaten, executed)
- reward of `victory` if agent group wins (either wolves or villagers)
- penalty of `lost` if agent group looses (either wolves or villagers)
- penalty of `execute_dead` if executed agent is already dead.

## Metrics
To understand better the nature of the learnign various metrics were added. All of them can be found in the `env.initialize_info()`, and are as follows:

```
dead_man_execution=0,  # number of times players vote to kill dead agent
dead_man_kill=0,  # number of times wolves try to kill dead agent
cannibalism=0,  # number of times wolves eat each other
suicide=0,  # number of times a player vote for itself
win_wolf=0,  # number of times wolves win
win_vil=0,  # number of times villagers win
tot_days=0,  # total number of days before a match is over
trg_diff=0,  # percentage of different votes  between targets before and after the communication phase
trg_influence=0,  # measure of how much each agent is influenced by the others     
```

Some of them can be normalized to be in range [0,1].

Special care should be given to the target metrics since they hold deeper information meening.

#### Target Difference

The `trg_diff` metric is simply the number of different agent ids between vote time and execution time. Given the output vecotr for each agent (of size `[num players, num players]`) the target difference is estimated as follow:

    diff = np.sum(cur_targets != prev_targets) / (num_player ** 2)

Where `prev_targets ` is the vote output and `cur_targets` is the execution one.

The aim of this metric is to check the consisntecy of vote/execution during the training.

#### Target Influence

Unlike the previous metric, this one is a little trickyer. It tryes to measure how much each individual agent is influenced by the general vote phase.

The alghoritm takes as input again `cur_targets` and `prev_targets` and uses a _Counter_ to keep trak of the most voted agents. Then it simply add up the squared distance between the to repetition dictionaries, after some filtering (dead agents).

The aim of this metric is seeing how much the overall vote influences the execution time. Seeing this value increasing means that agents are taking into high consideration past votes for their next one. This is strongly correlated with the `trg_accord` penalty, which penalizes agents for not being in accordance with the others on their vote.
On the other hand, when the value decreases then the agents may make use of the voting phase as something completely different than what was intended for, further metrics should be created on this subject.


#### Parametric Actions

Notice that with the introduction of the parametric action model the follwoing metrics will be zero and probably removed from the game entirerly:

- dead man execution
- dead man kill
- cannibalism



## Action Space
The action space for each agent is a vecotor of discrete values in range `[0, num players -1]` called __targets__. 

Using the `ray.rllib.MultiAgentEnv` wrapper the action object passed to the `env.step()` function is a dictionary mapping an agent id to a vector.

This vector is supposed to be used as a preference for voting. The first _n_ values from the target will be used to decide which player to eat/execute. When the time comes to kill an agent the first _n_ targets from every agent are counted and the most common is chosen (if there is no most common one then a random one is picked instead).

### Flexibility

The number _n_ of targets to be considered from an agent output is regulated by the `env.flex` parameter which can be in range `[1,num players -1]`, by default set to 1.


## Observations
The observation space is an instance of `gym.spaces.Dic` with the following entries:

```
# number of days passed
day=spaces.Discrete(self.max_days),

# idx is agent id_, value is boll for agent alive
status_map=spaces.MultiBinary(self.num_players),

# number in range number of phases [com night, night, com day, day]
phase=spaces.Discrete(4),

# targets are the preferences for each agent,
# it is basically a matrix in which rows are agents and cols are targets
targets=spaces.Box(low=-1, high=self.num_players, shape=(self.num_players, self.num_players)),
```

Most of the observations are stright forward but will be described nontheless:

- __day__: Discrete number counting the day passed during a match. In range `[1,max_days]` will be converted to a OneHotVector when feeded to the model. A day passes when the last phase (number 4) is concluded and an agent has been executed.
-  __status_map__ : MultiBinary vector of lenght _num player_. Used to map agent to the _being alive_ condition by index (0 dead, 1 alive). 
-  __phase__ : Discreate in range `[0,3]`, maps the phase integer to a OHV in the model.

#### Targets

The target observation is a `gym.spaces.Box` instance having a shape of `[num players, num players]` effectively being a matrix.

In this matrix each row is associated with an agent outputting the target action space.

It has as low the number -1 since that is the initialization number in which the agent have not outputted any targets yet. On the other hand the high value is simply the number of players.

## PA Model

To prevent the model to choose invalid action in the target vector a parametric wrapper is used aroud the original environment.

The wrapper keeps the original observation, flattening them in a numpy array, and adds a boolean action mask of size `[num. players,1]` which is then used in the model to set logits to zero for invalid actions. 

This speeds up the trainining and renders the reward shaping for such invalid actions useless. 