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