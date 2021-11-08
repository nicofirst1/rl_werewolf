## Paper is out!
Check out the [associated paper](https://arxiv.org/abs/2106.05018) for this project on arxiv.

# WereWolf Game
[Werewolf](https://en.wikipedia.org/wiki/Werewolf_social_deduction_game) is a simple deduction game that can be played with at least 5 players. It is also knows as:

- Mafia (Mafia, International)
- Lupus in fabula (Wolf from fable, Latin)
- Pueblo duerme (Sleeping villagers, Spain)
- Los Hombres Lobo de Castronegro (Spanish, Spain)
- Μια Νύχτα στο Palermo (One night in Palermo, Greek)
- Městečko palermo (Town of Palermo, Check)
- 狼人殺 (Werewolf kill, Cinese)
- Libahunt (Werewolf,Estonia)
- Loup Garous (Werewolves, French)
- Werewölfe (Werewolves, German)
- Weerwolven (Werewolves, Dutch)

In its most basic version there are __villagers__ (aka. vil) and __werewolf__ (aka.  ww). 
Notice that the number of wolves should always be less than the number of vil.

The game develops into tho phases, _night_ and _day_.

### Night
At _night_ time everyone closes their eyes, this prevents players to know which roles are assigned to other playser. 
Taking turnes each non vil player open his eyes and choose an action.
When only ww are present they open their eyes and choose someone to eat.

### Day
During the day everyone open their eyes, assert the events of the night before (eaten players) and decide who is to be executed.
Here wolves have to be smart not to get catch and executed, to do so they lie.

### Game over
The game ends when either there are no more ww alive or there are more wolves than vil.



# Install

To install follow the instructions in the [Installation](Resources/MarkDowns/Installation.md) markdown.

## Markdowns & READMEs
The [Markdowns dir](Resources/MarkDowns) contains usefull markdowns file covering different aspect of the implementation.

Most of the developing phase in reported in the [Journal](Resources/MarkDowns/Journal.md). For current and past issue please refer to it.

For a detailed adescription on how the whole architecture has been build please refer to the [Specs file](Resources/MarkDowns/Specs.md).

In the [enviroment dir](gym_ww/envs) there is a [README](gym_ww/envs/README.md) relative to the developing of each env and the differences between them.

Finally you can find my [theis](Resources/thesis.pdf) in which all the findings and procedures are described.


## TODO
- make period policy swapping
- make playable env

## Helpful Links

### Custom gym env
- [basics](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)
- [examples](https://stackoverflow.com/questions/45068568/how-to-create-a-new-gym-environment-in-openai)
- [Tutorial](https://ai-mrkogao.github.io/reinforcement%20learning/openaigymtutorial/)
- []()

#### Multi agent
- [MA obs/action spaces utils](https://github.com/koulanurag/ma-gym/tree/master/ma_gym/envs/utils)
- [Discussion on ma openAi](https://github.com/openai/gym/issues/934)

##### Ray/Rllib
- [Ray Example](https://github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py)
- [multi-agent-and-hierarchical](https://ray.readthedocs.io/en/latest/rllib-env.html#multi-agent-and-hierarchical)
- [Docs](https://ray.readthedocs.io/en/latest/index.html)
- [Model configs](https://ray.readthedocs.io/en/latest/rllib-models.html#built-in-model-parameters)
- [Common config](https://ray.readthedocs.io/en/latest/rllib-training.html#common-parameters)
- [SelfPlay](https://github.com/ray-project/ray/issues/6669)
- [PPO Configs](https://github.com/ray-project/ray/blob/4633d81c390fd33d54aa62a5eb43fe104062bb41/rllib/agents/ppo/ppo.py#L19)
- [Understanding of ppo plots](https://medium.com/aureliantactics/understanding-ppo-plots-in-tensorboard-cbc3199b9ba2)
- []()

### RL frameworks
- [Comparison between rl framework](https://winderresearch.com/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/)
- []()
- []()
- []()
