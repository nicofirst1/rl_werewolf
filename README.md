## WereWolf Game
[Werewolf](https://en.wikipedia.org/wiki/Werewolf_(social_deduction_game) is a simple deduction game that can be played with at least 5 players. It is also knows as:
- Mafia
- Lupus in fabula
- Pueblo duerme
- Los Hombres Lobo de Castronegro

In its most basic version there are __villagers__ (aka. vil) and __werewolf__ (aka.  ww). 
Notice that the number of wolves should always be less than the number of vil.

The game develops into tho phases, _night_ and _day_.
#### Night
At _night_ time everyone closes their eyes, this prevents players to know which roles are assigned to other playser. 
Taking turnes each non vil player open his eyes and choose an action.
When only ww are present they open their eyes and choose someone to eat.

#### Day
During the day everyone open their eyes, assert the events of the night before (eaten players) and decide who is to be executed.
Here wolves have to be smart not to get catch and executed, to do so they lie.

#### Game over
The game ends when either there are no more ww alive or there are more wolves than vil.


#Install

To install follow the instructions:
1. (Optional) source your custom environment
2. run `pip install -e .` 
3. Install [tensorflow](https://www.tensorflow.org/install/pip)

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
- []()
- []()

### RL frameworks
- [Comparison between rl framework](https://winderresearch.com/a-comparison-of-reinforcement-learning-frameworks-dopamine-rllib-keras-rl-coach-trfl-tensorforce-coach-and-more/)
- []()
- []()
- []()