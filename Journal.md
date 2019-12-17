- To install the gym env use: `pip install -e .` 

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
This topic is discussed over [here](https://github.com/hill-a/stable-baselines/issues/108)