import gym
import tqdm as tqdm

from gym_ww.envs import TurnEnvWw

num_players=10
env = TurnEnvWw(num_players)
agent_ids=env.reset().keys()
for i in tqdm.tqdm(range(1000)):
    actions={id:env.action_space.sample() for id in agent_ids}
    obs, rewards, dones, info=env.step(actions) # take a random action
    if dones["__all__"]:
        env.reset()

