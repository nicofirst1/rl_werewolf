import gym

from gym_ww.envs import ComMaWw

num_players=5
env = ComMaWw(num_players)
for i in range(1000):
    actions={id:env.action_space.sample() for id in range(num_players)}
    obs, rewards, dones, info=env.step(actions) # take a random action

env.close()
