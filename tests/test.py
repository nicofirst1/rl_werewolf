import gym
import gym_ww
num_players=5
env = gym.make('simpleWW-v0',num_players=num_players)
for _ in range(1000):
    actions=[env.action_space.sample() for _ in range(num_players)]
    obs, rewards, dones, info=env.step(actions) # take a random action

env.close()
