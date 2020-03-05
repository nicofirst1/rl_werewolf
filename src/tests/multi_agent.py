import tqdm as tqdm

from gym_ww.envs import TurnEnvWw

if __name__ == '__main__':

    env_configs = {'num_players': 10, "use_act_box": True}

    env = TurnEnvWw(env_configs)
    agent_ids = env.reset().keys()
    for i in tqdm.tqdm(range(100)):
        actions = {id: env.action_space.sample() for id in agent_ids}
        obs, rewards, dones, info = env.step(actions)  # take a random action
        if dones["__all__"]:
            env.reset()
