from random import random

from evaluation import Prof
from utils import Params
from utils.serialization import load_pkl

p=Prof()
episodes=load_pkl(Params.episode_file)
{k:v.complete_episode_info() for k,v in episodes.items()}
{k:v.stack_agent_targets() for k,v in episodes.items()}
p.episodes=episodes

ep=random.choice(episodes)
ep.agent_diff(0)
a=1