from gym.envs.registration import register
import logging

from ray.rllib.models import ModelCatalog

from WwPreprocessor import WwPreprocessor
from utils import Params

ww = "werewolf"
vil = "villager"




ModelCatalog.register_custom_preprocessor("wwPreproc", WwPreprocessor)


# Register envs
register(
    id='simpleWW-v0',
    entry_point='gym_ww.envs:SimpleWW',
)

register(
    id='MaWw-v0',
    entry_point='gym_ww.envs:MaWw',
)


register(
    id='ComMaWw-v0',
    entry_point='gym_ww.envs:ComMaWw',
)

register(
    id='PolicyWw-v0',
    entry_point='gym_ww.envs:PolicyWw',
)





# Initialize envs loggers
logging.basicConfig(
    format='%(asctime)s EnvLogger - %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("WwEnvs")
logger.setLevel(logging.WARN)

# adding file handler
f_handler = logging.FileHandler(Params.log_match_file)
f_handler.setLevel(logging.DEBUG)
logger.addHandler(f_handler)


logger.debug("Logger initialized")
