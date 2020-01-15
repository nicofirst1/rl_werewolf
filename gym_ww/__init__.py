from gym.envs.registration import register
import logging
from utils import Params

ww = "werewolf"
vil = "villager"



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

logger = logging.getLogger("WwEnvs")
logger.setLevel(logging.DEBUG)

# adding file handler
f_formatter = logging.Formatter('%(asctime)s - %(message)s')
f_handler = logging.FileHandler(Params.log_match_file)
f_handler.setLevel(logging.DEBUG)
f_handler.setFormatter(f_formatter)

# adding stream handler
c_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
c_handler = logging.StreamHandler()
c_handler.setFormatter(c_formatter)
c_handler.setLevel(logging.WARN)


# addign handlers to main logger
logger.addHandler(c_handler)
logger.addHandler(f_handler)

logger.debug("Logger initialized")
