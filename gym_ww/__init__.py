from gym.envs.registration import register
import logging


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

logger.debug("Logger initialized")
