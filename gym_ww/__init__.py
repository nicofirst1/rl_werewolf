from gym.envs.registration import register
import logging

register(
    id='simpleWW-v0',
    entry_point='gym_ww.envs:SimpleWW',
)




logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

logger.debug("Logger initialized")