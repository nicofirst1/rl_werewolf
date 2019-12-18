from gym.envs.registration import register
import logging

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
    entry_point='gym_ww.envs:MaWw',
)






logging.basicConfig(
    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger()
logger.setLevel(logging.WARN)

logger.debug("Logger initialized")