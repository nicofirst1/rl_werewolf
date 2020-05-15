from envs.old.ComMaWw import ComMaWw
from envs.old.SimpleWW import SimpleWW
from gym_ww import ww,vil
from gym_ww.envs.PaEnv import PaEnv
from other.custom_utils import str_id_map
from utils import Params

CONFIGS = dict(

    existing_roles=[ww, vil],  # list of existing roles [werewolf, villanger]
    num_players= Params.num_player,
    penalties=dict(
        # penalty dictionary
        # penalty to give for each day that has passed
        day=0,
        # when a player dies
        death=-5,
        # victory
        victory=+25,
        # lost
        lost=-25,
        # penalty used for punishing votes that are not chosen during execution/kill.
        # If agent1 outputs [4,2,3,1,0] as a target list and agent2 get executed then agent1 get
        # a penalty equal to index_of(agent2,targets)*penalty
        trg_accord=-1,

    ),
    max_days=10,

    # signal is used in the communication phase to signal other agents about intentions
    # the length concerns the dimension of the signal while the components is the range of values it can fall into
    # a range value of 2 is equal to binary variable
    signal_length=3,
    signal_range=9,


)
#CONFIGS['role2id'], CONFIGS['id2role'] = str_id_map(CONFIGS['existing_roles'])

