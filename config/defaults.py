import os
from yacs.config import CfgNode as CN

# ---------------------------------------------------------------------------- #
# Define Config Node
# ---------------------------------------------------------------------------- #
_C = CN()

# ---------------------------------------------------------------------------- #
# DMP
# ---------------------------------------------------------------------------- #
_C.DMP = CN()
_C.DMP.N_DMP = 10
_C.DMP.N_BFS = 10
_C.DMP.UNROLL_LENGTH = 5
_C.DMP.INFER_STATE = True
# TODO: these maybe should be outputted by $\Phi$ net
_C.DMP.AX = 1.0     # the canonical system's a_x
_C.DMP.AY = 25.0    # damping
_C.DMP.BY = None    # spring const.
_C.DMP.TAU = 1.0    # temporal scaling

# ---------------------------------------------------------------------------- #
# ENVIRONMENT
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.ENV_NAME = "inverted_pendulum"
_C.ENV.TIMESTEP = 0.05
_C.ENV.ACTION_REPEAT = 1

# ---------------------------------------------------------------------------- #
# OMEGA NET
# ---------------------------------------------------------------------------- #
_C.OMEGA_NET = CN()
_C.OMEGA_NET.FEATURES = [32, 32, 32, 32]
_C.OMEGA_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# PHI NET
# ---------------------------------------------------------------------------- #
_C.PHI_NET = CN()
_C.PHI_NET.FEATURES = [64, 32, 32, 32]
_C.PHI_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# VALUE NET
# ---------------------------------------------------------------------------- #
_C.VALUE_NET = CN()
_C.VALUE_NET.FEATURES = [256, 256, 256, 256, 256]
_C.VALUE_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()
_C.TRAIN.LOSS_FN = "ppo_loss"
_C.TRAIN.NUM_TIMESTEPS = 6000000
_C.TRAIN.EPISODE_LENGTH = 1000
_C.TRAIN.NUM_UPDATE_EPOCHS = 4
_C.TRAIN.ACTION_REPEAT = 1
_C.TRAIN.NUM_ENVS = 16 # 2048
_C.TRAIN.MAX_DEVICES_PER_HOST = 8
_C.TRAIN.LEARNING_RATE = 3e-4
_C.TRAIN.UNROLL_LENGTH = 10
_C.TRAIN.BATCH_SIZE = 1024
_C.TRAIN.NUM_MINIBATCHES = 32
_C.TRAIN.NORMALIZE_OBSERVATIONS = True
_C.TRAIN.DISCOUNTING = 0.97
_C.TRAIN.REWARD_SCALING = 10.0
_C.TRAIN.NUM_EVAL_ENVS = 10
    # ------------------------------------------------------------------------ #
    # PPO
    # ------------------------------------------------------------------------ #
_C.TRAIN.PPO = CN()
_C.TRAIN.PPO.ENTROPY_COST = 1e-2
_C.TRAIN.PPO.GAE_LAMBDA = 0.95
_C.TRAIN.PPO.EPSILON = 0.3

# ---------------------------------------------------------------------------- #
# LOGGING
# ---------------------------------------------------------------------------- #
_C.LOG = CN()
_C.LOG.FREQUENCY = 20


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
