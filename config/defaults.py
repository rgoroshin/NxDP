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
_C.DMP.DT = 0.01
_C.DMP.UNROLL_LENGTH = 20
# TODO: these maybe should be outputted by $\Phi$ net
_C.DMP.AX = 1.0     # the canonical system's a_x
_C.DMP.AY = 25.0    # damping
_C.DMP.BY = None    # spring const.
_C.DMP.TAU = 1.0    # temporal scaling

# ---------------------------------------------------------------------------- #
# ENVIRONMENT
# ---------------------------------------------------------------------------- #
_C.ENV = CN()
_C.ENV.ENV_NAME = "ant"
_C.ENV.TIMESTEP = 0.05

# ---------------------------------------------------------------------------- #
# VALUE NET
# ---------------------------------------------------------------------------- #
_C.VALUE_NET = CN()
_C.VALUE_NET.FEATURES = [32, 16, 8]
_C.VALUE_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# OMEGA NET
# ---------------------------------------------------------------------------- #
_C.OMEGA_NET = CN()
_C.OMEGA_NET.FEATURES = [32, 16, 8]
_C.OMEGA_NET.LR = 1e-3

# ---------------------------------------------------------------------------- #
# TRAIN
# ---------------------------------------------------------------------------- #
_C.TRAIN = CN()


# ---------------------------------------------------------------------------- #
# LOGGING
# ---------------------------------------------------------------------------- #
_C.LOG = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()
