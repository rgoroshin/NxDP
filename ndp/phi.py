from typing import Sequence
from functools import partial

import jax
from jax import numpy as jnp
from flax.core.frozen_dict import FrozenDict
from yacs.config import CfgNode

from ndp.dmp import DMP, StateDMP, ParamsDMP
from util.net import make_model


class PhiNet(object):

    def __init__(self, cfg: CfgNode, observation_size: int):

        self.n_dmp = cfg.DMP.N_DMP
        self.n_bfs = cfg.DMP.N_BFS
        output_size = self.n_dmp * self.n_bfs + self.n_dmp
        self._phi_net = make_model(
            cfg.OMEGA_NET.FEATURES + [output_size],
            observation_size,
        )


    def init(self, key):
        return self._phi_net.init(key)


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: FrozenDict, obs: jnp.ndarray) -> ParamsDMP:
        """Get the DMP parameters by taking observation as input
        .input:
            params: _phi_net parameters
            obs: (batch_size, observation_size)
        """
        dmp_params = self._phi_net.apply(params, obs)
        dmp_params = jnp.reshape(
            dmp_params,
            [dmp_params.shape[0], self.n_dmp, self.n_bfs + 1]
        )
        return ParamsDMP(
            w=dmp_params[:, :, :-1], # (batch_size, n_dmp, n_bfs)
            g=dmp_params[:, :,  -1], # (batch_size, n_dmp)
        )
