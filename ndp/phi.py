from typing import Sequence
from functools import partial

import jax
from jax import numpy as jnp
from brax.envs import env
from yacs.config import CfgNode

from ndp.dmp import DMP
from util.net import make_model
from util.types import *


class PhiNet(object):

    def __init__(self, cfg: CfgNode, observation_size: int):

        self.dmp_state_is_inferred = cfg.DMP.INFER_STATE
        self.n_dmp = cfg.DMP.N_DMP
        self.n_bfs = cfg.DMP.N_BFS
        output_size = (self.n_dmp + 1 + 2 * self.dmp_state_is_inferred) * self.n_bfs
        self._phi_net = make_model(
            cfg.PHI_NET.FEATURES + [output_size],
            observation_size,
        )


    def init(self, key: PRNGKey):
        return self._phi_net.init(key)


    @staticmethod
    def brax_state_to_dmp_state(state: env.State, x: float = .0) -> StateDMP:
        # TODO: implement this function i.e. handle qp.rot
        return StateDMP(
            y=jnp.stack(),
            yd=jnp.stack(),
            x=x,
        )


    @partial(jax.jit, static_argnums=(0,))
    def apply(self, params: Params, env_state: env.State) -> ParamsDMP:
        """Get the DMP parameters by taking observation as input
        .input:
            params: _phi_net parameters
            obs: (batch_size, observation_size)
        """
        dmp_params = self._phi_net.apply(params, env_state.obs)
        dmp_params = jnp.reshape(
            dmp_params,
            [dmp_params.shape[0], self.n_dmp, -1]
        )

        if self.dmp_state_is_inferred:
            return ParamsDMP(
                w=dmp_params[:, :, :-3], # (batch_size, n_dmp, n_bfs)
                g=dmp_params[:, :,  -3], # (batch_size, n_dmp)
                s=StateDMP(
                    y=dmp_params[:, :, -2],
                    yd=dmp_params[:, :, -1],
                    x=.0,
                ),
            )
        else:
            dmp_first_state = self.brax_state_to_dmp_state(env_state)
            return ParamsDMP(
                w=dmp_params[:, :, :-2], # (batch_size, n_dmp, n_bfs)
                g=dmp_params[:, :,  -2], # (batch_size, n_dmp)
                s=dmp_first_state,
            )
