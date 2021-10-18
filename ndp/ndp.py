from typing import Tuple

import jax
from jax import numpy as jnp
from brax.envs import env
from brax.training.types import PRNGKey, Param
from yacs.config import CfgNode

from ndp.dmp import DMP, StateDMP, ParamsDMP
from ndp.omega import OmegaNet


class NDP(object):

    def __init__(
            self,
            cfg: CfgNode,
            observation_size: int,
            action_size: int,
            timestep: float,
    ):
        self.dmp = DMP(cfg, timestep)
        self.phi_net = PhiNet(cfg, observation_size)
        self.omega_net = OmegaNet(cfg, action_size)


    def init(self, key: PRNGKey) -> Tuple[Param, Param]:
        key_phi, key_omega = jax.random.split(key, 2)
        return self.phi_net.init(key_phi), self.omega_net.init(key_omega)


    def apply(
            self,
            params: Tuple[Param, Param],
            env_state: env.State,
    ) -> jnp.ndarray:
        """ Takes the parameters of phi and omega, + a batch of
            observations as input, spits out the next
            `unroll_length` actions.
        """
        phi_params, omega_params = params
        dmp_params = self.phi_net.apply(phi_params, env_state)
        dmp_states = self.dmp.do_dmp_unroll(dmp_params)
        actions = self.omega_net.apply(omega_params, dmp_states)
        return actions
