from typing import Dict
from functools import partial

import jax
from jax import numpy as jnp
from brax.envs import env
from util.types import PRNGKey, Params
from yacs.config import CfgNode

from ndp.dmp import DMP
from ndp.phi import PhiNet
from ndp.omega import OmegaNet
from util.types import *


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


    def init(self, key: PRNGKey) -> Dict[str, Params]:
        key_phi, key_omega = jax.random.split(key, 2)
        return {
            'phi': self.phi_net.init(key_phi),
            'omega': self.omega_net.init(key_omega),
        }


    @partial(jax.jit, static_argnums=(0,))
    def apply(
            self,
            params: Dict[str, Params],
            env_state: env.State,
    ) -> jnp.ndarray:
        """ Takes the parameters of phi and omega, + a batch of
            observations as input, spits out the next
            `unroll_length` actions.
        """
        phi_params, omega_params = params['phi'], params['omega']
        dmp_params = self.phi_net.apply(phi_params, env_state)
        dmp_states = self.dmp.do_dmp_unroll(dmp_params)
        actions = self.omega_net.apply(omega_params, dmp_states)
        return actions
