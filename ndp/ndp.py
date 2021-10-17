from typing import Tuple

import jax
from flax.core.frozen_dict import FrozenDict
from yacs.config import CfgNode

from ndp.dmp import DMP, StateDMP, ParamsDMP
from ndp.omega import OmegaNet

Param = Any


class NDP(object):

    def __init__(
            self,
            cfg: CfgNode,
            observation_size: int,
            action_size: int,
    ):

        self.dmp = DMP(cfg)
        self.phi_net = PhiNet(cfg, observation_size)
        self.omega_net = OmegaNet(cfg, action_size)


    def init(self, key) -> Tuple[FrozenDict, FrozenDict]:
        key_phi, key_omega = jax.random.split(key, 2)
        return self.phi_net.init(key_phi), self.omega_net.init(key_omega)



    def apply(
            self,
            params: Tuple[Param, Param],
            obs: jnp.ndarray,
            dmp_first_state: StateDMP,
    ):
        """ Takes the parameters of phi and omega, + a batch of
            observations as input, spits out the next
            `unroll_length` actions.
        """
        phi_params, omega_params = params
        dmp_params = self.phi_net.apply(phi_params, obs)
        # TODO: first dmp state should also be outputted by phi? necessary if
        # we're planning in a latent space. The latent space should be inferred
        # from the observations.
        dmp_states = self.dmp.do_dmp_unroll(dmp_params, dmp_states)
        actions = self.omega_net.apply(omega_params, dmp_states)
        return actions
