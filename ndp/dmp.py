from typing import Tuple, Any
from functools import partial

import jax
from jax import numpy as jnp
from flax import struct
from yacs.config import CfgNode


@struct.dataclass
class ParamsDMP:
    w: jnp.ndarray
    g: jnp.ndarray

@struct.dataclass
class StateDMP:
    y: jnp.ndarray
    yd: jnp.ndarray
    x: jnp.ndarray


class DMP(object):

    def __init__(self, cfg: CfgNode):

        # TODO: feed env to DMP to extract the state dims etc

        self.n_dmps = cfg.DMP.N_DMP
        self.n_bfs = cfg.DMP.N_BFS
        self.dt = cfg.DMP.DT

        self.ax = cfg.DMP.AX
        self.ay = jnp.ones(self.n_dmps) * cfg.DMP.AY
        self.by = self.ay / 4.0 if cfg.DMP.BY is None else jnp.ones(self.n_dmps) * cfg.DMP.AY / cfg.DMP.BY

        self.unroll_length = cfg.DMP.UNROLL_LENGTH
        self.tau = cfg.DMP.TAU

        # generate centers
        self._gen_centers()


    def _gen_centers(self, ):
        # centers
        des_c = jnp.linspace(0, (1 + self.unroll_length) * self.dt, self.n_bfs)
        self.c = jax.tree_map(jnp.exp, -self.ax * des_c)
        # variance
        self.h = jnp.ones(self.n_bfs) * self.n_bfs ** 1.5 / self.c / self.ax


    def _get_psi(self, x: jnp.ndarray):
        # x = x[:, None]
        return jnp.exp(-self.h * (x - self.c) ** 2)


    @partial(jax.jit, static_argnums=(0,))
    def forcing_fn(
            self,
            x: float,
            y0: jnp.ndarray, # (batch_size, n_dmps)
            dmp_params: ParamsDMP,

    ):
        w = dmp_params.w # (batch_size, n_dmps, n_bfs)
        g = dmp_params.g # (batch_size, n_dmps)

        psi = self._get_psi(x) # (n_bfs, )

        return (x * (g - y0) * (w @ psi) / jnp.sum(psi))


    @partial(jax.jit, static_argnums=(0,))
    def do_one_cs_step(
            self,
            x: float,
    ):
        """Take one step in the canonical system
        """
        return x + (-self.ax * x) * self.tau * self.dt


    @partial(jax.jit, static_argnums=(0,))
    def do_one_dmp_step(self, carry, unused_t):
        y0, dmp_state, dmp_params = carry

        g = dmp_params.g

        y = dmp_state.y
        yd = dmp_state.yd
        x = dmp_state.x

        x = self.do_one_cs_step(x)
        f = self.forcing_fn(x, y0, dmp_params)

        ydd = self.ay * (self.by * (g - y) - yd) + f

        # Euler forward integration
        y = y + yd * self.tau * self.dt
        yd = yd + ydd * self.tau * self.dt

        return ((y0, StateDMP(y=y, yd=yd, x=x), dmp_params),
                StateDMP(y=y, yd=yd, x=x))


    @partial(jax.jit, static_argnums=(0,))
    def do_dmp_unroll(
            self,
            dmp_params: ParamsDMP,
            dmp_state: StateDMP,
    ) -> Tuple[StateDMP, Any]:

        y0 = dmp_state.y
        _, dmp_states = jax.lax.scan(
            self.do_one_dmp_step,
            (y0, dmp_state, dmp_params),
            (), self.unroll_length,
        )

        dmp_states = dmp_states.replace(
            y=jnp.concatenate([dmp_state.y[None, :], dmp_states.y]), # (unroll_length+1, batch_size, n_dmps)
            yd=jnp.concatenate([dmp_state.yd[None, :], dmp_states.yd]),
            x=jnp.hstack([jnp.array(dmp_state.x), dmp_states.x]), # (unroll_length+1, )
        )
        return dmp_states
