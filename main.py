import jax
from config.defaults import get_cfg_defaults
from ndp.dmp import *

import argparse


def main():
    parser = argparse.ArgumentParser(description="HJxB")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="file",
        help="path to yaml config file",
        type=str,
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line",
        default=[],
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # build the config
    cfg = get_cfg_defaults()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    dmp = DMP(cfg)
    dmp_params = ParamsDMP(g=jnp.ones([16, 10]), w=jnp.ones([16, 10, 10]))
    dmp_state = StateDMP(y=jnp.ones([16, 10]), yd=jnp.ones([16, 10]), x=1.0)

    key = jax.random.PRNGKey(0)
    dmp_states, _ = dmp.do_dmp_unroll(dmp_state, dmp_params, key)
    print(dmp_states)
    print(dmp_states.y.shape)
    print(dmp_states.x.shape)
    print(dmp_states.x)
    print(dmp_states.y[:, 0, 1])
    print(dmp_states.yd[:, 0, 1])


if __name__ == '__main__':
    main()
