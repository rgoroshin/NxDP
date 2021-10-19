from datetime import datetime
import argparse

from config.defaults import get_cfg_defaults
from brax import envs
import matplotlib.pyplot as plt
from IPython.display import clear_output

from train import train


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

    env_fn = envs.create_fn(cfg.ENV.ENV_NAME)

    xdata = []
    ydata = []
    times = [datetime.now()]


    def progress(num_steps, metrics):
        times.append(datetime.now())
        xdata.append(num_steps)
        ydata.append(metrics['eval/episode_reward'])
        clear_output(wait=True)
        plt.xlim([0, cfg.TRAIN.NUM_TIMESTEPS])
        plt.ylim([0, 6000])
        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.plot(xdata, ydata)
        plt.show()

    train(cfg, env_fn, progress_fn=progress)


if __name__ == '__main__':
    main()
