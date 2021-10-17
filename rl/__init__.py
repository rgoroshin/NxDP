from .ppo import ppo_loss

__all__ = [
    "ppo_loss",
]

def get_rl_loss(loss_fn: str):
    return globals()[loss_fn]
