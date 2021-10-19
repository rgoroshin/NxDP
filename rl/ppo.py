from typing import Any, Dict, Callable

from brax.training import distribution
from util.types import PRNGKey, Params
import flax
import jax
import jax.numpy as jnp
import optax
from yacs.config import CfgNode

from util.types import *


def compute_gae(truncation: jnp.ndarray,
                termination: jnp.ndarray,
                rewards: jnp.ndarray,
                values: jnp.ndarray,
                bootstrap_value: jnp.ndarray,
                lambda_: float = 1.0,
                discount: float = 0.99):
    r"""Calculates the Generalized Advantage Estimation (GAE).

    Args:
      truncation: A float32 tensor of shape [T, B] with truncation signal.
      termination: A float32 tensor of shape [T, B] with termination signal.
      rewards: A float32 tensor of shape [T, B] containing rewards generated by
        following the behaviour policy.
      values: A float32 tensor of shape [T, B] with the value function estimates
        wrt. the target policy.
      bootstrap_value: A float32 of shape [B] with the value function estimate at
        time T.
      lambda_: Mix between 1-step (lambda_=0) and n-step (lambda_=1). Defaults to
        lambda_=1.
      discount: TD discount.

    Returns:
      A float32 tensor of shape [T, B]. Can be used as target to
        train a baseline (V(x_t) - vs_t)^2.
      A float32 tensor of shape [T, B] of advantages.
    """

    truncation_mask = 1 - truncation
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = jnp.concatenate(
        [values[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    deltas = rewards + discount * (1 - termination) * values_t_plus_1 - values
    deltas *= truncation_mask

    acc = jnp.zeros_like(bootstrap_value)
    vs_minus_v_xs = []

    def compute_vs_minus_v_xs(carry, target_t):
      lambda_, acc = carry
      truncation_mask, delta, termination = target_t
      acc = delta + discount * (1 - termination) * truncation_mask * lambda_ * acc
      return (lambda_, acc), (acc)

    (_, _), (vs_minus_v_xs) = jax.lax.scan(compute_vs_minus_v_xs, (lambda_, acc),
                                           (truncation_mask, deltas, termination),
                                           length=int(truncation_mask.shape[0]),
                                           reverse=True)
    # Add V(x_s) to get v_s.
    vs = jnp.add(vs_minus_v_xs, values)

    vs_t_plus_1 = jnp.concatenate(
        [vs[1:], jnp.expand_dims(bootstrap_value, 0)], axis=0)
    advantages = (rewards + discount *
                  (1 - termination) * vs_t_plus_1 - values) * truncation_mask
    return jax.lax.stop_gradient(vs), jax.lax.stop_gradient(advantages)


def _get_policy_logits(
        policy_apply: Callable[[Params, jnp.ndarray], jnp.ndarray],
        policy_params: Params,
        observations: jnp.ndarray,
        dmp_unroll_length: int,
):
    def policy_apply_with_carry(carry, obs):
        return carry, policy_apply(policy_params, obs)
    _, logits = jax.lax.scan(policy_apply_with_carry, None, observations[::dmp_unroll_length])
    return logits.reshape([-1] + list(logits.shape[2:]))


def ppo_loss(
        models: Dict[str, Params],
        data: StepData,
        rng: jnp.ndarray,
        parametric_action_distribution: distribution.ParametricDistribution,
        policy_apply: Any,
        value_apply: Any,
        cfg: CfgNode(),
):
    """Computes PPO loss."""

    reward_scaling = cfg.TRAIN.REWARD_SCALING
    discounting = cfg.TRAIN.DISCOUNTING
    entropy_cost = cfg.TRAIN.PPO.ENTROPY_COST
    lambda_ = cfg.TRAIN.PPO.GAE_LAMBDA
    ppo_epsilon = cfg.TRAIN.PPO.EPSILON

    policy_params, value_params = models['policy'], models['value']
    # policy_logits = policy_apply(policy_params, data.obs[:-1])
    policy_logits = _get_policy_logits(policy_apply, policy_params, data.obs[:-1], cfg.DMP.UNROLL_LENGTH)
    baseline = value_apply(value_params, data.obs)
    baseline = jnp.squeeze(baseline, axis=-1)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]

    # At this point, we have unroll length + 1 steps. The last step is only used
    # as bootstrap value, so it's removed.

    # already removed at data generation time
    # actions = actions[:-1]
    # logits = logits[:-1]

    rewards = data.rewards[1:] * reward_scaling
    truncation = data.truncation[1:]
    termination = data.dones[1:] * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.actions)
    behaviour_action_log_probs = parametric_action_distribution.log_prob(
        data.logits, data.actions)

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=lambda_,
        discount=discounting)
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)

    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon,
                               1 + ppo_epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(
        parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    return policy_loss + v_loss + entropy_loss, {
        'total_loss': policy_loss + v_loss + entropy_loss,
        'policy_loss': policy_loss,
        'v_loss': v_loss,
        'entropy_loss': entropy_loss
    }
