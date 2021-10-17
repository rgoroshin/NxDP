from typing import Any, Callable, Dict, Optional, Tuple
import functools
import time

from absl import logging
from brax import envs
from brax.training import distribution
from brax.training import env
from brax.training import normalization
from brax.training.types import PRNGKey, Params
import flax
import jax
import jax.numpy as jnp
import optax
from yacs.config import CfgNode

from ndp import NDP
from rl import get_rl_loss
from util import net


@flax.struct.dataclass
class StepData:
    """Contains data for one environment step."""
    obs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    truncation: jnp.ndarray
    actions: jnp.ndarray
    logits: jnp.ndarray


@flax.struct.dataclass
class TrainingState:
    """Contains training state for the learner."""
    optimizer_state: optax.OptState
    params: Params
    key: PRNGKey
    normalizer_params: Params


def train(
        cfg: CfgNode,
        environment_fn: Callable[..., envs.Env],
        progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
):
    """Training pipeline. Stolen from BRAX for the most part."""

    # CONFIG EXTRACTION
    num_timesteps = cfg.TRAIN.NUM_TIMESTEPS
    episode_length = cfg.TRAIN.EPISODE_LENGTH
    num_update_epochs = cfg.TRAIN.NUM_UPDATE_EPOCHS
    action_repeat = cfg.TRAIN.ACTION_REPEAT
    num_envs = cfg.TRAIN.NUM_ENVS
    max_devices_per_host = cfg.TRAIN.MAX_DEVICES_PER_HOST
    learning_rate = cfg.TRAIN.LEARNING_RATE
    unroll_length = cfg.TRAIN.UNROLL_LENGTH
    batch_size = cfg.TRAIN.BATCH_SIZE
    num_minibatches = cfg.TRAIN.NUM_MINIBATCHES
    normalize_observations = cfg.TRAIN.NORMALIZE_OBSERVATIONS
    log_frequency = cfg.LOG.FREQUENCY

    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    # KEY MANAGEMENT
    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
      local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        'Device count: %d, process count: %d (id %d), local device count: %d, '
        'devices to be used count: %d',
        jax.device_count(), process_count, process_id, local_device_count,
        local_devices_to_use)

    key = jax.random.PRNGKey(seed)
    key, key_models, key_env, key_eval = jax.random.split(key, 4)
    # Make sure every process gets a different random key, otherwise they will be
    # doing identical work.
    key_env = jax.random.split(key_env, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    # key_models should be the same, so that models are initialized the same way
    # for different processes.
    # key_eval is also used in one process so no need to split.

    # ENV SETTINGS
    core_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_envs // local_devices_to_use // process_count,
        episode_length=episode_length)
    key_envs = jax.random.split(key_env, local_devices_to_use)
    step_fn = core_env.step
    first_state = jax.tree_multimap(lambda *args: jnp.stack(args),
                                    *[core_env.reset(key) for key in key_envs])

    core_eval_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_eval_envs,
        episode_length=episode_length)
    eval_first_state, eval_step_fn = env.wrap_for_eval(core_eval_env, key_eval)

    # NETWORKS
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=core_env.action_size)

    policy_model = NDP(
        cfg,
        core_env.observation_size,
        core_env.action_size,
        core_env.sys.config.dt,
    )
    value_model = net.make_model(
        cfg.VALUE_NET.FEATURES + [1],
        core_env.observation_size,
    )

    key_policy, key_value = jax.random.split(key_models)
    phi_params, omega_params = policy_model.init(key_policy)
    value_params = value_model.init(key_value)
    init_params = {
        'phi': phi_params,
        'omega': omega_params,
        'value': value_params,
    }
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = normalization.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use)

    normalizer_params, obs_normalizer_update_fn, obs_normalizer_apply_fn = (
        normalization.create_observation_normalizer(
            core_env.observation_size, normalize_observations,
            num_leading_batch_dims=2, pmap_to_devices=local_devices_to_use))

    key_debug = jax.random.PRNGKey(seed + 666)

    # LOSS AND GRAD
    loss_fn = functools.partial(
        get_rl_loss(cfg.TRAIN.LOSS_FN),
        cfg=cfg,
        parametric_action_distribution=parametric_action_distribution,
        policy_apply=policy_model.apply,
        value_apply=value_model.apply,
    )
    grad_loss = jax.grad(loss_fn, has_aux=True)

    # def do_one_step_eval(carry, unused_target_t):
    #   state, policy_params, normalizer_params, key = carry
    #   key, key_sample = jax.random.split(key)
    #   # TODO: Make this nicer ([0] comes from pmapping).
    #   obs = obs_normalizer_apply_fn(
    #       jax.tree_map(lambda x: x[0], normalizer_params), state.core.obs)
    #   logits = policy_model.apply(policy_params, obs)
    #   actions = parametric_action_distribution.sample(logits, key_sample)
    #   nstate = eval_step_fn(state, actions)
    #   return (nstate, policy_params, normalizer_params, key), ()

    # @jax.jit
    # def run_eval(state, key, policy_params,
    #              normalizer_params) -> Tuple[env.EvalEnvState, PRNGKey]:
    #   policy_params = jax.tree_map(lambda x: x[0], policy_params)
    #   (state, _, _, key), _ = jax.lax.scan(
    #       do_one_step_eval, (state, policy_params, normalizer_params, key), (),
    #       length=episode_length // action_repeat)
    #   return state, key

    # def do_one_step(carry, unused_target_t):
    #   state, normalizer_params, policy_params, key = carry
    #   key, key_sample = jax.random.split(key)
    #   normalized_obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
    #   logits = policy_model.apply(policy_params, normalized_obs)
    #   actions = parametric_action_distribution.sample_no_postprocessing(
    #       logits, key_sample)
    #   postprocessed_actions = parametric_action_distribution.postprocess(actions)
    #   nstate = step_fn(state, postprocessed_actions)
    #   return (nstate, normalizer_params, policy_params, key), StepData(
    #       obs=state.obs,
    #       rewards=state.reward,
    #       dones=state.done,
    #       truncation=state.info['truncation'],
    #       actions=actions,
    #       logits=logits)

    # def generate_unroll(carry, unused_target_t):
    #   state, normalizer_params, policy_params, key = carry
    #   (state, _, _, key), data = jax.lax.scan(
    #       do_one_step, (state, normalizer_params, policy_params, key), (),
    #       length=unroll_length)
    #   data = data.replace(
    #       obs=jnp.concatenate(
    #           [data.obs, jnp.expand_dims(state.obs, axis=0)]),
    #       rewards=jnp.concatenate(
    #           [data.rewards, jnp.expand_dims(state.reward, axis=0)]),
    #       dones=jnp.concatenate(
    #           [data.dones, jnp.expand_dims(state.done, axis=0)]),
    #       truncation=jnp.concatenate(
    #           [data.truncation, jnp.expand_dims(state.info['truncation'],
    #                                             axis=0)]))
    #   return (state, normalizer_params, policy_params, key), data

    # def update_model(carry, data):
    #   optimizer_state, params, key = carry
    #   key, key_loss = jax.random.split(key)
    #   loss_grad, metrics = grad_loss(params, data, key_loss)
    #   loss_grad = jax.lax.pmean(loss_grad, axis_name='i')

    #   params_update, optimizer_state = optimizer.update(loss_grad,
    #                                                     optimizer_state)
    #   params = optax.apply_updates(params, params_update)

    #   return (optimizer_state, params, key), metrics

    # def minimize_epoch(carry, unused_t):
    #   optimizer_state, params, data, key = carry
    #   key, key_perm, key_grad = jax.random.split(key, 3)
    #   permutation = jax.random.permutation(key_perm, data.obs.shape[1])
    #   def convert_data(data, permutation):
    #     data = jnp.take(data, permutation, axis=1, mode='clip')
    #     data = jnp.reshape(
    #         data, [data.shape[0], num_minibatches, -1] + list(data.shape[2:]))
    #     data = jnp.swapaxes(data, 0, 1)
    #     return data
    #   ndata = jax.tree_map(lambda x: convert_data(x, permutation), data)
    #   (optimizer_state, params, _), metrics = jax.lax.scan(
    #       update_model, (optimizer_state, params, key_grad),
    #       ndata,
    #       length=num_minibatches)
    #   return (optimizer_state, params, data, key), metrics

    # def run_epoch(carry: Tuple[TrainingState, envs.State], unused_t):
    #   training_state, state = carry
    #   key_minimize, key_generate_unroll, new_key = jax.random.split(
    #       training_state.key, 3)
    #   (state, _, _, _), data = jax.lax.scan(
    #       generate_unroll, (state, training_state.normalizer_params,
    #                         training_state.params['policy'],
    #                         key_generate_unroll), (),
    #       length=batch_size * num_minibatches // num_envs)
    #   # make unroll first
    #   data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
    #   data = jax.tree_map(
    #       lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data)

    #   # Update normalization params and normalize observations.
    #   normalizer_params = obs_normalizer_update_fn(
    #       training_state.normalizer_params, data.obs[:-1])
    #   data = data.replace(
    #       obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

    #   (optimizer_state, params, _, _), metrics = jax.lax.scan(
    #       minimize_epoch, (training_state.optimizer_state, training_state.params,
    #                        data, key_minimize), (),
    #       length=num_update_epochs)

    #   new_training_state = TrainingState(
    #       optimizer_state=optimizer_state,
    #       params=params,
    #       normalizer_params=normalizer_params,
    #       key=new_key)
    #   return (new_training_state, state), metrics

    # num_epochs = num_timesteps // (
    #     batch_size * unroll_length * num_minibatches * action_repeat)

    # def _minimize_loop(training_state, state):
    #   (training_state, state), losses = jax.lax.scan(
    #       run_epoch, (training_state, state), (),
    #       length=num_epochs // log_frequency)
    #   losses = jax.tree_map(jnp.mean, losses)
    #   return (training_state, state), losses

    # minimize_loop = jax.pmap(_minimize_loop, axis_name='i')

    # training_state = TrainingState(
    #     optimizer_state=optimizer_state,
    #     params=init_params,
    #     key=jnp.stack(jax.random.split(key, local_devices_to_use)),
    #     normalizer_params=normalizer_params)
    # training_walltime = 0
    # eval_walltime = 0
    # sps = 0
    # eval_sps = 0
    # losses = {}
    # state = first_state
    # metrics = {}

    # for it in range(log_frequency + 1):
    #   logging.info('starting iteration %s %s', it, time.time() - xt)
    #   t = time.time()

    #   if process_id == 0:
    #     eval_state, key_debug = (
    #         run_eval(eval_first_state, key_debug,
    #                  training_state.params['policy'],
    #                  training_state.normalizer_params))
    #     eval_state.completed_episodes.block_until_ready()
    #     eval_walltime += time.time() - t
    #     eval_sps = (episode_length * eval_first_state.core.reward.shape[0] /
    #                 (time.time() - t))
    #     avg_episode_length = (
    #         eval_state.completed_episodes_steps / eval_state.completed_episodes)
    #     metrics = dict(
    #         dict({
    #             f'eval/episode_{name}': value / eval_state.completed_episodes
    #             for name, value in eval_state.completed_episodes_metrics.items()
    #         }),
    #         **dict({
    #             f'losses/{name}': jnp.mean(value)
    #             for name, value in losses.items()
    #         }),
    #         **dict({
    #             'eval/completed_episodes': eval_state.completed_episodes,
    #             'eval/avg_episode_length': avg_episode_length,
    #             'speed/sps': sps,
    #             'speed/eval_sps': eval_sps,
    #             'speed/training_walltime': training_walltime,
    #             'speed/eval_walltime': eval_walltime,
    #             'speed/timestamp': training_walltime,
    #         }))
    #     logging.info(metrics)
    #     if progress_fn:
    #       progress_fn(int(training_state.normalizer_params[0][0]) * action_repeat,
    #                   metrics)

    #   if it == log_frequency:
    #     break

    #   t = time.time()
    #   previous_step = training_state.normalizer_params[0][0]
    #   # optimization
    #   (training_state, state), losses = minimize_loop(training_state, state)
    #   jax.tree_map(lambda x: x.block_until_ready(), losses)
    #   sps = ((training_state.normalizer_params[0][0] - previous_step) /
    #          (time.time() - t)) * action_repeat
    #   training_walltime += time.time() - t

    # # To undo the pmap.
    # normalizer_params = jax.tree_map(lambda x: x[0],
    #                                  training_state.normalizer_params)
    # policy_params = jax.tree_map(lambda x: x[0],
    #                              training_state.params['policy'])

    # logging.info('total steps: %s', normalizer_params[0] * action_repeat)

    # _, inference = make_params_and_inference_fn(core_env.observation_size,
    #                                             core_env.action_size,
    #                                             normalize_observations)
    # params = normalizer_params, policy_params

    # if process_count > 1:
    #   # Make sure all processes stay up until the end of main.
    #   x = jnp.ones([jax.local_device_count()])
    #   x = jax.device_get(jax.pmap(lambda x: jax.lax.psum(x, 'i'), 'i')(x))
    #   assert x[0] == jax.device_count()

    # return (inference, params, metrics)


def make_params_and_inference_fn(
        observation_size,
        action_size,
        normalize_observations
):
    """Creates params and inference function for the PPO agent."""
    obs_normalizer_params, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
        observation_size, normalize_observations)
    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=action_size)
    policy_model, _ = networks.make_models(
        parametric_action_distribution.param_size,
        observation_size)

    def inference_fn(params, obs, key):
      normalizer_params, policy_params = params
      obs = obs_normalizer_apply_fn(normalizer_params, obs)
      action = parametric_action_distribution.sample(
          policy_model.apply(policy_params, obs), key)
      return action

    params = (obs_normalizer_params, policy_model.init(jax.random.PRNGKey(0)))
    return params, inference_fn
