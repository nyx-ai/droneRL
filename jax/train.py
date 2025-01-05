import logging
import jax
import jax.numpy as jnp
import jax.lax
import jax.random
from tqdm import trange
from jax.experimental.compilation_cache import compilation_cache as cc
from timeit import default_timer as timer
from collections import defaultdict
import statistics

from env.env import DroneEnvParams, DeliveryDrones
from env.constants import Action
from agents.dqn import DQNAgent, DQNAgentParams
from buffers import ReplayBuffer

from render_util import render_video


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')


def train():
    # params
    batch_size = 64
    memory_size = 10_000
    hidden_layers = (32, 32)
    # n_drones = 3
    # grid_size = 10
    n_drones = 1024
    grid_size = 185
    # n_drones = 4096
    # grid_size = 370

    env_params = DroneEnvParams(n_drones=n_drones, grid_size=grid_size)
    ag_params = DQNAgentParams(hidden_layers=hidden_layers)
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)

    # init env
    state = env.reset(rng, env_params)
    obs = env.get_obs(state, env_params)
    obs = obs[0].ravel()
    actions = jax.random.randint(rng, (env_params.n_drones,), minval=0, maxval=Action.num_actions())
    state, rewards, dones = env.step(rng, state, actions, env_params)
    next_obs = env.get_obs(state, env_params)
    next_obs = next_obs[0].ravel()

    # init agent
    dqn_agent = DQNAgent()
    ag_state = dqn_agent.reset(rng, ag_params, env_params)

    # init buffer
    buffer = ReplayBuffer(buffer_size=memory_size, sample_batch_size=batch_size)
    exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
    bstate = buffer.init(exp)

    ##################
    # JAX scan loop
    ##################
    def _train(carry, step):
        rng, state, obs, ag_state, bstate = carry

        rng, key = jax.random.split(rng)
        # generate random actions for all drones
        actions = jax.random.randint(key, (env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent
        dqn_action = dqn_agent.act(key, obs, ag_state)
        actions = actions.at[0].set(dqn_action)

        # perform actions in env
        state, rewards, dones = env.step(key, state, actions, env_params)

        next_obs = env.get_obs(state, env_params)
        next_obs = next_obs[0].ravel()

        # add to buffer
        exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
        bstate = buffer.add(bstate, exp)

        # train step
        def train_if_can_sample(args):
            ag_state, bstate, key = args
            batch, key = buffer.sample(key, bstate)
            trained_state, loss = dqn_agent.train_step(
                ag_state,
                batch['obs'],
                batch['actions'],
                batch['rewards'],
                batch['next_obs'],
                batch['dones'],
                ag_params)
            return trained_state

        ag_state = jax.lax.cond(
            buffer.can_sample(bstate),
            train_if_can_sample,
            lambda x: x[0],
            (ag_state, bstate, key)
        )

        # update target network
        ag_state = jax.lax.cond(
            step % ag_params.target_update_interval == 0,
            lambda x: dqn_agent.update_target(x, ag_params),
            lambda x: x,
            ag_state
        )

        # update epsilon
        ag_state = jax.lax.cond(
            dones[0],
            lambda x: dqn_agent.update_epsilon(x, ag_params),
            lambda x: x,
            ag_state
        )

        return (rng, state, next_obs, ag_state, bstate), None

    num_steps = 100_000
    num_steps_scan = 1000
    num_batches = num_steps // num_steps_scan
    ts = timer()
    for batch in trange(num_batches, unit='batch'):
        init_carry = (rng, state, obs, ag_state, bstate)
        final_carry, _ = jax.lax.scan(_train, init_carry, jnp.arange(num_steps_scan))
        rng, state, _, ag_state, _ = final_carry
    rng.block_until_ready()
    logger.info(f'... training {num_steps:,} steps took {timer()-ts:.3f}s...')

    def _eval(carry, step):
        rng, state, ag_state = carry

        # get obs
        obs = env.get_obs(state, test_env_params)
        obs = obs[0].ravel()

        # generate random actions for all drones
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (test_env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent
        dqn_action = dqn_agent.act(key, obs, ag_state, greedy=True)
        actions = actions.at[0].set(dqn_action)

        # perform actions in env
        state, rewards, dones = env.step(key, state, actions, test_env_params)
        return (rng, state, ag_state), rewards

    test_env_params = DroneEnvParams(n_drones=3, grid_size=10)
    num_test_steps = 10_000
    test_state = env.reset(rng, test_env_params)
    ts = timer()
    final_carry, rewards = jax.lax.scan(_eval, (rng, test_state, ag_state), jnp.arange(num_test_steps))
    rng, _, _ = final_carry
    mean_reward = jnp.mean(rewards, axis=0)
    logger.info(f'... eval of {num_test_steps:,} steps took {timer()-ts:.3f}s. Mean reward: {mean_reward}')

    render_video(test_env_params, ag_state)
    __import__('pdb').set_trace()


    ##################
    # Python for-looop
    ##################
    # jit functions
    # do_jit = False
    do_jit = True
    train_step_jit = jax.jit(dqn_agent.train_step, static_argnums=(6,)) if do_jit else dqn_agent.train_step
    step_jit = jax.jit(env.step, static_argnums=(3,)) if do_jit else env.step
    act_jit = jax.jit(dqn_agent.act) if do_jit else dqn_agent.act
    get_obs_jit = jax.jit(env.get_obs, static_argnums=(1,)) if do_jit else env.get_obs
    buffer_add_jit = jax.jit(buffer.add) if do_jit else buffer.add
    buffer_sample_jit = jax.jit(buffer.sample) if do_jit else buffer.sample
    update_target_jit = jax.jit(dqn_agent.update_target, static_argnums=(1,)) if do_jit else dqn_agent.update_target
    update_eps_jit = jax.jit(dqn_agent.update_epsilon, static_argnums=(1,)) if do_jit else dqn_agent.update_epsilon
    timing = defaultdict(list)
    step_timing = []
    do_timing = True
    skip_timing = 100

    for step in trange(num_steps):
        ts_all = timer()
        rng, key = jax.random.split(rng)

        # generate random actions for all drones
        actions = jax.random.randint(key, (env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent
        ts = timer()
        dqn_action = act_jit(key, obs, ag_state)
        if do_timing and step > skip_timing:
            timing['agent_act'].append(timer() - ts)
        actions = actions.at[0].set(dqn_action)

        # perform actions in env
        ts = timer()
        state, rewards, dones = step_jit(key, state, actions, env_params)
        if do_timing and step > skip_timing:
            timing['env_step'].append(timer() - ts)

        ts = timer()
        next_obs = get_obs_jit(state, env_params)
        if do_timing and step > skip_timing:
            timing['get_obs'].append(timer() - ts)
        next_obs = next_obs[0].ravel()

        # add to buffer
        exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
        ts = timer()
        bstate = buffer_add_jit(bstate, exp)
        if do_timing and step > skip_timing:
            timing['buffer_add'].append(timer() - ts)

        # train step
        if buffer.can_sample(bstate):
            ts = timer()
            batch, key = buffer_sample_jit(key, bstate)
            if do_timing and step > skip_timing:
                timing['buffer_sample'].append(timer() - ts)
            ts = timer()
            ag_state, loss = train_step_jit(
                    ag_state,
                    batch['obs'],
                    batch['actions'],
                    batch['rewards'],
                    batch['next_obs'],
                    batch['dones'],
                    ag_params)
            if do_timing and step > skip_timing:
                timing['train_step'].append(timer() - ts)

        # update target network
        if step % ag_params.target_update_interval == 0:
            ts = timer()
            ag_state = update_target_jit(ag_state, ag_params)
            if do_timing and step > skip_timing:
                timing['update_network'].append(timer() - ts)

        # update eps
        if dones[0]:
            ts = timer()
            ag_state = update_eps_jit(ag_state, ag_params)
            if do_timing and step > skip_timing:
                timing['update_eps'].append(timer() - ts)

        obs = next_obs
        if do_timing and step > skip_timing:
            step_timing.append(timer() - ts_all)

    def _stat(mylist, factor=1000):
        return factor * statistics.mean(mylist), factor * statistics.stdev(mylist)
    step_mean, step_stdev = _stat(step_timing)
    stats = {k: _stat(v) for k, v in timing.items()}
    max_key_length = max(len(k) for k in stats.keys())
    spacing = 9 * ' '
    stats_str = f'\nFull step{spacing} {step_mean:.3f} ± {step_stdev:>6.3f} s/k steps\n'
    for k, (mean, std) in stats.items():
        stats_str += f'  - {k:<{max_key_length}} {mean:.3f} ± {std:>6.3f} s/k steps ({100*mean/step_mean:>5.2f}%)\n'
    logger.info(stats_str)
    __import__('pdb').set_trace()


if __name__ == "__main__":
    train()
