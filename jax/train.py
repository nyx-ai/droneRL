import logging
import jax
import jax.numpy as jnp
import jax.lax
import jax.random
from tqdm import trange
from jax.experimental.compilation_cache import compilation_cache as cc
from timeit import default_timer as timer
import statistics

from env.env import DroneEnvParams, DeliveryDrones
from env.constants import Action
from agents.dqn import DQNAgent, DQNAgentParams
from buffers import ReplayBuffer

from render_util import render_video


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')

def stats(mylist, factor: float = 1000.0):
    return factor * statistics.mean(mylist), factor * statistics.stdev(mylist)

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

    # Florian's benchmark
    # n_drones = 32
    # grid_size = 26
    # n_drones = 512
    # grid_size = 102
    # n_drones = 2048
    # grid_size = 203

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
    @jax.jit
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
            return trained_state, loss

        ag_state, loss = jax.lax.cond(
            buffer.can_sample(bstate),
            train_if_can_sample,
            lambda x: (x[0], 0.0),
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

        return (rng, state, next_obs, ag_state, bstate), loss

    num_repeats = 1
    skip_first = 1
    num_steps = 400
    num_steps_scan = 100
    num_batches = num_steps // num_steps_scan
    timings = []
    for _ in trange(num_repeats):
        for batch in trange(num_batches, unit='batch'):
            ts = timer()
            init_carry = (rng, state, obs, ag_state, bstate)
            final_carry, loss = jax.lax.scan(_train, init_carry, jnp.arange(num_steps_scan))
            rng, state, _, ag_state, _ = final_carry
            rng.block_until_ready()
            timings.append(timer() - ts)
    step_mean, step_stdev = stats(timings[skip_first:], factor=1000.0/num_steps_scan)
    logger.info(f'Scan training took {step_mean:.3f} ± {step_stdev:>6.3f} s/k steps')

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

    # render_video(test_env_params, ag_state)
    render_video(test_env_params, ag_state, temp_dir='output', num_steps=200)


if __name__ == "__main__":
    train()
