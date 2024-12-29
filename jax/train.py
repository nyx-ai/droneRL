import logging
import jax
import jax.numpy as jnp
import jax.random
from tqdm import trange
from jax.experimental.compilation_cache import compilation_cache as cc
from timeit import default_timer as timer

from env.env import DroneEnvParams, DeliveryDrones
from env.constants import Action
from agents.dqn import DQNAgent, DQNAgentParams
from buffers import ReplayBuffer


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')


def train():
    # params
    num_steps = 1000
    batch_size = 64
    memory_size = 10_000
    hidden_layers = (32, 32)
    n_drones = 3
    grid_size = 8

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
    buffer = ReplayBuffer()
    exp = {'obs': obs, 'actions': actions[None, 0], 'rewards': rewards[None, 0], 'next_obs': next_obs, 'dones': dones[None, 0]}

    bstate = buffer.init(memory_size, exp)

    # jit functions
    do_jit = True
    train_step_jit = jax.jit(dqn_agent.train_step, static_argnums=(6,)) if do_jit else dqn_agent.train_step
    step_jit = jax.jit(env.step, static_argnums=(3,)) if do_jit else env.step
    act_jit = jax.jit(dqn_agent.act) if do_jit else dqn_agent.act
    get_obs_jit = jax.jit(env.get_obs, static_argnums=(1,)) if do_jit else env.get_obs
    buffer_add_jit = jax.jit(buffer.add) if do_jit else buffer.add
    buffer_sample_jit = jax.jit(buffer.sample, static_argnums=(1,)) if do_jit else buffer.sample


    for step in trange(num_steps):
        ts_all = timer()
        rng, key = jax.random.split(rng)

        # generate random actions for all drones
        actions = jax.random.randint(key, (env_params.n_drones,), minval=0, maxval=Action.num_actions())

        ts = timer()
        # run action for DQN agent
        dqn_action = act_jit(key, obs, ag_state)
        actions = actions.at[0].set(dqn_action)
        logger.info(f'{1000*(timer()-ts):.2f}ms agent act')

        ts = timer()
        # perform actions in env
        state, rewards, dones = step_jit(key, state, actions, env_params)
        logger.info(f'{1000*(timer()-ts):.2f}ms env step')

        ts = timer()
        next_obs = get_obs_jit(state, env_params)
        next_obs = next_obs[0].ravel()
        logger.info(f'{1000*(timer()-ts):.2f}ms env get obs')

        # add to buffer
        ts = timer()
        exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
        bstate = buffer_add_jit(bstate, exp)
        logger.info(f'{1000*(timer()-ts):.2f}ms buffer add')

        # train step
        if bstate.current_size >= batch_size:
            ts = timer()
            batch, key = buffer_sample_jit(key, batch_size, bstate)
            logger.info(f'{1000*(timer()-ts):.2f}ms buffer sample')
            ts = timer()
            ag_state, loss = train_step_jit(
                    ag_state,
                    batch['obs'],
                    batch['actions'],
                    batch['rewards'],
                    batch['next_obs'],
                    batch['dones'],
                    ag_params)
            logger.info(f'{1000*(timer()-ts):.2f}ms train step')

        ts = timer()
        # update target network
        if step % ag_params.target_update_interval == 0:
            ag_state = dqn_agent.update_target(ag_state, ag_params)
        logger.info(f'{1000*(timer()-ts):.2f}ms update network')

        # update eps
        ts = timer()
        if dones[0]:
            ag_state = dqn_agent.update_epsilon(ag_state, ag_params)
        logger.info(f'{1000*(timer()-ts):.2f}ms update eps')

        obs = next_obs
        logger.info(f'... full step {1000*(timer()-ts_all):.1f}s/1k steps')


if __name__ == "__main__":
    train()
