import logging
import math
import jax
import jax.numpy as jnp
import jax.lax
import jax.random
from tqdm import trange
from jax.experimental.compilation_cache import compilation_cache as cc
from timeit import default_timer as timer
import wandb


from env.env import DroneEnvParams, DeliveryDrones
from ..common.constants import Action
from agents.dqn import DQNAgent, DQNAgentParams
from buffers import ReplayBuffer


wandb.login()

SWEEP_NAME = "dronerl-dense-1-jax"
NUM_TRAINING_STEPS = 1_000
NUM_TESTING_STEPS = 1_000

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')


def evaluate(config):
    drone_density = 0.03
    n_drones = 1024
    grid_size = int(math.ceil(math.sqrt(n_drones / drone_density)))
    env_params = DroneEnvParams(n_drones=n_drones, grid_size=grid_size)
    logger.info(f'Running env of {grid_size:,}x{grid_size:,} and {env_params.n_drones:,} drones')
    hidden_layers = config.num_layers * (config.size_layers,)
    ag_params = DQNAgentParams(
            hidden_layers=hidden_layers,
            gamma=config.gamma,
            epsilon_decay=config.epsilon_decay,
            learning_rate=config.learning_rate,
            target_update_interval=config.target_update_interval,
            )
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
    buffer = ReplayBuffer(
            buffer_size=config.memory_size,
            sample_batch_size=config.batch_size)
    exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
    bstate = buffer.init(exp)

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
            batch = buffer.sample(key, bstate)
            trained_state, loss = dqn_agent.train_step(ag_state, batch, ag_params)
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

    logger.info('Start training...')
    num_steps = NUM_TRAINING_STEPS
    num_steps_scan = 1000
    num_batches = num_steps // num_steps_scan
    ts = timer()
    for _ in trange(num_batches, unit='batch'):
        init_carry = (rng, state, obs, ag_state, bstate)
        final_carry, loss = jax.lax.scan(_train, init_carry, jnp.arange(num_steps_scan))
        rng, state, _, ag_state, _ = final_carry
    logger.info(f'... training {num_steps:,} steps took {(timer()-ts):.2f}s')

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

    logger.info('Starting eval...')
    test_env_params = DroneEnvParams(n_drones=3, grid_size=10)
    test_state = env.reset(rng, test_env_params)
    ts = timer()
    num_steps = NUM_TESTING_STEPS
    num_steps_scan = 1000
    num_batches = num_steps // num_steps_scan
    ts = timer()
    carry = (rng, test_state, ag_state)
    all_rewards = []
    for _ in trange(num_batches, unit='batch'):
        carry, rewards = jax.lax.scan(_eval, carry, jnp.arange(num_steps_scan))
        all_rewards.append(rewards[:, 0])  # reward of drone #1
    mean_reward = jnp.mean(jnp.hstack(all_rewards), axis=0).item()
    sum_reward = jnp.sum(jnp.hstack(all_rewards), axis=0).item()
    logger.info(f'... eval of {num_steps:,} steps took {timer()-ts:.3f}s. Mean reward: {mean_reward:.3f}, sum reward: {sum_reward:.3f}')
    return mean_reward, sum_reward


def main():
    wandb.init(project=SWEEP_NAME)
    scores = evaluate(wandb.config)
    wandb.log({"mean_reward": scores[0], 'cum_reward': scores[1]})


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "mean_reward"},
    "parameters": {
        "size_layers": {
            'values': [2 ** i for i in range(9)]
        },
        "num_layers": {
            'values': [2 ** i for i in range(8)]
        },
        "gamma": {
            "min": 0.8,
            "max": 1.0,
        },
        "epsilon_decay": {
            "min": 0.8,
            "max": 1.0,
        },
        "learning_rate": {
            "min": 1e-6,
            "max": 1e-1,
        },
        "target_update_interval": {
            "min": 1,
            "max": 100,
        },
        "batch_size": {
            'values': [2 ** i for i in range(10)]
        },
        "memory_size": {
            "distribution": "q_uniform",
            "min": 1,
            "max": 50_000,
            'q': 1000,
        },
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=SWEEP_NAME, entity='nyxai')
wandb.agent(sweep_id, function=main, count=1000)
