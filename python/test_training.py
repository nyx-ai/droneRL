import statistics
import pytest
from tqdm import tqdm
from env.env import DeliveryDrones
from env.wrappers import WindowedGridView
from agents.dqn import DQNAgent, DenseQNetworkFactory
from agents.random import RandomAgent


@pytest.fixture
def env():
    return WindowedGridView(DeliveryDrones(), radius=3)


@pytest.fixture
def base_config():
    return {
        'n_steps': 1_000,
        'n_drones': 4,
        'pickup_reward': 0.5
    }


def train(env, config, n_drones, n_steps):
    config_params = {**config, 'n_drones': n_drones, 'pickup_reward': 0.5}
    env.env_params.update(config_params)
    states = env.reset()

    rewards_log = {}
    agents = {drone.index: RandomAgent(env) for drone in env.drones}
    rewards_log = {key: [] for key in agents.keys()}
    agents[0] = DQNAgent(
        env=env,
        dqn_factory=DenseQNetworkFactory(
            env.observation_space.shape,
            env.action_space.n,
            hidden_layers=[16] * 2
        ),
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=0.01 ** (2 / n_steps),
        epsilon_end=0.01,
        memory_size=10_000_000,
        batch_size=32,
        target_update_interval=4
    )

    for _ in tqdm(range(n_steps), leave=True):
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}
        next_states, rewards, dones, info, _ = env.step(actions)
        for key, agent in agents.items():
            agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
            rewards_log[key].append(rewards[key])
        states = next_states

    return {
        'reward_trained': statistics.mean(rewards_log[0][-100:]),
        'reward_untrained': statistics.mean(rewards_log[1][-100:])
    }


def test_drones_only(env, base_config):
    config = {'packets_factor': 0, 'dropzones_factor': 0, 'stations_factor': 0, 'skyscrapers_factor': 0}
    train(env, config, base_config['n_drones'], base_config['n_steps'])


def test_default_environment(env, base_config):
    train(env, {}, base_config['n_drones'], base_config['n_steps'])


def test_high_density(env, base_config):
    config = {'packets_factor': 4, 'dropzones_factor': 4, 'stations_factor': 4, 'skyscrapers_factor': 4}
    train(env, config, base_config['n_drones'], base_config['n_steps'])
