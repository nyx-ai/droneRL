import statistics
from tqdm import tqdm
from pprint import pprint

from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.agents.dqn import DQNAgent, DenseQNetworkFactory
from torch_impl.agents.random import RandomAgent


def benchmark_implementation(env, config, n_drones, n_steps):
    config_params = {**config, 'n_drones': n_drones, 'pickup_reward': 0.5}
    env.env_params.update(config_params)
    states = env.reset()
    print(env.render())

    rewards_log = {}
    agents = {drone.index: RandomAgent(env) for drone in env.drones_list}
    rewards_log = {key: [] for key in agents.keys()}
    agents[0] = DQNAgent(
        env=env,
        dqn_factory=DenseQNetworkFactory(
            env.observation_space.shape,
            (env.action_space.n,),
            hidden_layers=(16,) * 2
        ),
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=0.01 ** (2 / n_steps),
        epsilon_end=0.01,
        memory_size=10_000_000,
        batch_size=32,
        target_update_interval=4
    )

    for i in tqdm(range(n_steps), leave=True):
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}
        step = env.step(actions)

        next_states, rewards, dones, info, _ = step
        for key, agent in agents.items():
            agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
            rewards_log[key].append(rewards[key])
        states = next_states

    reward_trained = statistics.mean(rewards_log[0][-100:])
    reward_untrained = statistics.mean(rewards_log[1][-100:])

    result = {
        'n_drones': n_drones,
        'grid_size': f"{env.side_size}x{env.side_size}",
        'reward_trained': reward_trained,
        'reward_untrained': reward_untrained
    }
    pprint(result)


def test_small_grid_drones_only():
    env = WindowedGridView(DeliveryDrones(), radius=3)
    config = {'packets_factor': 0, 'dropzones_factor': 0, 'stations_factor': 0, 'skyscrapers_factor': 0}
    n_steps = 1000
    n_drones = 4
    benchmark_implementation(env, config, n_drones, n_steps)


# def test_small_grid_default():
#     env = WindowedGridView(DeliveryDrones(), radius=3)
#     config = {}
#     n_steps = 1000
#     n_drones = 4
#     benchmark_implementation(env, config, n_drones, n_steps)


# def test_small_grid_high_density():
#     env = WindowedGridView(DeliveryDrones(), radius=3)
#     config = {'packets_factor': 4, 'dropzones_factor': 4, 'stations_factor': 4, 'skyscrapers_factor': 4}
#     n_steps = 1000
#     n_drones = 4
#     benchmark_implementation(env, config, n_drones, n_steps)


if __name__ == '__main__':
    test_small_grid_drones_only()
    # test_small_grid_default()
    # test_small_grid_high_density()
