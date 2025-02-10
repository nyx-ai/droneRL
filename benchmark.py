import statistics
import time
from dataclasses import dataclass

import gym.spaces as spaces
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
import matplotlib.pyplot as plt

from python.env.env import DeliveryDrones as DeliveryDronesV3
from python.env.wrappers import WindowedGridView as WindowedGridViewV3

from python.agents.dqn import DQNAgent, DenseQNetworkFactory
from python.agents.random import RandomAgent


@dataclass
class Impl:
    name: str
    env: object
    desc: str


@dataclass
class Config:
    name: str
    params: dict


# CONFIG #
train = False
n_steps = 25
# drone_counts = [3]
drone_counts = [32, 128, 512, 2048]

configs = [
    Config(
        name="DronesOnly",
        params={'packets_factor': 0, 'dropzones_factor': 0, 'stations_factor': 0, 'skyscrapers_factor': 0}
    ),
    Config(
        name="Default",
        params={}
    ),
    Config(
        name="HighDensity",
        params={'packets_factor': 4, 'dropzones_factor': 4, 'stations_factor': 4, 'skyscrapers_factor': 4}
    ),
]

impls = [
    # Impl(
    #     name="v1",
    #     env=WindowedGridViewV1(DeliveryDronesV1(), radius=3),
    #     desc="original 2020 version"
    # ),
    # Impl(
    #     name="v2",
    #     env=WindowedGridViewV2(DeliveryDronesV2(), radius=3),
    #     desc="2020 grid-based version using constants instead of objects for what's on the map"
    # ),
    Impl(
        name="v3",
        env=WindowedGridViewV3(DeliveryDronesV3(), radius=3),
        desc="dict-based version"
    ),
]
# CONFIG #

results = []
reference_speeds = {}


def plot_rewards(rewards_log, title="Reward Progression", window_size=1000):
    plt.figure(figsize=(10, 6))
    for drone, rewards in rewards_log.items():
        smoothed_rewards = np.convolve(rewards, np.ones(window_size) / window_size, mode='valid')
        plt.plot(smoothed_rewards, label=f"Drone {drone}")

    plt.xlabel("Steps")
    plt.ylabel("Rewards")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def benchmark_implementation(imp, config, n_drones, n_steps, train):
    # Update config with the number of drones for this run
    config_params = {**config.params, 'n_drones': n_drones, 'pickup_reward': 0.5}
    imp.env.env_params.update(config_params)
    states = imp.env.reset()
    # print(imp.env.render(mode='ansi'))

    rewards_log = {}
    if train:
        agents = {drone.index: RandomAgent(imp.env) for drone in imp.env.drones}
        rewards_log = {key: [] for key in agents.keys()}
        agents[0] = DQNAgent(
            env=imp.env,
            dqn_factory=DenseQNetworkFactory(
                imp.env,
                hidden_layers=[16] * 2
            ),
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_decay=0.01 ** (2 / n_steps),  # decay to 0.01 after half of the steps
            epsilon_end=0.01,
            memory_size=10_000_000,
            batch_size=32,
            target_update_interval=4
        )

    total_time_act = 0
    total_time_env = 0
    total_time_learn = 0
    start_time = time.perf_counter()

    pbar = tqdm(range(n_steps), desc=f"{imp.name} {config.name} {n_drones} drones", leave=True)
    for i in pbar:
        time_act_start = time.perf_counter()
        if train:
            actions = {key: agent.act(states[key]) for key, agent in agents.items()}
        else:
            actions = {drone.index: imp.env.action_space.sample() for drone in imp.env.drones}
        total_time_act += time.perf_counter() - time_act_start

        time_env_start = time.perf_counter()
        step = imp.env.step(actions)
        total_time_env += time.perf_counter() - time_env_start
        # print(imp.env.render(mode='ansi'))

        time_learn_start = time.perf_counter()
        if train:
            next_states, rewards, dones, info, _ = step
            for key, agent in agents.items():
                agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
                rewards_log[key].append(rewards[key])
            states = next_states
        total_time_learn += time.perf_counter() - time_learn_start

        # pbar.set_description(f"{statistics.mean(rewards_log[0][-500:]):.3f}")

    total_time = time.perf_counter() - start_time
    mean_time = (total_time / n_steps) * 1000
    sps = n_steps / total_time
    mean_time_act = (total_time_act / n_steps) * 1000
    mean_time_env = (total_time_env / n_steps) * 1000
    mean_time_lean = (total_time_learn / n_steps) * 1000

    reward_trained, reward_untrained = "-", "-"
    if train:
        reward_trained = statistics.mean(rewards_log[0][-100:])
        reward_untrained = statistics.mean(rewards_log[1][-100:])

    # Identify reference time for comparison
    key = (config.name, n_drones)
    if imp.name == "v1":
        reference_speeds[key] = mean_time
    percent_diff = (reference_speeds[key] / mean_time) if key in reference_speeds else 0

    results.append([
        imp.name, config.name, n_drones, f"{imp.env.side_size}x{imp.env.side_size}", f"{mean_time:.2f} spKs",
        f"{sps:.1f} sps",
        # f"{percent_diff:.2f}",
        f"{mean_time_act:.3f} ms", f"{mean_time_env:.3f} ms", f"{mean_time_lean:.3f} ms",
        reward_trained, reward_untrained
    ])

    # plot_rewards(rewards_log)


if __name__ == '__main__':
    for imp in impls:
        for config in configs:
            for n_drones in drone_counts:
                benchmark_implementation(imp, config, n_drones, n_steps, train)

    print(f"Training? {train}")
    print(
        tabulate(results, headers=[
            'Implementation', 'Config', 'Drones', "Size", 'Speed', 'Speed',
            # 'Speedup from V1',
            'Time act', 'Time env', 'Time learn',
            'Score trained', 'Score untrained'
        ])
    )
    print("=" * 15)
    for imp in impls:
        print(f"{imp.name}: {imp.desc}")
    print("=" * 15)
    print(f"{n_steps:,} steps per run.")
