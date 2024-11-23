import os
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.autograd.set_detect_anomaly(True)

import os
import statistics
import time
import numpy as np

NUMBER_OF_RUNS = 3


def time_function(function, *args, **kwargs):
    times = []
    for _ in range(NUMBER_OF_RUNS):
        start_time = time.time()
        function(*args, **kwargs)
        end_time = time.time()
        times.append(end_time - start_time)

    mean_time = np.mean(times)
    min_time = np.min(times)
    max_time = np.max(times)
    std_dev = np.std(times)

    result = {
        "Mean Time": f"{mean_time:.4f} seconds",
        "Min Time": f"{min_time:.4f} seconds",
        "Max Time": f"{max_time:.4f} seconds",
        "Standard Deviation": f"{std_dev:.4f} seconds"
    }

    return result


# ------------------------------
# Original version (f17edd7f0f2e5c3e4bae95c886dd476e3cdaf78f)
# Mean Time: 3.3069 seconds
# Min Time: 3.2401 seconds
# Max Time: 3.3337 seconds
# Standard Deviation: 0.0343 seconds
# ------------------------------

from env.env import DeliveryDrones
from env.wrappers import WindowedGridView
from agents.random import RandomAgent
from agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory
from helpers.rl_helpers import MultiAgentTrainer, plot_rolling_rewards
from helpers.rl_helpers import test_agents, plot_cumulative_rewards

DEVICE = "mps"

testing_env = WindowedGridView(DeliveryDrones(device=DEVICE), radius=3)
testing_env.env_params.update({
    'n_drones': 1,
    'charge_reward': -0.1,
    'crash_reward': -1,
    'pickup_reward': 0,
    'delivery_reward': 1,
    'charge': 20,
    'discharge': 10,
    'drone_density': 0.05,
    'dropzones_factor': 2,
    'packets_factor': 3,
    'skyscrapers_factor': 3,
    'stations_factor': 2
})

training_env = WindowedGridView(DeliveryDrones(device=DEVICE), radius=3)
training_env.env_params.update({
    'n_drones': 1,
    'charge_reward': -0.1,
    'crash_reward': -1,
    'pickup_reward': 0.1,
    'delivery_reward': 1,
    'charge': 20,
    'discharge': 10,
    'drone_density': 0.05,
    'dropzones_factor': 2,
    'packets_factor': 3,
    'skyscrapers_factor': 3,
    'stations_factor': 2
})
training_env.reset()
print(training_env.render(mode='ansi'))
agents = {drone.index: RandomAgent(training_env) for drone in training_env.drones}
agents[0] = DQNAgent(
    env=training_env,
    dqn_factory=DenseQNetworkFactory(training_env, hidden_layers=[32] * 2),
    # dqn_factory=ConvQNetworkFactory(
    #     training_env,
    #     conv_layers=[{'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1}],
    #     dense_layers=[64] * 4
    # ),
    gamma=0.95,
    epsilon_start=1.0,
    epsilon_decay=0.9,
    epsilon_end=1.0,
    memory_size=10000,
    batch_size=4,
    target_update_interval=5
)
trainer = MultiAgentTrainer(training_env, agents, reset_agents=True, seed=0)

timing_results = time_function(trainer.train, 1500)
for key, value in timing_results.items():
    print(f"{key}: {value}")
# rewards_log = test_agents(testing_env, agents, n_steps=10_000, seed=0)
# print({k: statistics.mean(v) for k, v in rewards_log.items()})
# path = os.path.join('output', 'agents', f'benchmark-agent.pt')
# agents[0].save(path)
