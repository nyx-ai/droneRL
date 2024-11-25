import os
import sys

from tqdm import tqdm

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from env.env import DeliveryDrones
from agents.random import RandomAgent

env = DeliveryDrones()
env.env_params.update({
    'n_drones': 4096,
    'drone_density': 0.03,
    'charge_reward': -0.1,
    'crash_reward': -1,
    'pickup_reward': 0,
    'delivery_reward': 1,
    'charge': 20,
    'discharge': 10,
    'dropzones_factor': 0,
    'packets_factor': 0,
    'skyscrapers_factor': 0,
    'stations_factor': 0
})

states, info = env.reset()
agents = {drone: RandomAgent(env) for drone in env.drones}

for _ in tqdm(range(10_000_000)):
    actions = {key: agent.act(None) for key, agent in agents.items()}
    env.step(actions)
    # print(env.render(mode='ansi'))
