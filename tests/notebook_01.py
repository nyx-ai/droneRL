import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from PIL import Image

from env.env import DeliveryDrones

# Create environment
env = DeliveryDrones()

# Resets it and get the initial observation
observation = env.reset()

# Render in text
print(env.render(mode='ansi'))

Image.fromarray(env.render(mode='rgb_array'))

print(observation)

observation['ground'].grid

from env.wrappers import CompassQTable, CompassChargeQTable, LidarCompassQTable, LidarCompassChargeQTable

# Create the environment
env = DeliveryDrones()

# Use an observation wrappers
#env = CompassQTable(env)
env = CompassChargeQTable(env)
#env = LidarCompassQTable(env)
#env = LidarCompassChargeQTable(env)

# Reset the environment and print inital observation
observation = env.reset()
print(observation)

# Render as an RGB image
Image.fromarray(env.render(mode='rgb_array'))

# Print the state in a nicer way using `env.format_state`
{drone: env.format_state(observation) for drone, observation in observation.items()}

from env.env import Action

observation, reward, done, info = env.step({0: Action.STAY})

print('Rewards: {}'.format(reward))
Image.fromarray(env.render(mode='rgb_array'))

{drone: env.format_state(observation) for drone, observation in observation.items()}

from env.wrappers import WindowedGridView

env = WindowedGridView(DeliveryDrones(), radius=2)
states = env.reset()
Image.fromarray(env.render(mode='rgb_array'))
{drone: env.format_state(state) for drone, state in states.items()}
states[0][:, :, 5] # Obstacles from the perspective of drone 0

from agents.random import RandomAgent

# Create and setup the environment
env = WindowedGridView(DeliveryDrones(), radius=3)
states = env.reset()

# Create random agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents
# The random agents just pick an action randomly
from helpers.rl_helpers import test_agents

# Run agents for 1000 steps
rewards_log = test_agents(env, agents, n_steps=1000, seed=0)

# Print rewards
for drone_index, rewards in rewards_log.items():
    print('Drone {} rewards: {} ..'.format(drone_index, rewards[:10]))

from helpers.rl_helpers import plot_cumulative_rewards

plot_cumulative_rewards(
    rewards_log,
    events={'pickup': [1], 'crash': [-1]}, # Optional, default: pickup/crash ±1
    drones_labels={0: 'My drone'}, # Optional, default: drone index
)

from agents.dqn import DQNAgent, DenseQNetworkFactory
from helpers.rl_helpers import MultiAgentTrainer, plot_rolling_rewards

# Create and setup the environment
env = WindowedGridView(DeliveryDrones(), radius=3)
env.env_params.update({'n_drones': 3, 'skyscrapers_factor': 0, 'charge_reward': 0, 'discharge': 0})
states = env.reset()

# Create random agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}

# Use a DQNAgent for agent 0 - we will see how this works next
agents[0] = DQNAgent(
    env, DenseQNetworkFactory(env, hidden_layers=[32, 32]),
    gamma=0.95, epsilon_start=1.0, epsilon_decay=0.999, epsilon_end=0.01,
    memory_size=10000, batch_size=64, target_update_interval=5
)

agents
# Create trainer
trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)

# Train with different grids
trainer.train(1000)

# Print rewards
for drone_index, rewards in trainer.rewards_log.items():
    print('Drone {} rewards: {} ..'.format(drone_index, rewards[:10]))

plot_rolling_rewards(
    trainer.rewards_log,
    drones_labels={0: 'My drone'}, # Optional: specify drone names
)

rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(rewards_log, drones_labels={0: 'My drone'})

from helpers.rl_helpers import render_video, ColabVideo

path = os.path.join('output', 'videos', 'intro-run.mp4')
render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=None)
ColabVideo(path)

path = os.path.join('output', 'agents', 'first-agent.pt')
agents[0].save(path)