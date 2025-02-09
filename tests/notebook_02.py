import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

from env.env import DeliveryDrones
from env.wrappers import CompassQTable, LidarCompassQTable, LidarCompassChargeQTable, WindowedGridView
from agents.random import RandomAgent
from agents.qlearning import QLearningAgent
from agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory
from helpers.rl_helpers import MultiAgentTrainer, plot_rolling_rewards, test_agents, plot_cumulative_rewards, \
    render_video, set_seed

# Environment without Skyscrapers + discharge
env = CompassQTable(DeliveryDrones())
env.env_params.update({'n_drones': 3, 'skyscrapers_factor': 0, 'stations_factor': 0, 'discharge': 0})
states = env.reset()

print('Observation space:', env.observation_space)
print('Initial state:', {drone_index: env.format_state(state) for drone_index, state in states.items()})
Image.fromarray(env.render(mode='rgb_array'))

# Create the agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents[0] = QLearningAgent(
    env,
    gamma=0.95,  # Discount factor
    alpha=0.1,  # Learning rate
    # Exploration rate
    epsilon_start=1, epsilon_decay=0.99, epsilon_end=0.01
)
agents
# Train agents
trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)
trainer.train(5000)
plot_rolling_rewards(trainer.rewards_log, drones_labels={0: 'Q-learning'})
agents[0].get_qtable()
plt.plot(agents[0].gamma ** np.arange(100))
plt.title('Discount factor: {}'.format(agents[0].gamma))
plt.xlabel('Number of steps')
plt.ylabel('Discount')
plt.show()
rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(rewards_log, drones_labels={0: 'Q-learning'})
path = os.path.join('output', 'videos', 'ql-compass.mp4')
render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=0)

# Environment with skyscrapers but without discharge
env = LidarCompassQTable(DeliveryDrones())
env.env_params.update({'n_drones': 3, 'skyscrapers_factor': 3, 'stations_factor': 0, 'discharge': 0})
states = env.reset()

print('Observation space:', env.observation_space)
print('Sample state:', {drone_index: env.format_state(state) for drone_index, state in states.items()})
Image.fromarray(env.render(mode='rgb_array'))
# Create the agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents[0] = QLearningAgent(
    env,
    gamma=0.95,  # Discount factor
    alpha=0.1,  # Learning rate
    # Exploration rate
    epsilon_start=1, epsilon_decay=0.99, epsilon_end=0.01
)
agents
# Train agents
trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)
trainer.train(5000)
plot_rolling_rewards(trainer.rewards_log, drones_labels={0: 'Q-learning'})
rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(rewards_log, drones_labels={0: 'Q-learning'})
path = os.path.join('output', 'videos', 'ql-compass-lidar-1st-try.mp4')
render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=0)

q_table = agents[0].get_qtable()
print('Q-table:', q_table.shape)
q_table.sample(10)
plt.plot(agents[0].epsilons)
plt.xlabel('Number of episodes')
plt.ylabel('Exploration rate (epsilon)')
plt.show()

# (1/2) Sparse rewards: Create an intermediate "pickup" reward
env.env_params.update({
    'n_drones': 3, 'pickup_reward': 0.99, 'delivery_reward': 1,
    'skyscrapers_factor': 3, 'stations_factor': 0, 'discharge': 0})
states = env.reset()

# (2/2) Train longer ..
agents[0].epsilon = 1
agents[0].epsilon_decay = 0.999

set_seed(env, seed=0)  # Make things deterministic
trainer.train(30000)

plot_rolling_rewards(
    trainer.rewards_log,
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1]},
    drones_labels={0: 'Q-learning'})
plt.plot(agents[0].epsilons)
plt.xlabel('Number of episodes')
plt.ylabel('Exploration rate (epsilon)')
plt.show()
rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(
    rewards_log,
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1]},
    drones_labels={0: 'Q-learning'}
)

rewards_log = test_agents(env, agents, n_steps=1000, seed=1)
plot_cumulative_rewards(
    rewards_log,
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1]},
    drones_labels={0: 'Q-learning'}
)

path = os.path.join('output', 'videos', 'ql-compass-lidar-2nd-try.mp4')
render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=1)

env = LidarCompassChargeQTable(DeliveryDrones())
env.env_params.update({
    'n_drones': 3, 'pickup_reward': 0.99, 'delivery_reward': 1,
    'discharge': 10, 'charge': 20, 'charge_reward': -0.1  # (default values)
})
states = env.reset()

print('Observation space:', env.observation_space)
print('Sample state:', env.format_state(states[0]))
Image.fromarray(env.render(mode='rgb_array'))
# Create the agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents[0] = QLearningAgent(
    env, gamma=0.95, alpha=0.1,
    epsilon_start=1, epsilon_decay=0.999, epsilon_end=0.01
)

trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)
trainer.train(35000)
plot_rolling_rewards(trainer.rewards_log, events={'pickup': [0.99], 'delivery': [1], 'crash': [-1], 'charging': [-0.1]})
q_table = agents[0].get_qtable()
print('Q-table:', q_table.shape)
q_table.sample(10)

rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(
    rewards_log,
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1], 'charging': [-0.1]},
    drones_labels={0: 'Q-learning'}
)

# path = os.path.join('output', 'videos', 'ql-compass-lidar-charge.mp4')
# render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=0)

# Create environment
env = LidarCompassChargeQTable(DeliveryDrones())
env.env_params.update({
    'n_drones': 3, 'pickup_reward': 0.99, 'delivery_reward': 1
})
states = env.reset()

# Create the agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents[0] = DQNAgent(
    env, DenseQNetworkFactory(env, hidden_layers=[256, 256]),
    gamma=0.95, epsilon_start=1, epsilon_decay=0.999, epsilon_end=0.01,
    memory_size=10000, batch_size=64, target_update_interval=5
)
trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)
agents[0].qnetwork
# Train the agents
trainer.train(2500)
plot_rolling_rewards(
    trainer.rewards_log, drones_labels={0: 'DQN'},
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1], 'charging': [-0.1]})
plt.plot(agents[0].epsilons)
plt.xlabel('Number of episodes')
plt.ylabel('Exploration rate (epsilon)')
plt.show()

rewards_log = test_agents(env, agents, n_steps=1000, seed=0)
plot_cumulative_rewards(
    rewards_log, drones_labels={0: 'DQN'},
    events={'pickup': [0.99], 'delivery': [1], 'crash': [-1], 'charging': [-0.1]})
# Inspect replay memory buffer
agents[0].inspect_memory(top_n=10, max_col=80)

# path = os.path.join('output', 'videos', 'dqn-compass-lidar-charge.mp4')
# render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=0)

# Create environment
env = WindowedGridView(DeliveryDrones(), radius=3)
env.env_params.update({
    'n_drones': 3, 'pickup_reward': 0.99, 'delivery_reward': 1
})
states = env.reset()

# Create the agents
agents = {drone.index: RandomAgent(env) for drone in env.drones}
agents[0] = my_agent = DQNAgent(
    env, ConvQNetworkFactory(env, conv_layers=[
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
        {'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1},
    ], dense_layers=[1024, 256]),
    gamma=0.95, epsilon_start=1, epsilon_decay=0.99, epsilon_end=0.01,
    memory_size=10000, batch_size=64, target_update_interval=5
)
trainer = MultiAgentTrainer(env, agents, reset_agents=True, seed=0)
agents[0].qnetwork
# Train the agents
for run in range(1):
    trainer.train(250)
    plot_rolling_rewards(
        trainer.rewards_log, drones_labels={0: 'DQN'},
        events={'pickup': [0.99], 'delivery': [1], 'crash': [-1], 'charging': [-0.1]})
path = os.path.join('output', 'videos', 'dqn-windowed.mp4')
render_video(env, agents, video_path=path, n_steps=120, fps=1, seed=0)

path = os.path.join('output', 'agents', 'dqn-agent.pt')
agents[0].save(path)
