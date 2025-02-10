import random
from collections import defaultdict, Counter
from collections import deque

import gym.spaces as spaces
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor, LongTensor
import os
from safetensors.torch import save_file, safe_open
import ast
import operator as op
from functools import reduce

from .logging import Logger, NoLogger

# Determine and store the best available device globally
device = os.getenv('DEVICE')
if device is None:
    if torch.cuda.is_available():
        device = 'cuda'
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        # For now not using mps device
        print('device "mps" is available, but falling back to "cpu"')
        device = 'cpu'
    else:
        device = 'cpu'
print(device)


class QNetwork(nn.Module):
    """
    A Q-network for OpenAI Gym Environments
    """

    def forward(self, states):
        raise NotImplementedError


class DenseQNetwork(QNetwork):
    """
    A dense Q-network for OpenAI Gym Environments
    The network flattens the obs/action spaces and adds dense layers in between
    """

    def __init__(self, obs_shape, action_shape, hidden_layers=[]):
        assert isinstance(obs_shape, tuple), 'Observation space should be a tuple'
        assert isinstance(action_shape, int), 'Action space should be an integer'

        # Create network
        super().__init__()  # Initialize module
        # self.env = env  # Save environment

        self.input_size = reduce(op.mul, obs_shape, 1)
        self.output_size = action_shape
        self.dense_layers = hidden_layers

        self.network = nn.Sequential()
        hidden_layers = hidden_layers + [self.output_size]
        for i, hidden_size in enumerate(hidden_layers):
            # Create layer
            in_features = self.input_size if i == 0 else hidden_layers[i - 1]
            out_features = hidden_size
            layer = nn.Linear(in_features, out_features)

            # Add layer + activation
            if i > 0:
                self.network.add_module('dense_act_{}'.format(i), nn.ReLU())
            self.network.add_module('dense_{}'.format(i + 1), layer)

        # Move network to GPU if available
        self.network.to(device)

    def forward(self, states):
        states_tensor = torch.Tensor(np.array(states).reshape(-1, self.input_size)).to(device)
        return self.network(states_tensor)


class ConvQNetwork(QNetwork):
    """
    A convolutional Q-network with conv + dense architecture
    """

    def __init__(self, obs_shape, action_shape, conv_layers=[{'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1}], dense_layers=[]):
        assert isinstance(obs_shape, tuple), 'Observation space should be a tuple'
        assert isinstance(action_shape, int), 'Action space should be an integer'

        # Create network
        super().__init__()  # Initialize module

        self.input_shape = obs_shape
        self.output_size = action_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.network = nn.Sequential()

        # Convolutional layers
        for conv_i, conv_kwds in enumerate(self.conv_layers):
            assert 'out_channels' in conv_kwds, 'You need to define the number of kernels "out_channels" for each conv. layer'
            assert 'kernel_size' in conv_kwds, 'You need to define the kernel size "kernel_size" for each conv. layer'

            # Create layer
            in_channels = self.input_shape[-1] if conv_i == 0 else self.conv_layers[conv_i - 1]['out_channels']
            layer = nn.Conv2d(in_channels, **conv_kwds)

            # Add layer + activation
            self.network.add_module('conv2d_{}'.format(conv_i + 1), layer)
            self.network.add_module(
                'conv2d_act_{}'.format(conv_i + 1), nn.ReLU())

        # Flatten
        self.network.add_module('flatten', nn.Flatten())
        _, flatsize = self.network(torch.ones(
            [1, self.input_shape[2], self.input_shape[0], self.input_shape[1]])).shape

        # Dense layers
        dense_layers = self.dense_layers + [self.output_size]
        for dense_i, dense_layer in enumerate(dense_layers):
            # Create layer
            in_features = flatsize if dense_i == 0 else dense_layers[dense_i - 1]
            out_features = dense_layer
            layer = nn.Linear(in_features, out_features)

            # Add layer + activation
            if dense_i > 0:
                self.network.add_module(
                    'dense_act_{}'.format(dense_i), nn.ReLU())
            self.network.add_module('dense_{}'.format(dense_i + 1), layer)

        # Move network to GPU if available
        self.network.to(device)

    def forward(self, states):
        # Convert states to tensor and rearrange dimensions
        # states input shape is [batch_size, height, width, channels]
        # we need [batch_size, channels, height, width] for PyTorch convolutions
        states_tensor = torch.tensor(
            np.array(states), dtype=torch.float32, device=device).permute(0, 3, 1, 2)
        return self.network(states_tensor)


class BaseDQNFactory():
    """
    A template class to generate custom Q-networks and their optimizers
    """

    def create_qnetwork(self):
        # Should return network + optimizer
        raise NotImplementedError

    # static method creating appropriate instance based on safetensors metadata
    @classmethod
    def from_checkpoint(cls, path):
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            if metadata["network_type"] == "dense":
                return DenseQNetworkFactory.from_checkpoint(path)
            elif metadata["network_type"] == "conv":
                return ConvQNetworkFactory.from_checkpoint(path)
            else:
                raise ValueError(f"Unknown network type: {metadata['network_type']}")


class DenseQNetworkFactory(BaseDQNFactory):
    """
    A Q-network factory for dense Q-networks
    """

    def __init__(self, obs_shape, action_shape, hidden_layers=[], learning_rate: float = 1e-3, state_dict=None):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.dense_layers = hidden_layers
        self.learning_rate = learning_rate
        self.state_dict = state_dict

    def create_qnetwork(self):
        network = DenseQNetwork(self.obs_shape, self.action_shape, self.dense_layers)
        optimizer = optim.Adam(network.parameters(), lr=self.learning_rate)
        if self.state_dict is not None:
            network.load_state_dict(self.state_dict)
        return network, optimizer

    @staticmethod
    def from_checkpoint(path):
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            obs_shape = ast.literal_eval(metadata["obs_shape"])
            action_shape = ast.literal_eval(metadata["action_shape"])
            hidden_layers = ast.literal_eval(metadata["dense_layers"])
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
            return DenseQNetworkFactory(
                obs_shape=obs_shape,
                action_shape=action_shape,
                hidden_layers=hidden_layers,
                state_dict=state_dict
            )


class ConvQNetworkFactory(BaseDQNFactory):
    """
    A Q-network factory for convolutional Q-networks
    """

    def __init__(self, obs_shape, action_shape, conv_layers=None, dense_layers=None, state_dict=None):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.conv_layers = conv_layers
        self.dense_layers = dense_layers
        self.state_dict = state_dict

        # Validate conv layers have required parameters
        for layer in self.conv_layers:
            required = {'out_channels', 'kernel_size', 'stride', 'padding'}
            missing = required - set(layer.keys())
            if missing:
                raise ValueError(f"Conv layer missing required parameters: {missing}")

    def create_qnetwork(self):
        network = ConvQNetwork(self.obs_shape, self.action_shape, self.conv_layers, self.dense_layers)
        optimizer = optim.Adam(network.parameters())
        if self.state_dict is not None:
            network.load_state_dict(self.state_dict)
        return network, optimizer

    @staticmethod
    def from_checkpoint(path):
        with safe_open(path, framework="pt", device="cpu") as f:
            metadata = f.metadata()
            conv_layers = ast.literal_eval(metadata["conv_layers"])
            dense_layers = ast.literal_eval(metadata["dense_layers"])
            obs_shape = ast.literal_eval(metadata["obs_shape"])
            action_shape = ast.literal_eval(metadata["action_shape"])
            state_dict = {k: f.get_tensor(k) for k in f.keys()}
            return ConvQNetworkFactory(
                obs_shape=obs_shape,
                action_shape=action_shape,
                conv_layers=conv_layers,
                dense_layers=dense_layers,
                state_dict=state_dict
            )


class DQNAgent():
    """
    Deep Q-network agent (DQN) implementation
    Uses a NN to approximate the Q-function, a replay memory buffer
    and a target network.
    """

    def __init__(self, env, dqn_factory, gamma, epsilon_start, epsilon_decay, epsilon_end, memory_size, batch_size,
                 target_update_interval, logger: Logger = None):
        # Save parameters
        self.env = env
        self.dqn_factory = dqn_factory  # Factory to create q-networks + optimizers
        self.gamma = gamma  # Discount factor
        self.epsilon_start = epsilon_start  # Exploration rate
        self.epsilon_decay = epsilon_decay  # Decay after each episode
        self.epsilon_end = epsilon_end  # Minimum value
        self.memory_size = memory_size  # Size of the replay buffer
        self.batch_size = batch_size  # Batch size
        self.target_update_interval = target_update_interval  # Update rate
        self.is_greedy = False  # Does the agent behave greedily?
        self.logger = logger or NoLogger()

        self.reset()

    def reset(self):
        # Create networks with episode counter to know when to update them
        self.qnetwork, self.optimizer = self.dqn_factory.create_qnetwork()
        self.target_qnetwork, _ = self.dqn_factory.create_qnetwork()
        self.num_episode = 0
        self.episode_reward = 0
        self.total_steps = 0

        # Reset exploration rate
        self.epsilon = self.epsilon_start
        self.epsilons = []

        # Create new replay memory
        self.memory = deque(maxlen=self.memory_size)

    def save(self, path):
        state_dict = self.qnetwork.state_dict()
        metadata = {
            "network_type": "dense" if isinstance(self.qnetwork, DenseQNetwork) else "conv",
            "dense_layers": str(self.dqn_factory.dense_layers),
            "obs_shape": str(self.dqn_factory.obs_shape),
            "action_shape": str(self.dqn_factory.action_shape),
        }
        if hasattr(self.dqn_factory, 'conv_layers'):
            metadata["conv_layers"] = str([{
                'out_channels': layer['out_channels'],
                'kernel_size': layer['kernel_size'],
                'stride': layer['stride'],
                'padding': layer['padding']
            } for layer in self.dqn_factory.conv_layers])
        save_file(state_dict, path, metadata=metadata)

    def load(self, path):
        self.dqn_factory = BaseDQNFactory.from_checkpoint(path)
        self.qnetwork, self.optimizer = self.dqn_factory.create_qnetwork()
        self.target_qnetwork, _ = self.dqn_factory.create_qnetwork()

    def act(self, state):
        # Exploration rate
        epsilon = 0.0 if self.is_greedy else self.epsilon

        if np.random.rand() < epsilon:
            return self.env.action_space.sample()
        else:
            q_values = self.qnetwork([state])[0]
            return q_values.argmax().item()  # Greedy action

    def learn(self, state, action, reward, next_state, done):
        # Memorize experience
        self.memory.append((state, action, reward, next_state, done))
        self.episode_reward += reward
        self.total_steps += 1

        # End of episode
        if done:
            self.num_episode += 1  # Episode counter
            self.logger.log_dict(self.total_steps, {
                'episode_reward': self.episode_reward,
                'memory_size': len(self.memory),
            })
            self.epsilons.append(self.epsilon)  # Log epsilon value

            # Epsilon decay
            self.epsilon = max(
                self.epsilon * self.epsilon_decay, self.epsilon_end)

            self.episode_reward = 0

        # Periodically update target network with current one
        if self.total_steps % self.target_update_interval == 0:
            self.target_qnetwork.load_state_dict(self.qnetwork.state_dict())

        # Train when we have enough experiences in the replay memory
        if len(self.memory) > self.batch_size:
            # Sample batch of experience
            # batch = random.sample(self.memory, self.batch_size)
            batch_indices = np.random.choice(
                len(self.memory), self.batch_size, replace=False)
            batch = [self.memory[idx] for idx in batch_indices]
            state, action, reward, next_state, done = zip(*batch)

            action = LongTensor(action).to(device)
            reward = Tensor(reward).to(device)
            done = Tensor(done).to(device)

            # Q-value for current state given current action
            q_values = self.qnetwork(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)

            # Compute the TD target
            next_q_values = self.target_qnetwork(next_state)
            next_q_value = next_q_values.max(1)[0]

            td_target = reward + self.gamma * next_q_value * (1 - done)

            # Optimize quadratic loss
            loss = (q_value - td_target.detach()).pow(2).mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.logger.log_dict(self.total_steps, {
                'dqn/loss': loss.item(),
                'dqn/reward': reward.mean().item(),
            })
        else:
            # print(f"Not learning yet! {len(self.memory)}/{self.batch_size} experiences")
            pass

    def __repr__(self) -> str:
        return f"DQNAgent(dqn_factory={self.dqn_factory})"

    def inspect_memory(self, top_n=10, max_col=80):
        # Functions to encode/decode states
        def encode_state(s): return tuple(
            spaces.flatten(self.env.observation_space, s))

        def decode_state(s): return spaces.unflatten(
            self.env.observation_space, s)

        # Function to create barchart from counter
        def count_barchart(counter, ax, xlabel=None, normalize=True):
            # Sort and extract key, counts
            sorted_tuples = counter.most_common()
            sorted_keys = [key for key, count in sorted_tuples]
            sorted_counts = [count for key, count in sorted_tuples]

            # Normalize counts
            if normalize:
                total = sum(counters['reward'].values())
                sorted_counts = [c / total for c in sorted_counts]

            # Plotting
            x_indexes = range(len(sorted_counts))
            ax.bar(x_indexes, sorted_counts)
            ax.set_xticks(x_indexes)
            ax.set_xticklabels(sorted_keys)
            ax.set_ylabel('proportion')
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            ax.set_title('Replay Memory')

        # Function to print top states from counter
        def top_states(counter):
            for i, (state, count) in enumerate(counter.most_common(top_n), 1):
                state_label = str(decode_state(state))
                state_label = state_label.replace('\n', ' ')
                state_label = state_label[:max_col] + \
                    '..' if len(state_label) > max_col else state_label
                print('{:>2}) Count: {} state: {}'.format(
                    i, count, state_label))

        # Count statistics
        counters = defaultdict(Counter)
        for state, action, reward, next_state, done in self.memory:
            counters['state'][encode_state(state)] += 1
            counters['action'][action] += 1
            counters['reward'][reward] += 1
            counters['next_state'][encode_state(next_state)] += 1
            counters['done'][done] += 1

        # Plot reward/action
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
        count_barchart(counters['reward'], ax1, 'rewards')
        count_barchart(counters['action'], ax2, 'actions')
        plt.plot()
        plt.show()

        # Print top states
        print('Top state:')
        top_states(counters['state'])
        print()

        print('Top next_state:')
        top_states(counters['next_state'])
        print()

        # Done signal
        print('Proportion of done: {:.2f}%'.format(
            100 * counters['done'][True] / sum(counters['done'].values())))
