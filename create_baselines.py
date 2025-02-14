import os
import torch
import argparse
import tqdm
import time

from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory
from torch_impl.agents.random import RandomAgent
from torch_impl.helpers.rl_helpers import set_seed


def create_baseline_models(num_models=5, num_steps=1000):
    # Create environment
    env_params = {
        'n_drones': 8,
        'drone_density': 0.05,

        # rewards
        'pickup_reward': 0.1,  # reward hacking (should be 0)
        'charge_reward': -0.1,  # reward hacking (should be -0.1)
        'delivery_reward': 1.0,  # reward hacking (should be 1.0)
        'crash_reward': -1.0,  # reward hacking (should be -1.0)

        # densities
        'stations_factor': 2,
        'packets_factor': 3,
        'dropzones_factor': 2,
        'skyscrapers_factor': 3,
    }
    env = WindowedGridView(DeliveryDrones(env_params), radius=3)
    set_seed(env, 0)

    # Create factory with different architectures for each baseline
    obs_shape = env.observation_space.shape
    action_shape = (env.action_space.n,)
    factories = [
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=(8,)),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=(16, 16)),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=(32, 32)),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=(64, 32)),
        ConvQNetworkFactory(
            obs_shape,
            action_shape,
            conv_layers=[{'out_channels': 4, 'kernel_size': 3, 'stride': 1, 'padding': 1}],
            dense_layers=(8,)
        ),
        ConvQNetworkFactory(
            obs_shape,
            action_shape,
            conv_layers=[
                {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1},
                {'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1}
            ],
            dense_layers=(16,)),
    ]

    # Create and save models
    for i in range(num_models):
        print(f"\nTraining model {i+1}/{num_models}")
        agent = DQNAgent(
            env=env,
            dqn_factory=factories[i % len(factories)],
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_decay=0.999,
            epsilon_end=0.1,
            memory_size=1000,
            batch_size=1,
            target_update_interval=8,
        )

        # Train for specified steps
        state = env.reset()
        total_reward = 0
        recent_rewards = []

        agents = {drone.index: RandomAgent(env) for drone in env.drones_list}
        agents[0] = agent

        pbar = tqdm.tqdm(range(num_steps))
        for step in pbar:
            actions = {key: agent.act(state[key]) for key, agent in agents.items()}
            next_state, rewards, dones, _, _ = env.step(actions)
            step_reward = rewards[0]

            # Learn from experience
            agent.learn(state[0], actions[0], rewards[0], next_state[0], dones[0])

            recent_rewards.append(step_reward)
            if len(recent_rewards) > 1000:
                recent_rewards.pop(0)
            total_reward += step_reward

            # Log progress
            if step > 0 and step % 10 == 0:
                avg_recent_reward = sum(recent_rewards) / len(recent_rewards)
                pbar.set_description(f"epsilon:{agent.epsilon:.2f}, reward_1000:{avg_recent_reward/100:.4f}")

            state = next_state

        # Save model
        save_path = os.path.join(os.path.dirname(__file__), 'sample_models', f'dqn-agent-{i+1}.safetensors')
        agent.save(save_path)
        print(f"Saved model {i+1} to {save_path}")
        print(f"Final average reward: {total_reward / (num_steps):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create baseline DQN models')
    parser.add_argument(
        '--num-models', type=int, default=5,
        help='Number of baseline models to create'
    )
    parser.add_argument(
        '--num-steps', type=int, default=10000,
        help='Number of training steps per model'
    )
    args = parser.parse_args()

    create_baseline_models(args.num_models, args.num_steps)
