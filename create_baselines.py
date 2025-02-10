import os
import torch
import argparse
import tqdm
import time

from python.env.env import DeliveryDrones
from python.env.wrappers import WindowedGridView
from python.agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory


def create_baseline_models(num_models=5, num_steps=1000):
    # Create environment
    env_params = {
        'n_drones': 3,
        'drone_density': 0.05,
        'packets_factor': 3,
        'dropzones_factor': 2,
        'stations_factor': 2,
        'skyscrapers_factor': 3
    }
    env = WindowedGridView(DeliveryDrones(env_params), radius=3)

    # Create factory with different architectures for each baseline
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.n
    factories = [
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=[32]),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=[64]),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=[32, 32]),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=[64, 32]),
        DenseQNetworkFactory(obs_shape, action_shape, hidden_layers=[64, 64]),
        ConvQNetworkFactory(obs_shape, action_shape, conv_layers=[{'out_channels': 8, 'kernel_size': 3, 'stride': 1, 'padding': 1}], dense_layers=[8, 8, 8])
    ]

    # Create and save models
    for i in range(num_models):
        print(
            f"\nTraining model {i+1}/{num_models}")
        agent = DQNAgent(
            env=env,
            dqn_factory=factories[i % len(factories)],
            gamma=0.99,
            epsilon_start=0.1,
            epsilon_decay=0.995,
            epsilon_end=0.01,
            memory_size=10000,
            batch_size=32,
            target_update_interval=100
        )

        # Train for specified steps
        state = env.reset()
        total_reward = 0
        episode_rewards = []
        start_time = time.time()

        for step in tqdm.tqdm(range(num_steps), desc="Training"):
            actions = {j: agent.act(state[j])
                       for j in range(env_params['n_drones'])}
            next_state, rewards, dones, _, _ = env.step(actions)

            # Learn from experience
            for j in range(env_params['n_drones']):
                agent.learn(state[j], actions[j], rewards[j],
                            next_state[j], dones[j])
                total_reward += rewards[j]

            # Log progress
            if step > 0 and step % 100 == 0:
                avg_reward = total_reward / (step * env_params['n_drones'])
                elapsed = time.time() - start_time
                print(
                    f"\nStep {step}: Avg reward = {avg_reward:.3f}, Time elapsed = {elapsed:.1f}s")
                episode_rewards.append(avg_reward)

            state = next_state

        # Save model
        save_path = os.path.join(os.path.dirname(
            __file__), 'sample_models', f'dqn-agent-{i+1}.safetensors')
        agent.save(save_path)
        print(f"Saved model {i+1} to {save_path}")
        print(
            f"Final average reward: {total_reward / (num_steps * env_params['n_drones']):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create baseline DQN models')
    parser.add_argument(
        '--num-models', type=int, default=5,
        help='Number of baseline models to create'
    )
    parser.add_argument(
        '--num-steps', type=int, default=1000,
        help='Number of training steps per model'
    )
    args = parser.parse_args()

    create_baseline_models(args.num_models, args.num_steps)
