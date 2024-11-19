import statistics

import wandb

from agents.dqn import DQNAgent, DenseQNetworkFactory
from agents.random import RandomAgent
from env.env import DeliveryDrones
from env.wrappers import WindowedGridView
from helpers.rl_helpers import MultiAgentTrainer
from helpers.rl_helpers import test_agents

wandb.login()

SWEEP_NAME = "dronerl-dense-1"
NUM_TRAINING_STEPS = 15_000
NUM_TESTING_STEPS = 50_000


def evaluate(config):
    testing_env = WindowedGridView(DeliveryDrones(), radius=3)
    testing_env.env_params.update({
        'n_drones': 3,
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

    training_env = WindowedGridView(DeliveryDrones(), radius=3)
    training_env.env_params.update({
        'n_drones': config.n_drones,
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
    training_env.reset()
    agents = {drone.index: RandomAgent(training_env) for drone in training_env.drones}
    agents[0] = DQNAgent(
        env=training_env,
        dqn_factory=DenseQNetworkFactory(training_env, hidden_layers=[config.size_layers] * config.num_layers),
        gamma=config.gamma,
        epsilon_start=1.0,
        epsilon_decay=config.epsilon_decay,
        epsilon_end=0.01,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update_interval=config.target_update_interval
    )
    trainer = MultiAgentTrainer(training_env, agents, reset_agents=True, seed=0)
    trainer.train(NUM_TRAINING_STEPS)
    rewards_log = test_agents(training_env, agents, n_steps=NUM_TESTING_STEPS, seed=0)
    rewards = {k: statistics.mean(v) for k, v in rewards_log.items()}
    return rewards[0]


def main():
    wandb.init(project=SWEEP_NAME)
    score = evaluate(wandb.config)
    wandb.log({"score": score})


sweep_configuration = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "size_layers": {
            "min": 1,
            "max": 512,
        },
        "num_layers": {
            "min": 1,
            "max": 8,
        },
        "gamma": {
            "min": 0.8,
            "max": 1.0,
        },
        "epsilon_decay": {
            "min": 0.8,
            "max": 1.0,
        },
        "target_update_interval": {
            "min": 1,
            "max": 100,
        },
        "batch_size": {
            "min": 1,
            "max": 1000,
        },
        "memory_size": {
            "min": 1,
            "max": 1000,
        },
        "n_drones": {
            "min": 1,
            "max": 10,
        }
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=SWEEP_NAME)
wandb.agent(sweep_id, function=main, count=10)
