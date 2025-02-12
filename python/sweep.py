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
NUM_TRAINING_STEPS = 100
NUM_TESTING_STEPS = 100


def evaluate(config):
    testing_env = WindowedGridView(DeliveryDrones(), radius=3)
    # official AIcrowd values
    testing_env.env_params.update({
        'n_drones': 6,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'pickup_reward': 0,
        'delivery_reward': 1,
        'charge': 20,
        'discharge': 10,
        'drone_density': 0.03,
        'dropzones_factor': 2,
        'packets_factor': 3,
        'skyscrapers_factor': 3,
        'stations_factor': 2
    })

    drone_density = 0.03
    n_drones = 1024

    training_env = WindowedGridView(DeliveryDrones(), radius=3)
    training_env.env_params.update({
        'n_drones': n_drones,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'pickup_reward': config.pickup_reward,
        'delivery_reward': 1,
        'charge': 20,
        'discharge': 10,
        'drone_density': drone_density,
        'dropzones_factor': 2,
        'packets_factor': 3,
        'skyscrapers_factor': 3,
        'stations_factor': 2
    })

    # TRAINING
    training_env.reset()
    agents = {drone.index: RandomAgent(training_env) for drone in training_env.drones}
    agents[0] = DQNAgent(
        env=training_env,
        dqn_factory=DenseQNetworkFactory(
            training_env,
            hidden_layers=[config.size_layers] * config.num_layers,
            learning_rate=config.learning_rate),
        gamma=config.gamma,
        epsilon_start=1.0,
        epsilon_decay=config.epsilon_decay,
        epsilon_end=0.01,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update_interval=config.target_update_interval
    )
    trainer = MultiAgentTrainer(training_env, training_agents, reset_agents=True, seed=0)
    trainer.train(NUM_TRAINING_STEPS)

    # EVALUATION
    testing_env.reset()
    testing_agents = {drone.index: RandomAgent(testing_env) for drone in testing_env.drones}
    testing_agents[0] = training_agents[0]
    rewards_log = test_agents(testing_env, testing_agents, n_steps=NUM_TESTING_STEPS, seed=0)
    rewards = {k: statistics.mean(v) for k, v in rewards_log.items()}
    return rewards[0]


def main():
    wandb.init(project=SWEEP_NAME)
    score = evaluate(wandb.config)
    wandb.log({"score": score})


# https://docs.wandb.ai/guides/sweeps/define-sweep-configuration
sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "pickup_reward": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0,
        },
        "size_layers": {
            "distribution": "q_log_uniform_values",
            "min": 1,
            "max": 256,
        },
        "num_layers": {
            "distribution": "int_uniform",
            "min": 1,
            "max": 4,
        },
        "gamma": {
            "distribution": "q_log_uniform_values",
            "q": 0.001,
            "min": 0.9,
            "max": 1.0,
        },
        "epsilon_decay": {
            "distribution": "q_log_uniform_values",
            "q": 0.001,
            "min": 0.9,
            "max": 1.0,
        },
        "learning_rate": {
            "min": 1e-6,
            "max": 1e-1,
        },
        "target_update_interval": {
            "distribution": "q_log_uniform_values",
            "min": 1,
            "max": 256,
        },
        "batch_size": {
            "distribution": "q_log_uniform_values",
            "min": 1,
            "max": 256,
        },
        "n_drones": {
            "distribution": "int_uniform",
            "min": 2,
            "max": 5,
        },
        # TODO implement support
        # "lr": {
        #     "distribution": "log_uniform_values",
        #     "min": 0.0001,
        #     "max": 0.01
        # }
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=SWEEP_NAME, entity='nyxai')
wandb.agent(sweep_id, function=main, count=1000)
