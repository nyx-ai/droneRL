import statistics
import wandb
from tqdm import tqdm
from torch_impl.agents.dqn import DQNAgent, DenseQNetworkFactory
from torch_impl.agents.random import RandomAgent
from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.helpers.rl_helpers import test_agents

wandb.login()

SWEEP_NAME = "dronerl-torch-dense"
NUM_TRAINING_STEPS = 50_000
NUM_TESTING_STEPS = 50_000


def evaluate(config):
    # Testing environment setup
    testing_env = WindowedGridView(DeliveryDrones(), radius=3)
    testing_env.env_params.update({
        'n_drones': 6,
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

    # Training environment setup
    training_env = WindowedGridView(DeliveryDrones(), radius=3)
    training_env.env_params.update({
        'n_drones': config.n_drones,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'pickup_reward': config.pickup_reward,
        'delivery_reward': 1,
        'charge': 20,
        'discharge': 10,
        'drone_density': 0.05,
        'dropzones_factor': 2,
        'packets_factor': 3,
        'skyscrapers_factor': 3,
        'stations_factor': 2
    })

    # Training setup
    training_env.reset()
    agents = {drone.index: RandomAgent(training_env) for drone in training_env.drones_list}
    agents[0] = DQNAgent(
        env=training_env,
        dqn_factory=DenseQNetworkFactory(
            training_env.observation_space.shape,
            (training_env.action_space.n,),
            hidden_layers=(config.size_layers,) * config.num_layers,
        ),
        gamma=config.gamma,
        epsilon_start=1.0,
        epsilon_decay=config.epsilon_decay,
        epsilon_end=0.01,
        memory_size=config.memory_size,
        batch_size=config.batch_size,
        target_update_interval=config.target_update_interval
    )

    # Training
    states = training_env.reset()
    for _ in tqdm(range(NUM_TRAINING_STEPS), desc="Training"):
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}
        next_states, rewards, dones, _, _ = training_env.step(actions)
        for key, agent in agents.items():
            agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])
        states = next_states

    # Testing
    testing_env.reset()
    testing_agents = {drone.index: RandomAgent(testing_env) for drone in testing_env.drones_list}
    testing_agents[0] = agents[0]
    rewards_log = test_agents(testing_env, testing_agents, n_steps=NUM_TESTING_STEPS, seed=0)
    rewards = {k: statistics.mean(v) for k, v in rewards_log.items()}
    print(f"Score: {rewards[0]}")
    return rewards[0]


def main():
    wandb.init(project=SWEEP_NAME)
    score = evaluate(wandb.config)
    wandb.log({"score": score})


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "score"},
    "parameters": {
        "pickup_reward": {
            "values": [0.0, 0.1, 0.5, 1.0]
        },
        "size_layers": {
            "values": [1, 2, 4, 8, 16, 32, 64, 128]
        },
        "num_layers": {
            "values": [1, 2, 3, 4]
        },
        "gamma": {
            "values": [0.9, 0.95, 0.99, 0.995, 0.999]
        },
        "epsilon_decay": {
            "values": [0.9, 0.95, 0.99, 0.995, 0.999]
        },
        "target_update_interval": {
            "values": [1, 10, 100]
        },
        "batch_size": {
            "values": [8, 16, 32, 64, 128]
        },
        "n_drones": {
            "values": [2, 4, 8]
        },
        "memory_size": {
            "values": [1_000, 10_000, 100_000]
        },
    },
}

if __name__ == "__main__":
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=SWEEP_NAME, entity='nyxai')
    wandb.agent(sweep_id, function=main, count=1000)
