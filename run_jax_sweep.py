import logging
from jax.experimental.compilation_cache import compilation_cache as cc
import wandb

from train_jax import train_jax, parse_args


wandb.login()
PROJECT_NAME = "dronerl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

def main():
    wandb.init(project=PROJECT_NAME)
    args = parse_args()
    for k, v in wandb.config.items():
        setattr(args, k, v)
    if args.network_type == 'conv':
        # make it more beefy
        args.conv_layers = [
                {"kernel_size": 3, "out_channels": 16, "padding": 1, "stride": 1},
                {"kernel_size": 3, "out_channels": 32, "padding": 1, "stride": 1},
                ]
        args.conv_dense_layers = (128, 64)
    elif args.network_type == 'dense':
        args.hidden_layers = (256, 128, 64)
    args.target_update_interval = 100
    args.memory_size = 20_000
    args.render_video = True
    args.wandb = True
    args.eval_while_training = True
    score = train_jax(args)
    wandb.log({"mean_reward": score})


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "mean_reward"},
    "parameters": {
        "network_type": {
            'values': ['dense', 'conv']
        },
        "num_envs": {
            'min': 1,
            'max': 8,
        },
        "epsilon_end": {
            'min': 0.0,
            'max': 0.3,
        },
        "batch_size": {
            "values": [8, 16, 32, 64]
        },
        "learning_rate": {
            "values": [1e-3, 1e-4, 5e-4]
        },
        "reset_env_every": {
            "values": [10, 100, 1000, 10_000]
        },
        "num_steps": {
            "values": [1_000_000, 5_000_000]
        },
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME, entity='nyxai')
wandb.agent(sweep_id, function=main, count=1000)
