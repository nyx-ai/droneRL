import logging
from jax.experimental.compilation_cache import compilation_cache as cc
import wandb

from train_jax import train_jax, parse_args


wandb.login()
PROJECT_NAME = "dronerl"

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')


def main():
    wandb.init(project=PROJECT_NAME)
    args = parse_args()
    for k, v in wandb.config.items():
        setattr(args, k, v)
    args.wandb = True
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
            'max': 16,
        },
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME, entity='nyxai')
wandb.agent(sweep_id, function=main, count=1)
