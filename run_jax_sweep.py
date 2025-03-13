import logging
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
    args.memory_size = 100_000
    args.render_video = True
    args.wandb = True
    args.eval_while_training = True
    args.save_final_checkpoint = True
    args.use_sharding = args.num_envs > 1
    metrics = train_jax(args)
    wandb.log({"mean_reward": metrics['eval_reward_mean']})


sweep_configuration = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "mean_reward"},
    "parameters": {
        "network_type": {
            'values': ['dense', 'conv']
        },
        "num_envs": {
            "values": [1, 8, 16, 32]
        },
        "epsilon_end": {
            'min': 0.0,
            'max': 0.2,
        },
        "batch_size": {
            "values": [8, 16, 32, 64]
        },
        "learning_rate": {
            "values": [1e-4, 5e-4]
        },
        "reset_env_every": {
            "values": [100]
        },
        "num_steps": {
            "values": [5_000_000]
        },
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT_NAME, entity='nyxai')
wandb.agent(sweep_id, function=main, count=1000)
