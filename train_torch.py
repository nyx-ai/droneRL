import logging
import ast
import copy
import json
from datetime import datetime
import os
import pprint
import statistics
import argparse
import math
import jax
import jax.numpy as jnp
import jax.lax
import jax.random
from tqdm import trange
from timeit import default_timer as timer
import wandb

from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.agents.dqn import DQNAgent, DenseQNetworkFactory, ConvQNetworkFactory
from torch_impl.agents.random import RandomAgent
from torch_impl.helpers.rl_helpers import set_seed
from torch_impl.render_util import render_video



logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


def train_torch(args: argparse.Namespace):
    # env
    env_params = {
        'pickup_reward': args.pickup_reward,
        'delivery_reward': args.delivery_reward,
        'crash_reward': args.crash_reward,
        'charge_reward': args.charge_reward,
        'packets_factor': args.packets_factor,
        'dropzones_factor': args.dropzones_factor,
        'stations_factor': args.stations_factor,
        'skyscrapers_factor': args.skyscrapers_factor,
        'n_drones': args.n_drones,
        'drone_density': args.n_drones / (args.grid_size ** 2)
    }
    logger.info(f'Training env:')
    pprint.pprint(env_params)
    env = WindowedGridView(
            DeliveryDrones(env_params=env_params),
            radius=args.window_radius)
    assert env.side_size == args.grid_size
    assert env.n_drones == args.n_drones
    set_seed(env, args.seed)
    states = env.reset()

    # agents
    agents = {drone.index: RandomAgent(env) for drone in env.drones_list}
    if args.network_type == 'dense':
        factory = DenseQNetworkFactory(
            env.observation_space.shape,
            (env.action_space.n,),
            hidden_layers=args.hidden_layers,
            learning_rate=args.learning_rate
        )
    elif args.network_type == 'conv':
        factory = ConvQNetworkFactory(
            env.observation_space.shape,
            (env.action_space.n,),
            conv_layers=args.conv_layers,
            dense_layers=args.conv_dense_layers,
            learning_rate=args.learning_rate
        )
    else:
        raise Exception(f'Unexpected network type {args.network_type}')
    if args.epsilon_decay is None:
        eps_decay = (1 - 0.5 * (1 - args.epsilon_end/args.epsilon_start))**(1/(args.epsilon_decay_half_life_fraction*args.num_steps))
    else:
        eps_decay = args.epsilon_decay
    agents[0] = DQNAgent(
        env=env,
        dqn_factory=factory,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_decay=eps_decay,
        epsilon_end=args.epsilon_end,
        epsilon_decay_every=args.epsilon_decay_every,
        memory_size=args.memory_size,
        batch_size=args.batch_size,
        target_update_interval=args.target_update_interval,
    )
    logger.info(f'Full config:')
    pprint.pprint(args)

    # W&B
    if args.wandb:
        wandb.login()
        wandb.init(
                project=args.wandb_project,
                group=args.wandb_group,
                entity=args.wandb_entity,
                config=vars(args))

    # training setup
    now_str =  datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join('output', f'torch_run_{now_str}')
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f'Training output will be saved in {run_dir}...')
    ts = timer()

    # train
    for step in trange(args.num_steps):
        # take action given obs
        actions = {key: agent.act(states[key]) for key, agent in agents.items()}

        # take step in env
        next_states, rewards, dones, _, _ = env.step(actions)

        # learn given new obs
        for key, agent in agents.items():
            agent.learn(states[key], actions[key], rewards[key], next_states[key], dones[key])

        # reset env every once in a while
        if step % args.reset_env_every == 0:
            next_states = env.reset()

        states = next_states

        if args.wandb and step % 100 == 0:
            wandb.log({'epsilon': agents[0].epsilon})

        if args.eval_while_training and step < args.num_steps - 1 and step % args.eval_every == 0:
            agent_eval, random_eval = eval_torch(args, agents)
            logger.info(f'Mean eval reward step {step:,}: {agent_eval[0]:.3f} ± {agent_eval[1]:.3f} (random agent: {random_eval[0]:.3f} ± {random_eval[1]:.3f})')
            if args.wandb:
                wandb.log({'eval_reward': agent_eval[0], 'random_reward': random_eval[0]}, step=step)

    time_taken = timer() - ts
    logger.info(f'Trained {args.num_steps:,} steps with {args.num_envs:,} envs in {time_taken:.2f}s ({(args.num_envs * args.num_steps)/time_taken:,.0f} obs/s)')
    if args.save_final_checkpoint:
        f_name = os.path.join(run_dir, f'agent_{args.num_steps}_steps_jax.safetensors')
        logger.info(f'Saving torch checkpoint to {f_name}...')
        agents[0].save(f_name)

    # final eval
    logger.info(f'Running final eval...')
    agent_eval, random_eval = eval_torch(args, agents)
    logger.info(f'Final mean eval reward: {agent_eval[0]:.3f} ± {agent_eval[1]:.3f} (random agent: {random_eval[0]:.3f} ± {random_eval[1]:.3f})')
    if args.wandb:
        wandb.log({'eval_reward': agent_eval[0], 'random_reward': random_eval[0]}, step=args.num_steps)

    # video
    if args.render_video:
        f_out = os.path.join(run_dir, f'training_{args.num_steps}_steps.mp4')
        logger.info(f'Rendering video {f_out}...')
        render_video(env, agents, output_path=f_out, num_steps=args.render_video_steps)
        if args.wandb:
            logger.info(f'Logging video to W&B...')
            wandb.log({"eval_video": wandb.Video(f_out, format="mp4")}, step=args.num_steps)

    return agent_eval[0]

def eval_torch(args: argparse.Namespace, agents):
    grid_size = args.grid_size if args.eval_grid_size is None else args.eval_grid_size
    n_drones = args.n_drones if args.eval_n_drones is None else args.eval_n_drones
    env_params = {
        'pickup_reward': args.pickup_reward,
        'delivery_reward': args.delivery_reward,
        'crash_reward': args.crash_reward,
        'charge_reward': args.charge_reward,
        'packets_factor': args.packets_factor,
        'dropzones_factor': args.dropzones_factor,
        'stations_factor': args.stations_factor,
        'skyscrapers_factor': args.skyscrapers_factor,
        'n_drones': n_drones,
        'drone_density': n_drones / (grid_size ** 2)
    }
    env = WindowedGridView(
            DeliveryDrones(env_params=env_params),
            radius=args.window_radius)
    assert env.side_size == grid_size
    assert env.n_drones == args.n_drones

    mean_rewards = []
    random_mean_rewards = []
    for k, agent in agents.items():
        agent.is_greedy = True
    for i in trange(args.num_evals):
        set_seed(env, args.seed)
        state = env.reset()
        rewards = []
        rand_rewards = []
        for step in range(args.num_eval_steps):
            actions = {key: agent.act(state[key]) for key, agent in agents.items()}
            state, rews, _, _, _ = env.step(actions)
            rewards.append(rews[0])
            if n_drones > 1:
                rand_rewards.append(rews[1])
            else:
                rand_rewards.append(0.0)
        mean_rewards.append(statistics.mean(rewards))
        random_mean_rewards.append(statistics.mean(rand_rewards))
    mean, std = statistics.mean(mean_rewards), statistics.stdev(mean_rewards)
    rmean, rstd = statistics.mean(random_mean_rewards), statistics.stdev(random_mean_rewards)

    # reset agents
    for k, agent in agents.items():
        agent.is_greedy = False
    return (mean, std), (rmean, rstd)


def parse_args():
    def _parse_conv_layers(value: str):
        try:
            layers = json.loads(value)
        except json.JSONDecodeError:
            try:
                layers = ast.literal_eval(value)
                if isinstance(layers, dict):
                    return (layers,)
                return tuple(layers)
            except (SyntaxError, ValueError):
                raise argparse.ArgumentTypeError(f"Invalid format for conv_layers: {value}.")
        return tuple(layers)

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # env
    parser.add_argument("--n_drones", type=int, default=4, help="Number of drones")
    parser.add_argument("--grid_size", type=int, default=9, help="Size of the grid")
    parser.add_argument("--window_radius", type=int, default=3, help="Radius of observation window")
    parser.add_argument("--packets_factor", type=int, default=3, help="Number of packages relative to n_drones")
    parser.add_argument("--dropzones_factor", type=int, default=2, help="Number of dropzones relative to n_drones")
    parser.add_argument("--stations_factor", type=int, default=2, help="Number of charging stations relative to n_drones")
    parser.add_argument("--skyscrapers_factor", type=int, default=3, help="Number of skyscrapers relative to n_drones")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to run. Increasing the number of envs will generate more experiences per training step.")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    # training
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to train")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--memory_size", type=int, default=100_000, help="Size of replay memory")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=None, help="Decay rate for epsilon. If not set, decay will be adjusted based on number of steps")
    parser.add_argument("--epsilon_decay_half_life_fraction", type=float, default=0.2, help="Fraction of training steps after which epsilon should be 50% of initial value")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon value")
    parser.add_argument("--epsilon_decay_every", type=int, default=5, help="Steps between epsilon decay")
    parser.add_argument("--target_update_interval", type=int, default=10, help="Steps between target network updates")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--reset_env_every", type=int, default=100, help="Reset env every n training steps")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update parameter. A value of 1.0 corresponds to hard updates.")
    parser.add_argument("--save_final_checkpoint", action='store_true', default=False, help="Whether to save a final checkpoint")
    # model
    parser.add_argument("--network_type", choices=['dense', 'conv'], default='dense', help="DQN network type")
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=(16, 16), help="Dense network hidden layer sizes")
    parser.add_argument('--conv_layers', type=_parse_conv_layers, default=('[{"kernel_size": 3, "out_channels": 8, "padding": 1, "stride": 1}]'), help='ConvNet network config (expects stringified JSON)')
    parser.add_argument("--conv_dense_layers", nargs='+', type=int, default=(), help="ConvNet additional hidden layer sizes")
    # rewards
    parser.add_argument("--pickup_reward", type=float, default=0.0, help="Reward for pickup")
    parser.add_argument("--delivery_reward", type=float, default=1.0, help="Reward for delivery")
    parser.add_argument("--crash_reward", type=float, default=-1.0, help="Reward for crashing")
    parser.add_argument("--charge_reward", type=float, default=-0.1, help="Reward for charging")
    # eval
    parser.add_argument("--eval_n_drones", type=int, default=None, help="Number of drones in eval env. By default same as training.")
    parser.add_argument("--eval_grid_size", type=int, default=None, help="Size of the grid in eval env. By default same as training.")
    parser.add_argument("--eval_seed", type=int, default=0, help="Eval seed")
    parser.add_argument("--num_eval_steps", type=int, default=10_000, help="Num eval steps")
    parser.add_argument("--num_evals", type=int, default=5, help="Number of evaluations")
    parser.add_argument("--eval_while_training", action='store_true', default=False, help="Whether to run eval while training")
    parser.add_argument("--eval_every", type=int, default=10_000, help="Whether to run eval while training")
    # video
    parser.add_argument("--render_video", action='store_true', default=False, help="Whether to render a video at the end")
    parser.add_argument("--render_video_steps", type=int, default=200, help="Number of steps to render video for")
    # W&B
    parser.add_argument("--wandb", action='store_true', default=False, help="Whether to log to W&B")
    parser.add_argument("--wandb_project", type=str, default="dronerl", help="W&B project name")
    parser.add_argument("--wandb_entity", type=str, default='nyxai', help="W&B entity")
    parser.add_argument("--wandb_group", type=str, default=None, help="W&B group for organizing related runs")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # some validations
    if args.num_envs > 1:
        raise ValueError(f'Currently the torch implementation only supports running one environment')
    if args.num_steps <= 0:
        raise ValueError(f'Number of steps need to be at least 1')

    # train!
    train_torch(args)
