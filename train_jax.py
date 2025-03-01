import logging
import argparse
import math
import jax
import jax.numpy as jnp
import jax.lax
import jax.random
from tqdm import trange
from jax.experimental.compilation_cache import compilation_cache as cc
from timeit import default_timer as timer

from jax_impl.env.env import DroneEnvParams, DeliveryDrones
from common.constants import Action
from jax_impl.agents.dqn import DQNAgent, DQNAgentParams
from jax_impl.buffers import ReplayBuffer
from jax_impl.render_util import render_video


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

cc.set_cache_dir('./jax_cache')


def train_jax(args: argparse.Namespace):
    @jax.jit
    def _train(carry, _):
        rng, env_state, obs, ag_state, bstate, step = carry

        rng, key = jax.random.split(rng)
        # generate random actions for all drones
        actions = jax.random.randint(key, (env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent
        rng, key = jax.random.split(rng)
        dqn_action = dqn_agent.act(key, obs, ag_state)
        actions = actions.at[0].set(dqn_action)

        # perform actions in env
        env_state, rewards, dones = env.step(key, env_state, actions, env_params)

        next_obs = env.get_obs(env_state, env_params)
        next_obs = next_obs[0].ravel()

        # add to buffer
        exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
        bstate = buffer.add(bstate, exp)

        # train step
        def train_if_can_sample(args):
            ag_state, bstate, key = args
            batch = buffer.sample(key, bstate)
            ag_state, loss = dqn_agent.train_step(ag_state, batch, ag_params)
            return ag_state, loss

        # jax.lax.cond(condition, true_function(*args), false_function(*args), *args)
        rng, key = jax.random.split(rng)
        ag_state, loss = jax.lax.cond(
                buffer.can_sample(bstate),
                train_if_can_sample,
                lambda x: (x[0], 0.0),
                (ag_state, bstate, key)
                )

        # update target network
        ag_state = jax.lax.cond(
                step % ag_params.target_update_interval == 0,
                lambda x: dqn_agent.update_target(x, ag_params),
                lambda x: x,
                ag_state
                )

        # update epsilon
        ag_state = jax.lax.cond(
                dqn_agent.should_update_epsilon(ag_params, step, dones[0]),
                lambda x: dqn_agent.update_epsilon(x, ag_params),
                lambda x: x,
                ag_state
                )

        # reset env
        def _reset_env(key):
            env_state = env.reset(key, env_params)
            next_obs = env.get_obs(env_state, env_params)
            next_obs = next_obs[0].ravel()
            return env_state, next_obs

        rng, key = jax.random.split(rng)
        env_state, next_obs = jax.lax.cond(
                step % args.reset_env_every == 0,
                _reset_env,
                lambda _: (env_state, next_obs),
                key
                )

        return (rng, env_state, next_obs, ag_state, bstate, step + 1), (rewards, ag_state.epsilon)

    env_params = DroneEnvParams(
            n_drones=args.n_drones,
            grid_size=args.grid_size,
            window_radius=args.window_radius,
            pickup_reward=args.pickup_reward,
            delivery_reward=args.delivery_reward,
            crash_reward=args.crash_reward,
            charge_reward=args.charge_reward,
            packets_factor=args.packets_factor,
            dropzones_factor=args.dropzones_factor,
            stations_factor=args.stations_factor,
            skyscrapers_factor=args.skyscrapers_factor,
            )
    ag_params = DQNAgentParams(
            hidden_layers=args.hidden_layers,
            target_update_interval=args.target_update_interval,
            epsilon_start=args.epsilon_start,
            epsilon_decay=args.epsilon_decay,
            epsilon_end=args.epsilon_end,
            epsilon_decay_every=args.epsilon_decay_every,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            tau=args.tau,
            )

    rng = jax.random.PRNGKey(args.seed)
    env = DeliveryDrones()
    env_state = env.reset(rng, env_params)
    dqn_agent = DQNAgent()
    ag_state = dqn_agent.reset(rng, ag_params, env_params)

    # init buffer
    buffer = ReplayBuffer(buffer_size=args.memory_size, sample_batch_size=args.batch_size)
    obs = env.get_obs(env_state, env_params)
    obs = obs[0].ravel()
    actions = jax.random.randint(rng, (env_params.n_drones,), minval=0, maxval=Action.num_actions())
    env_state, rewards, dones = env.step(rng, env_state, actions, env_params)
    next_obs = env.get_obs(env_state, env_params)
    next_obs = next_obs[0].ravel()
    exp = {'obs': obs, 'actions': actions[0], 'rewards': rewards[0], 'next_obs': next_obs, 'dones': dones[0]}
    bstate = buffer.init(exp)

    carry = (rng, env_state, next_obs, ag_state, bstate, jnp.array(0))  # intial carry
    max_scan_steps = 100_000
    scan_steps = min(args.num_steps, max_scan_steps)
    num_iterations = math.ceil(args.num_steps / scan_steps)
    ts = timer()
    for _ in trange(num_iterations):
        carry, (rewards, epsilons) = jax.lax.scan(_train, carry, length=scan_steps)
    rewards.block_until_ready()  # for accurate timing
    time_taken = timer() - ts
    logger.info(f'Trained {args.num_steps:,} steps in {time_taken:.2f}s ({args.num_steps/time_taken:.1f} steps/s)')

    if args.render_video:
        print(f'Rendering video {args.video_output_file}...')
        render_video(env_params, carry[-3], output_path=args.video_output_file, num_steps=args.render_video_steps)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment
    parser.add_argument("--num_steps", type=int, required=True, help="Number of steps to train")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    # env
    parser.add_argument("--n_drones", type=int, default=4, help="Number of drones")
    parser.add_argument("--grid_size", type=int, default=9, help="Size of the grid")
    parser.add_argument("--window_radius", type=int, default=3, help="Radius of observation window")
    parser.add_argument("--packets_factor", type=int, default=0, help="Number of packages relative to n_drones")
    parser.add_argument("--dropzones_factor", type=int, default=0, help="Number of dropzones relative to n_drones")
    parser.add_argument("--stations_factor", type=int, default=1, help="Number of charging stations relative to n_drones")
    parser.add_argument("--skyscrapers_factor", type=int, default=0, help="Number of skyscrapers relative to n_drones")
    # parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to run. Increasing the number of envs will generate more experiences per training step.")
    # training
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--memory_size", type=int, default=100_000, help="Size of replay memory")
    parser.add_argument("--epsilon_start", type=float, default=1.0, help="Starting epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.999, help="Decay rate for epsilon")
    parser.add_argument("--epsilon_end", type=float, default=0.01, help="Minimum epsilon value")
    parser.add_argument("--epsilon_decay_every", type=int, default=5, help="Steps between epsilon decay")
    parser.add_argument("--target_update_interval", type=int, default=10, help="Steps between target network updates")
    parser.add_argument("--gamma", type=float, default=0.9, help="Discount factor")
    parser.add_argument("--reset_env_every", type=int, default=100, help="Reset env every n training steps")
    parser.add_argument("--tau", type=float, default=1.0, help="Soft update parameter. A value of 1.0 corresponds to hard updates.")
    # model
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[16, 16], help="Hidden layer sizes")
    # rewards
    parser.add_argument("--pickup_reward", type=float, default=0.0, help="Reward for pickup")
    parser.add_argument("--delivery_reward", type=float, default=1.0, help="Reward for delivery")
    parser.add_argument("--crash_reward", type=float, default=-1.0, help="Reward for crashing")
    parser.add_argument("--charge_reward", type=float, default=-0.1, help="Reward for charging")
    # video
    parser.add_argument("--render_video", action='store_true', default=False, help="Whether to render a video at the end")
    parser.add_argument("--render_video_steps", type=int, default=200, help="Number of steps to render video for")
    parser.add_argument("--video_output_file", type=str, default='./jax_training_out.mp4', help="Number of steps to render video for")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    train_jax(args)
