import logging
import pprint
import statistics
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
        rng, env_states, obs, ag_state, bstate, step = carry

        # generate random actions for all drones in all envs
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (args.num_envs, env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent in all envs
        act_keys = jax.random.split(rng, args.num_envs)
        dqn_actions = jax.vmap(dqn_agent.act, in_axes=(0, 0, None))(act_keys, obs, ag_state)
        actions = actions.at[:, 0].set(dqn_actions)

        # perform actions in envs
        rng, key = jax.random.split(rng)
        env_step_keys = jax.random.split(key, args.num_envs)
        env_states, rewards, dones = jax.vmap(env.step, in_axes=(0, 0, 0, None))(env_step_keys, env_states, actions, env_params)
        next_obs = jax.vmap(env.get_obs, in_axes=(0, None))(env_states, env_params)
        next_obs = next_obs[:, 0, :, :].reshape(args.num_envs, -1)  # only consider obs from agent 0

        # add to buffer
        exps = {'obs': obs, 'actions': actions[:, 0], 'rewards': rewards[:, 0], 'next_obs': next_obs, 'dones': dones[:, 0]}
        bstate = buffer.add_many(bstate, exps)

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
        def _reset_envs(rng):
            reset_env_keys = jax.random.split(rng, args.num_envs)
            env_states = jax.vmap(env.reset, in_axes=(0, None))(reset_env_keys, env_params)
            next_obs = jax.vmap(env.get_obs, in_axes=(0, None))(env_states, env_params)
            next_obs = next_obs[:, 0, :, :].reshape(args.num_envs, -1)
            return env_states, next_obs

        env_states, next_obs = jax.lax.cond(
                step % args.reset_env_every == 0,
                _reset_envs,
                lambda _: (env_states, next_obs),
                rng
                )

        return (rng, env_states, next_obs, ag_state, bstate, step + 1), (rewards, ag_state.epsilon)

    # params
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
    logger.info(f'Training env:')
    pprint.pprint(env_params)
    if args.epsilon_decay is None:
        eps_decay = (1 - 0.5 * (1 - args.epsilon_end/args.epsilon_start))**(1/(args.epsilon_decay_half_life_fraction*args.num_steps))
    else:
        eps_decay = args.epsilon_decay
    ag_params = DQNAgentParams(
            hidden_layers=args.hidden_layers,
            target_update_interval=args.target_update_interval,
            epsilon_start=args.epsilon_start,
            epsilon_decay=eps_decay,
            epsilon_end=args.epsilon_end,
            epsilon_decay_every=args.epsilon_decay_every,
            gamma=args.gamma,
            learning_rate=args.learning_rate,
            tau=args.tau,
            )
    logger.info(f'Agent params:')
    pprint.pprint(ag_params)

    # init buffer
    buffer = ReplayBuffer(buffer_size=args.memory_size, sample_batch_size=args.batch_size)
    obs_size = 6 * (2 * args.window_radius + 1) **2
    exp = {
        'obs': jnp.zeros(obs_size,),
        'actions': jnp.array(0, dtype=jnp.int32),
        'rewards': jnp.array(0.0, dtype=jnp.float32),
        'next_obs': jnp.zeros(obs_size,),
        'dones': jnp.array(True, dtype=jnp.bool),
        }
    bstate = buffer.init(exp)

    # init states
    rng = jax.random.PRNGKey(args.seed)
    env = DeliveryDrones()
    reset_env_keys = jax.random.split(rng, args.num_envs)
    env_states = jax.vmap(env.reset, in_axes=(0, None))(reset_env_keys, env_params)
    dqn_agent = DQNAgent()
    ag_state = dqn_agent.reset(rng, ag_params, env_params)
    obs = jax.vmap(env.get_obs, in_axes=(0, None))(env_states, env_params)
    obs = obs[:, 0, :, :].reshape(args.num_envs, -1)

    # train
    carry = (rng, env_states, obs, ag_state, bstate, jnp.array(0))  # intial carry
    max_scan_steps = 100_000
    scan_steps = min(args.num_steps, max_scan_steps)
    num_iterations = math.ceil(args.num_steps / scan_steps)
    ts = timer()
    for _ in trange(num_iterations):
        carry, (rewards, epsilons) = jax.lax.scan(_train, carry, length=scan_steps)
    ag_state = carry[-3]
    rewards.block_until_ready()  # for accurate timing
    time_taken = timer() - ts
    logger.info(f'Trained {args.num_steps:,} steps with {args.num_envs:,} envs in {time_taken:.2f}s ({(args.num_envs * args.num_steps)/time_taken:,.0f} obs/s)')

    # evals
    logger.info(f'Running eval...')
    agent_eval, random_eval  = eval_jax(args, ag_state)
    logger.info(f'Mean eval reward: {agent_eval[0]:.3f} ± {agent_eval[1]:.3f} (random agent: {random_eval[0]:.3f} ± {random_eval[1]:.3f})')

    if args.render_video:
        print(f'Rendering video {args.video_output_file}...')
        render_video(env_params, ag_state, output_path=args.video_output_file, num_steps=args.render_video_steps)

def eval_jax(args: argparse.Namespace, ag_state):
    def _eval(carry, step):
        rng, state, ag_state = carry

        # get obs
        obs = env.get_obs(state, env_params)
        obs = obs[0].ravel()

        # generate random actions for all drones
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (env_params.n_drones,), minval=0, maxval=Action.num_actions())

        # run action for DQN agent
        dqn_action = dqn_agent.act(key, obs, ag_state, greedy=True)
        actions = actions.at[0].set(dqn_action)

        # perform actions in env
        state, rewards, dones = env.step(key, state, actions, env_params)
        return (rng, state, ag_state), rewards

    env_params = DroneEnvParams(
            n_drones=args.n_drones if args.eval_n_drones is None else args.eval_n_drones,
            grid_size=args.grid_size if args.eval_grid_size is None else args.eval_grid_size,
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
    env = DeliveryDrones()
    dqn_agent = DQNAgent()
    mean_rewards = []
    random_mean_rewards = []
    for i in trange(args.num_evals):
        rng = jax.random.PRNGKey(args.eval_seed + i)
        env_state = env.reset(rng, env_params)
        final_carry, rewards = jax.lax.scan(_eval, (rng, env_state, ag_state), jnp.arange(args.num_eval_steps))
        mean_rewards.append(jnp.mean(rewards[:, 0]).item())
        if env_params.n_drones > 1:
            random_mean_rewards.append(jnp.mean(rewards[:, 1]).item())
        else:
            random_mean_rewards.append(0.0)
    mean, std = statistics.mean(mean_rewards), statistics.stdev(mean_rewards)
    rmean, rstd = statistics.mean(random_mean_rewards), statistics.stdev(random_mean_rewards)
    return (mean, std), (rmean, rstd)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # experiment
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of steps to train")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    # env
    parser.add_argument("--n_drones", type=int, default=4, help="Number of drones")
    parser.add_argument("--grid_size", type=int, default=9, help="Size of the grid")
    parser.add_argument("--window_radius", type=int, default=3, help="Radius of observation window")
    parser.add_argument("--packets_factor", type=int, default=0, help="Number of packages relative to n_drones")
    parser.add_argument("--dropzones_factor", type=int, default=0, help="Number of dropzones relative to n_drones")
    parser.add_argument("--stations_factor", type=int, default=1, help="Number of charging stations relative to n_drones")
    parser.add_argument("--skyscrapers_factor", type=int, default=0, help="Number of skyscrapers relative to n_drones")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of envs to run. Increasing the number of envs will generate more experiences per training step.")
    # training
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
    # model
    parser.add_argument("--hidden_layers", nargs='+', type=int, default=[16, 16], help="Hidden layer sizes")
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
    # video
    parser.add_argument("--render_video", action='store_true', default=False, help="Whether to render a video at the end")
    parser.add_argument("--render_video_steps", type=int, default=200, help="Number of steps to render video for")
    parser.add_argument("--video_output_file", type=str, default='./jax_training_out.mp4', help="Number of steps to render video for")

    args = parser.parse_args()
    return args



if __name__ == "__main__":
    args = parse_args()
    # some validations
    if args.num_envs <= 0:
        raise ValueError(f'Number of envs need to be at least 1')
    if args.num_steps <= 0:
        raise ValueError(f'Number of steps need to be at least 1')
    # train!
    train_jax(args)
