from typing import Tuple, Optional
import shutil
import os
import tempfile
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import trange
import logging

from common.render import Renderer
from jax_impl.env.env import DroneEnvParams, DeliveryDrones
from jax_impl.agents.dqn import DQNAgent, DQNAgentState

logger = logging.getLogger(__name__)


def convert_jax_state(state, actions, rewards) \
         -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ground = jax.device_get(state.ground)
    actions = jax.device_get(actions)
    rewards = jax.device_get(rewards)
    charge = jax.device_get(state.charge)
    air_x = jax.device_get(state.air_x)
    air_y = jax.device_get(state.air_y)
    air = np.zeros_like(ground, dtype=np.object_)
    air[:] = None
    air[air_y, air_x] = np.arange(air_x.size)
    carrying_package = jax.device_get(state.carrying_package)
    return ground, air, carrying_package, charge, rewards, actions


def render_video(
        env_params: DroneEnvParams,
        ag_state: DQNAgentState,
        num_steps: int = 200,
        output_path: str = './out.mp4',
        temp_dir: Optional[str] = None,
        fps: int = 3,
        resolution_scale_factor: float = 3,
        seed: int = 0):
    renderer = Renderer(env_params.n_drones, env_params.grid_size, resolution_scale_factor=resolution_scale_factor)
    renderer.init()
    rng = jax.random.PRNGKey(seed)
    env = DeliveryDrones()
    dqn_agent = DQNAgent()
    env_state = env.reset(rng, env_params)
    step_jit = jax.jit(env.step, static_argnums=(3,))
    act_jit = jax.jit(dqn_agent.act, static_argnums=(3,))
    get_obs_jit = jax.jit(env.get_obs, static_argnums=(1,))

    # starting frame
    img = renderer.render_frame(*convert_jax_state(env_state, jnp.array(env_params.n_drones * [4]), jnp.array(env_params.n_drones * [0.0])))

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    renderer.save_frame(img, temp_dir)

    logger.info('Generating video...')
    for step in trange(1, num_steps):
        rng, key = jax.random.split(rng)
        actions = jax.random.randint(key, (env_params.n_drones,), 0, 5, dtype=jnp.int32)
        obs = get_obs_jit(env_state, env_params)
        obs = obs[0].reshape(1, -1)
        dqn_action = act_jit(key, obs, ag_state, greedy=True)
        actions = actions.at[0].set(dqn_action)
        env_state, rewards, dones = step_jit(rng, env_state, actions, env_params)
        img = renderer.render_frame(*convert_jax_state(env_state, actions, rewards))
        renderer.save_frame(img, temp_dir)
    renderer.generate_video(temp_dir, output_path, output_resolution=img.size, fps=fps)
    shutil.rmtree(temp_dir)
    logger.info(f'Generated video {os.path.abspath(output_path)}')
