from typing import Tuple, Optional
import shutil
import torch
import logging
from tqdm import trange
import os
import tempfile
import numpy as np
from jax.env.render import Renderer
from env.env import DeliveryDrones
from env.wrappers import WindowedGridView
from agents.random import RandomAgent
from agents.dqn import DQNAgent, DenseQNetworkFactory

logger = logging.getLogger(__name__)


def convert_python_state(step, env, rewards, actions) \
        -> Tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # TODO: Currently this works for the old Python env
    ground = env.env.ground.grid.copy()
    ground = np.where(ground == 4, 5, ground)
    ground = np.where(ground == 3, 4, ground)
    ground = np.where(ground == 2, 3, ground)
    ground = np.where(ground == 1, 2, ground)
    air = np.zeros_like(ground)
    air[:] = None
    for _, position in env.air.get_objects(0, zip_results=True):
        drone = env.drone_data[position]
        air[position[0], position[1]] = drone.index
    carrying_package = np.array([a.packet is not None for a in env.drones], dtype=np.bool)
    charge = np.array([a.charge for a in env.drones], dtype=np.int32)
    actions = np.array([actions[k] for k in range(len(actions))], dtype=np.int32)
    rewards = np.array([rewards[k] for k in range(len(rewards))], dtype=np.int32)
    return step, ground, air, carrying_package, charge, rewards, actions


def render_video(
        model_path: str,
        n_drones: int = 3,
        num_steps: int = 300,
        output_path: str = './out.mp4',
        temp_dir: Optional[str] = None,
        drone_density: float = 0.03,
        fps: int = 2,
        ):
    # TODO: Currently this works for the old Python env
    env = WindowedGridView(DeliveryDrones(), radius=3)
    env.env_params.update({
        'n_drones': n_drones,
        'drone_density': drone_density,
        'charge_reward': -0.1,
        'crash_reward': -1,
        'pickup_reward': 0.1,
        'delivery_reward': 1,
        'charge': 20,
        'discharge': 10,
        'dropzones_factor': 2,
        'packets_factor': 3,
        'skyscrapers_factor': 3,
        'stations_factor': 2
    })
    states, info = env.reset()
    agents = {drone.index: RandomAgent(env) for drone in env.drones}
    agents[0] = DQNAgent(
        env=env,
        dqn_factory=DenseQNetworkFactory(env, hidden_layers=[32] * 2),
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_decay=0.999,
        epsilon_end=0.01,
        memory_size=10_000,
        batch_size=64,
        target_update_interval=5
    )
    agents[0].reset()
    agents[0].load(model_path)

    # renderer
    grid_size = env.env.ground.shape[0]
    renderer = Renderer(n_drones, grid_size, rgb_render_rescale=4)
    renderer.init()

    # starting frame
    # img = renderer.render_frame(*convert_python_state(0, env))
    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)

    for key, agent in agents.items():
        agent.is_greedy = True
    # renderer.save_frame(img, 0, temp_dir)


    logger.info('Generating video...')
    for step in trange(1, num_steps):
        with torch.no_grad():
            actions = {key: agent.act(states[key]) for key, agent in agents.items()}
        states, rewards, dones, _, _ = env.step(actions)

        img = renderer.render_frame(*convert_python_state(step, env, rewards, actions))
        renderer.save_frame(img, step, temp_dir)
    renderer.generate_video(temp_dir, output_path, output_resolution=img.size, fps=fps)
    shutil.rmtree(temp_dir)
    logger.info(f'Generated video {os.path.abspath(output_path)}')


if __name__ == "__main__":
    render_video('baseline_models/checkpoint.pt')
