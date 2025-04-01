from enum import IntEnum
import shutil
import logging
import os
import tempfile
import numpy as np
from typing import Optional, Tuple
from tqdm import trange
from gym import ObservationWrapper


from torch_impl.helpers.rl_helpers import set_seed
from common.render import Renderer


logger = logging.getLogger(__name__)

class Action(IntEnum):
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    STAY = 4

    @classmethod
    def num_actions(cls) -> int:
        return len(cls)


class Object(IntEnum):
    SKYSCRAPER = 2
    STATION = 3
    DROPZONE = 4
    PACKET = 5


def convert_for_rendering(env) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    side_size = env.side_size
    ground = np.full((side_size, side_size), None)
    for (y, x) in env.dropzones.keys():
        ground[y, x] = Object.DROPZONE
    for (y, x) in env.stations.keys():
        ground[y, x] = Object.STATION
    for (y, x) in env.skyscrapers.keys():
        ground[y, x] = Object.SKYSCRAPER
    for (y, x) in env.packets.keys():
        ground[y, x] = Object.PACKET

    air = np.full((side_size, side_size), None)

    carrying_package = []
    charge = []
    # We sort the drones by index to ensure the order is consistent
    for pos, drone in sorted(env.drones.items(), key=lambda x: x[1].index):
        y, x = pos
        air[y, x] = drone.index
        carrying_package.append(drone.packet)
        charge.append(drone.charge)
    return ground, air, np.array(carrying_package), np.array(charge)


def render_video(
        env: ObservationWrapper,
        agents,
        num_steps: int = 200,
        output_path: str = './out.mp4',
        temp_dir: Optional[str] = None,
        fps: int = 3,
        resolution_scale_factor: float = 3,
        seed: int = 0):
    renderer = Renderer(env.n_drones, env.side_size, resolution_scale_factor=resolution_scale_factor)
    renderer.init()
    set_seed(env, seed)
    state = env.reset()

    # starting frame
    ground, air, carrying_package, charge = convert_for_rendering(env)
    img = renderer.render_frame(
        ground,
        air,
        carrying_package,
        charge,
        np.zeros(env.n_drones, dtype=np.float32),
        4 * np.ones(env.n_drones, dtype=np.int32) # STAY actions
        )

    if temp_dir is None:
        temp_dir = tempfile.mkdtemp()
    os.makedirs(temp_dir, exist_ok=True)
    renderer.save_frame(img, temp_dir)

    # setting agents to greedy
    for agent in agents.values():
        agent.is_greedy = True

    def _dict_to_np(input_dict, dtype = np.float32):
        num_items = len(input_dict)
        arr = num_items * [0]
        for k, val in input_dict.items():
            arr[k] = val
        return np.array(arr, dtype=dtype)

    logger.info('Generating video...')
    for step in trange(1, num_steps):
        actions = {key: agent.act(state[key]) for key, agent in agents.items()}
        state, rews, _, _, _ = env.step(actions)
        ground, air, carrying_package, charge = convert_for_rendering(env)
        img = renderer.render_frame(
            ground,
            air,
            carrying_package,
            charge,
            _dict_to_np(rews, dtype=np.float32),
            _dict_to_np(actions, dtype=np.int32)
            )
        renderer.save_frame(img, temp_dir)

    renderer.generate_video(temp_dir, output_path, output_resolution=img.size, fps=fps)
    shutil.rmtree(temp_dir)

    # reset agents
    for agent in agents.values():
        agent.is_greedy = False

    logger.info(f'Generated video {os.path.abspath(output_path)}')
