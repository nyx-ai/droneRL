import pytest
import numpy as np
import tempfile
import os

from torch_impl.env.env import DeliveryDrones
from torch_impl.agents.random import RandomAgent
from common.render import Renderer
from enum import IntEnum


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


def convert_states_for_rendering(state, side_size):
    ground = np.full((side_size, side_size), None)
    for (y, x) in state['dropzones'].keys():
        ground[y, x] = Object.DROPZONE
    for (y, x) in state['stations'].keys():
        ground[y, x] = Object.STATION
    for (y, x) in state['skyscrapers'].keys():
        ground[y, x] = Object.SKYSCRAPER
    for (y, x) in state['packets'].keys():
        ground[y, x] = Object.PACKET

    air = np.full((side_size, side_size), None)
    carrying_package = []
    charge = []
    for drone_idx, ((y, x), drone) in enumerate(state['drones'].items()):
        air[y, x] = drone_idx
        if drone.packet:
            carrying_package.append(1)
        else:
            carrying_package.append(0)
        charge.append(drone.charge)
    return ground, air, carrying_package, charge


def test_rendering(tmp_path):
    env_params = {'n_drones': 8}
    env = DeliveryDrones(env_params)
    env.reset()
    agents = {drone.index: RandomAgent(env) for drone in env.drones}

    renderer = Renderer(
        env.n_drones,
        env.side_size,
        resolution_scale_factor=2.0
    )

    artifacts_dir = os.getenv('ARTIFACTS_DIR', str(tmp_path))
    os.makedirs(artifacts_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        renderer.init()

        # Test a few frames
        for step in range(500):
            actions = {key: agent.act(None) for key, agent in agents.items()}
            next_states, rewards, dones, info, _ = env.step(actions)
            ground, air, carrying_package, charge = convert_states_for_rendering(next_states, env.side_size)

            img = renderer.render_frame(ground, air, carrying_package, charge, rewards, actions)
            renderer.save_frame(img, temp_dir)

            # Basic assertions
            assert img.size[0] > 0 and img.size[1] > 0

        output_path = renderer.generate_video(temp_dir, os.path.join(
            artifacts_dir, "render_test.mp4"), output_resolution=img.size, fps=30)
        print(f'Generated video in {os.path.abspath(artifacts_dir)}')
        assert os.path.exists(output_path)


if __name__ == "__main__":
    tmp_dir = tempfile.mkdtemp()
    test_rendering(tmp_dir)
