import pytest
import numpy as np
import tempfile
import os

from torch_impl.env.env import DeliveryDrones
from torch_impl.agents.random import RandomAgent
from common.render import Renderer
from torch_impl.render_util import convert_for_rendering
def test_rendering(tmp_path):
    env_params = {'n_drones': 6}
    env = DeliveryDrones(env_params)
    env.reset()
    agents = {drone.index: RandomAgent(env) for drone in env.drones_list}

    renderer = Renderer(
        env.n_drones,
        env.side_size,
        resolution_scale_factor=2.0
    )
    renderer.init()

    artifacts_dir = os.getenv('ARTIFACTS_DIR', str(tmp_path))
    os.makedirs(artifacts_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        for step in range(60):
            actions = {key: agent.act(None) for key, agent in agents.items()}
            next_states, rewards, dones, info, _ = env.step(actions)
            ground, air, carrying_package, charge = convert_for_rendering(env)

            img = renderer.render_frame(ground, air, carrying_package, charge, rewards, actions)
            renderer.save_frame(img, temp_dir)

            assert img.size[0] > 0 and img.size[1] > 0

        output_path = renderer.generate_video(temp_dir, os.path.join(
            artifacts_dir, "render_test.mp4"), output_resolution=img.size, fps=1)
        print(f'Generated video in {os.path.abspath(artifacts_dir)}')
        assert os.path.exists(output_path)


if __name__ == "__main__":
    tmp_dir = tempfile.mkdtemp()
    test_rendering(tmp_dir)
