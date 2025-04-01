import numpy as np
import tqdm
import tempfile
from torch_impl.agents.dqn import BaseDQNFactory
from torch_impl.env.env import DeliveryDrones
from torch_impl.env.wrappers import WindowedGridView
from torch_impl.helpers.rl_helpers import set_seed
from common.render import Renderer
from common.constants import Action
from torch_impl.render_util import convert_for_rendering
from torch_impl.agents.random import RandomAgent
from pprint import pprint

# Setup the env
env_params = {
    'n_drones': 8,
    'charge_reward': -0.1,
    'crash_reward': -1,
    'delivery_reward': 1,
    'charge': 20,
    'discharge': 10,
    'drone_density': 0.05,
    'dropzones_factor': 2,
    'packets_factor': 3,
    'pickup_reward': 0,
    'rgb_render_rescale': 1.0,
    'skyscrapers_factor': 3,
    'stations_factor': 2
}
env = WindowedGridView(DeliveryDrones(env_params), radius=3)
set_seed(env, 0)
state = env.reset()

video_directory_path = tempfile.mkdtemp()
renderer = Renderer(
    env.n_drones,
    env.side_size,
    resolution_scale_factor=2.0
)
renderer.init()

# Hard-coded sequence of actions
actions = [
    Action.STAY,
    Action.DOWN,
    Action.RIGHT,
    Action.UP,
    Action.UP,
    Action.RIGHT,
    Action.UP,
    Action.UP,
    Action.UP,
    Action.UP,
]

agents = {drone.index: RandomAgent(env) for drone in env.drones_list}

for _step in (range(len(actions))):
    print("="*100)
    # Act randomly, except for the first drone
    _action_dictionary = {key: agent.act(None) for key, agent in agents.items()}
    _action_dictionary[0] = actions[_step]

    print(f"Step: {_step}")
    print(f"Actions: {_action_dictionary}")
    _, rewards, dones, _, _ = env.step(_action_dictionary)

    ground, air, carrying_package, charge = convert_for_rendering(env)
    # pprint(env.get_state())
    print(f"Rewards: {rewards}")
    print(f"Dones: {dones}")
    print(f"Charges: {charge}")
    print(f"Carrying package: {carrying_package}")

    if _step == 0:
        assert all(c == 90 for c in charge), "All drones should start with 90 charge after first step"
        assert list(carrying_package) == [0, 0, 1, 1, 1, 0, 0, 0], "Initial package state incorrect"

    if _step == 1:
        assert list(charge) == [80, 80, 80, 80, 80, 80, 80, 80], "Incorrect charge values at step 1"
        assert list(carrying_package) == [1, 0, 1, 1, 1, 0, 0, 0], "Drone 0 should pick up package"

    if _step == 2:
        assert list(charge) == [100, 70, 70, 70, 70, 70, 70, 70], "Incorrect charge values at step 2"
        assert list(carrying_package) == [1, 0, 1, 1, 1, 0, 0, 1], "Drone 7 should pick up package"

    if _step == 3:
        assert list(charge) == [90, 60, 100, 60, 60, 60, 100, 60], "Incorrect charge values at step 3"
        assert list(carrying_package) == [0, 0, 0, 1, 1, 0, 0, 1], "Drone 0 should deliver, D2&D6 crash"
        assert rewards[0] == 1, "Drone 0 should get delivery reward"
        assert rewards[2] == rewards[6] == -1, "Drones 2 and 6 should get crash penalty"

    if _step == 7:
        assert list(charge) == [50, 20, 60, 20, 20, 100, 60, 100], "Incorrect charge values at step 7"
        assert list(carrying_package) == [0, 0, 0, 1, 1, 0, 0, 0], "D7&D5 crash, D7 loses package"
        assert rewards[5] == rewards[7] == -1, "Drones 5 and 7 should get crash penalty"

    # Render
    _step_frame_im = renderer.render_frame(ground, air, carrying_package, charge, rewards, _action_dictionary)
    _step_frame_im.save("{}/{}.jpg".format(video_directory_path, str(_step).zfill(4)))
    print(env.render())

print(f"Video saved to {video_directory_path}")
