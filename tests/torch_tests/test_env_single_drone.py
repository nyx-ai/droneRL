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

# Setup the env
env_params = {
    'n_drones': 1,
    'charge_reward': 0.0,
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
    Action.UP,
    Action.LEFT,
    Action.LEFT,
    Action.UP,
    Action.UP,
    Action.RIGHT,
    Action.UP,
    Action.RIGHT,
    Action.RIGHT,
    Action.DOWN,
    Action.DOWN,
    Action.DOWN,
    Action.DOWN,
    Action.DOWN,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
    Action.STAY,
]

total_reward = 0
for _step in (range(len(actions))):
    _action_dictionary = {}
    _idx = 0
    action = actions[_step]
    _action_dictionary[_idx] = action

    print(f"Step: {_step}")
    print(f"Actions: {_action_dictionary}")
    state, rewards, _, _, _ = env.step(_action_dictionary)
    _step_score = np.array(list(rewards.values()))
    total_reward += _step_score[0]

    ground, air, carrying_package, charge = convert_for_rendering(env)
    print(f"Charges: {charge}")
    print(f"Carrying package: {carrying_package}")

    if _step == 3:
        assert carrying_package[0] == 1, "Should pick up package at step 3"
        assert charge[0] == 80, "Charge should be 80 at step 3"
        assert rewards[0] == env_params['pickup_reward'], "Should get pickup reward"
    
    elif _step == 7:
        assert carrying_package[0] == 0, "Should deliver package at step 7"
        assert charge[0] == 70, "Charge should be 70 at step 7"
        assert rewards[0] == env_params['delivery_reward'], "Should get delivery reward"
    
    elif _step == 8:
        assert carrying_package[0] == 1, "Should pick up second package at step 8"
        assert rewards[0] == env_params['pickup_reward'], "Should get pickup reward"
    
    elif _step == 10:
        assert carrying_package[0] == 0, "Should deliver second package at step 10"
        assert rewards[0] == env_params['delivery_reward'], "Should get delivery reward"
    
    elif _step == 13:
        assert charge[0] == 100, "Should respawn with full charge after crash"
        assert rewards[0] == env_params['crash_reward'], "Should get crash penalty"
    
    elif _step == 23:
        assert charge[0] == 100, "Should respawn with full charge after battery death"
        assert rewards[0] == env_params['crash_reward'], "Should get crash penalty"

    # Render
    _step_frame_im = renderer.render_frame(ground, air, carrying_package, charge, rewards, _action_dictionary)
    _step_frame_im.save("{}/{}.jpg".format(video_directory_path, str(_step).zfill(4)))
    print(env.render())

print(f"Video saved to {video_directory_path}")
