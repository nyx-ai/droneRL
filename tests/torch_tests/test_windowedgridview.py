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
import pytest


def debug_state(env, state):
    print("="*100)
    print(env.render())

    print("Drones (including self)")
    print(state[0][:, :, 0])

    print("Packets (including those carried by drones)")
    print(state[0][:, :, 1])

    print("Dropzones")
    print(state[0][:, :, 2])

    print("Stations")
    print(state[0][:, :, 3])

    print("Charges (including for other drones)")
    print(state[0][:, :, 4])

    print("Skyscrapers")
    print(state[0][:, :, 5])


def test_windowed_grid_view_states():
    env_params = {
        'n_drones': 2,
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

    state, _, _, _, _ = env.step({0: Action.UP, 1: Action.STAY})
    debug_state(env, state)

    # Step 0 verification - all layers
    assert np.array_equal(state[0][:, :, 0], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 1], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 2], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 3], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 1., 0.]
    ]))

    assert np.allclose(state[0][:, :, 4], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.9, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0.9, 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]), atol=0.01)

    assert np.array_equal(state[0][:, :, 5], np.array([
        [1., 1., 1., 1., 1., 1., 1.],
        [0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1.],
        [0., 1., 0., 0., 0., 0., 1.],
        [0., 0., 1., 0., 1., 0., 1.],
        [0., 0., 1., 1., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0., 1.]
    ]))

    state, _, _, _, _ = env.step({0: Action.RIGHT, 1: Action.STAY})
    print(env.render())
    debug_state(env, state)

    # Step 1 verification - all layers
    assert np.array_equal(state[0][:, :, 0], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 1], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 2], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 3], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 1., 0., 0.]
    ]))

    assert np.allclose(state[0][:, :, 4], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.8, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0.8, 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]), atol=0.01)

    assert np.array_equal(state[0][:, :, 5], np.array([
        [1., 1., 1., 1., 1., 1., 1.],
        [1., 0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1.],
        [1., 0., 0., 0., 0., 1., 1.],
        [0., 1., 0., 1., 0., 1., 1.],
        [0., 1., 1., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1., 1.]
    ]))

    state, _, _, _, _ = env.step({0: Action.RIGHT, 1: Action.STAY})
    debug_state(env, state)

    # Step 2 verification - all layers
    assert np.array_equal(state[0][:, :, 0], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 1], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 2], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]))

    assert np.array_equal(state[0][:, :, 3], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 1., 0., 0., 0., 0., 0.],
        [1., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 1., 0., 0., 0.]
    ]))

    assert np.allclose(state[0][:, :, 4], np.array([
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0.7, 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0.]
    ]), atol=0.01)

    assert np.array_equal(state[0][:, :, 5], np.array([
        [1., 1., 1., 1., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1.],
        [1., 0., 1., 0., 1., 1., 1.],
        [1., 1., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1., 1.]
    ]))


if __name__ == "__main__":
    test_windowed_grid_view_states()
