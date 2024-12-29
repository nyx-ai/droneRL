import pytest
import jax
import jax.numpy as jnp
import jax.random

from env.env import DroneEnvState, DroneEnvParams, DeliveryDrones
from env.constants import Object, Action


##########
# FIXTURES
##########

@pytest.fixture
def single_drone_env():
    params = DroneEnvParams(n_drones=1, grid_size=8)
    air_x = jnp.array([3], dtype=jnp.int32)
    air_y = jnp.array([3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array(params.n_drones * [100], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def dual_drone_env():
    params = DroneEnvParams(n_drones=2, grid_size=8)
    air_x = jnp.array([1, 3], dtype=jnp.int32)
    air_y = jnp.array([3, 3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array(params.n_drones * [100], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def drone_env_packages():
    params = DroneEnvParams(n_drones=1, grid_size=8)
    air_x = jnp.array([3], dtype=jnp.int32)
    air_y = jnp.array([3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    ground = ground.at[3, 4].set(Object.PACKET)   # package to the right
    ground = ground.at[3, 5].set(Object.DROPZONE)   # dropzone to the right
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array(params.n_drones * [100], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def drone_env_skyscrapers():
    params = DroneEnvParams(n_drones=2, grid_size=8)
    air_x = jnp.array([3, 0], dtype=jnp.int32)
    air_y = jnp.array([3, 3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    ground = ground.at[3, 4].set(Object.SKYSCRAPER)   # skyscraper to the right
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array(params.n_drones * [100], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params

@pytest.fixture
def drone_env_charge():
    params = DroneEnvParams(n_drones=3, grid_size=8)
    air_x = jnp.array([3, 3, 0], dtype=jnp.int32)
    air_y = jnp.array([0, 3, 0], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    ground = ground.at[3, 4].set(Object.STATION)   # charging station to the right
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array([50, 50, 10], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def drone_respawn_behaviour():
    params = DroneEnvParams(n_drones=2, grid_size=8)
    air_x = jnp.array([1, 3], dtype=jnp.int32)
    air_y = jnp.array([3, 3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    ground = ground.at[2, 6].set(Object.PACKET)  # drone 0 will respawn here after collision
    carrying_package = jnp.zeros((params.n_drones,), dtype=jnp.bool)
    charge = jnp.array(params.n_drones * [100], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def drone_env_get_obs():
    params = DroneEnvParams(n_drones=2, grid_size=8)
    air_x = jnp.array([1, 3], dtype=jnp.int32)
    air_y = jnp.array([3, 3], dtype=jnp.int32)
    ground = jnp.zeros((params.grid_size, params.grid_size), dtype=jnp.int8)
    ground = ground.at[2, 6].set(Object.PACKET)
    ground = ground.at[6, 6].set(Object.PACKET)
    ground = ground.at[1, 3].set(Object.STATION)
    ground = ground.at[2, 3].set(Object.STATION)
    ground = ground.at[6, 3].set(Object.DROPZONE)
    ground = ground.at[0, 0].set(Object.SKYSCRAPER)
    carrying_package = jnp.array([True, False])
    charge = jnp.array([80, 60], dtype=jnp.int32)
    state = DroneEnvState(air_x=air_x, air_y=air_y, ground=ground, carrying_package=carrying_package, charge=charge)
    return state, params


@pytest.fixture
def simple_params():
    return DroneEnvParams(n_drones=3, grid_size=8, packets_factor=3, dropzones_factor=2, stations_factor=2, skyscrapers_factor=3)


#######
# TESTS
#######

def test_reset(simple_params):
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(4)
    state = env.reset(rng, simple_params)
    assert state.air_x.size == simple_params.n_drones
    num_packets = simple_params.packets_factor * simple_params.n_drones
    num_dropzones = simple_params.dropzones_factor * simple_params.n_drones
    num_stations = simple_params.stations_factor * simple_params.n_drones
    num_skyscrapers = simple_params.skyscrapers_factor * simple_params.n_drones
    assert jnp.sum(state.ground == Object.PACKET) == num_packets - 1  # 1 package has been picked up
    assert jnp.sum(state.ground == Object.STATION) == num_stations
    assert jnp.sum(state.ground == Object.SKYSCRAPER) == num_skyscrapers
    assert jnp.sum(state.ground == Object.DROPZONE) == num_dropzones
    assert jnp.sum(state.charge) == 100 * simple_params.n_drones
    assert state.carrying_package[0]
    assert not state.carrying_package[1]
    assert not state.carrying_package[2]


def test_respawn(drone_respawn_behaviour):
    state, params = drone_respawn_behaviour
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    actions = jnp.array([Action.RIGHT, Action.LEFT], dtype=jnp.int32)  # make them crash!
    assert jnp.sum(state.ground == Object.PACKET) == 1  # we have one package
    assert jnp.sum(state.carrying_package) == 0  # no one is carrying a package
    state_out, rewards, dones = env.step(rng, state, actions, params)
    assert jnp.sum(dones) == 2  # drones crashed
    assert jnp.sum(state_out.ground == Object.PACKET) == 0  # package was picked up
    assert jnp.sum(state_out.carrying_package) == 1  # no one is carrying a package
    assert jnp.sum(rewards) == -2  # we don't give a reward for this


def test_charge(drone_env_charge):
    state, params = drone_env_charge
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(1)
    actions = jnp.array([Action.RIGHT, Action.RIGHT, Action.RIGHT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state, actions, params)
    assert state_out.charge[0] == max(50 - params.discharge, 0)
    assert state_out.charge[1] == min(50 + params.charge, 100)
    assert state_out.charge[2] == 100  # died and respawned with 100% charge
    assert dones[2]
    assert rewards[0] == 0
    assert rewards[1] == params.charge_reward
    assert rewards[2] == params.crash_reward
    actions = jnp.array([Action.RIGHT, Action.STAY, Action.RIGHT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state_out, actions, params)
    assert state_out.charge[0] == max(50 - 2 * params.discharge, 0)
    assert state_out.charge[1] == min(50 + 2 * params.charge, 100)
    assert state_out.charge[2] == max(100 - params.discharge, 0)
    actions = jnp.array([Action.RIGHT, Action.STAY, Action.RIGHT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state_out, actions, params)
    assert state_out.charge[0] == max(50 - 3 * params.discharge, 0)
    assert state_out.charge[1] == min(50 + 3 * params.charge, 100)
    assert state_out.charge[2] == max(100 - 2 * params.discharge, 0)
    actions = jnp.array([Action.RIGHT, Action.DOWN, Action.RIGHT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state_out, actions, params)
    actions = jnp.array([Action.RIGHT, Action.STAY, Action.RIGHT], dtype=jnp.int32)
    assert state_out.charge[0] == max(50 - 4 * params.discharge, 0)
    if dones[1]:
        assert state_out.charge[1] == 100
    else:
        assert state_out.charge[1] == min(50 + 3 * params.charge, 100) - params.discharge
    if dones[2]:
        assert state_out.charge[2] == 100
    else:
        assert state_out.charge[2] == max(100 - 3 * params.discharge, 0)


def test_skyscrapers(drone_env_skyscrapers):
    state, params = drone_env_skyscrapers
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    actions = jnp.array([Action.RIGHT, Action.LEFT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state, actions, params)
    assert jnp.sum(dones) == 2  # both died
    assert jnp.sum(rewards) == params.crash_reward * 2
    assert state_out.ground[3, 4] == Object.SKYSCRAPER  # skyscraper hasn't moved


def test_packages(drone_env_packages):
    state, params = drone_env_packages
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    actions = jnp.array([Action.RIGHT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state, actions, params)
    # check that package has been picked up
    assert state_out.carrying_package[0]
    assert state_out.air_x == jnp.array([4])
    assert state_out.air_y == jnp.array([3])
    assert state_out.ground[3, 4] == 0
    assert rewards[0] == params.pickup_reward
    state_out2, rewards, dones = env.step(rng, state_out, actions, params)
    assert not state_out2.carrying_package[0]
    assert state_out2.air_x == jnp.array([5])
    assert state_out2.air_y == jnp.array([3])
    assert rewards[0] == params.pickup_reward + params.delivery_reward
    assert state_out2.ground[3, 5] == Object.DROPZONE
    assert jnp.sum(state_out2.ground > 0) == 2


def test_collisions(dual_drone_env):
    state, params = dual_drone_env
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    actions = jnp.array([Action.RIGHT, Action.LEFT], dtype=jnp.int32)
    state_out, rewards, dones = env.step(rng, state, actions, params)
    assert jnp.sum(dones) == 2 # both should be done
    assert jnp.sum(state_out.charge) == 200 # both should again have full charge after respawn


def test_single_movements(single_drone_env):
    state, params = single_drone_env
    env = DeliveryDrones()
    rng = jax.random.PRNGKey(0)
    # start (3, 3) | left, down, right, up, stay
    for action, (x, y) in zip([0, 1, 2, 3, 4], [(2, 3), (3, 4), (4, 3), (3, 2), (3, 3)]):
        actions = jnp.array([action], dtype=jnp.int32)
        state_out, rewards, dones = env.step(rng, state, actions, params)
        assert state_out.air_x == jnp.array([x], dtype=jnp.int32)
        assert state_out.air_y == jnp.array([y], dtype=jnp.int32)


def test_get_obs(drone_env_packages):
    state, params = drone_env_packages
    env = DeliveryDrones()
    for radius in [2, 3, 4]:
        obs = env.get_obs(state, params)
        assert obs.shape == (1, radius * 2 + 1, radius * 2 + 1, 6)
        assert obs[0, radius, radius, 0] == 1
        assert obs[0, radius, radius + 1, 1] == 1
        assert obs[0, radius, radius + 2, 2] == 1


@pytest.mark.focus
def test_get_obs_v2(drone_env_get_obs):
    state, params = drone_env_get_obs
    env = DeliveryDrones()
    radius = 3
    obs = env.get_obs(state, params)
    assert obs[0, radius, radius, 0] == 1
    assert obs[0, radius, radius + 2, 0] == 1
    assert obs[0, radius, radius, 1] == 1  # carrying package
    assert obs[1, radius, radius, 1] == 0  # not carrying package
    assert obs[1, 2, 6, 1] == 1
    assert obs[1, 6, 6, 1] == 1
    assert obs[0, radius, radius, 4] == .8  # charge
    assert obs[1, radius, radius, 4] == .6
    assert obs[0, 0, 2, 5] == 1  # skyscraper
    assert jnp.all(obs[0, :, :2, 5] == 1)  # walls
